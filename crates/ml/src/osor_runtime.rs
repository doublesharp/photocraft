//! Synchronous bounded OSOR bridge. The caller's process-wide lease drains every forward.
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use osor::{DType, InferenceInput, InferenceLimits, LoadOptions, Osor, Tensor};

use crate::resample::Geometry;
use crate::store::osor_reference::{MAX_DECODED_WEIGHTS, Reference};
use crate::{Error, OsorBackend, RemovalControl, RemovalInput, RemovalOutput, Result, check};

pub(crate) struct Cached {
    identity: Vec<u8>,
    backend: String,
    model: Osor,
}

pub(crate) fn clear_cache() {
    crate::cache::clear_model("osor-sdxl");
}

fn error(error: osor::Error) -> Error {
    match error {
        osor::Error::Cancelled => Error::Cancelled,
        osor::Error::Unavailable(reason) => Error::Inference(format!("OSOR backend unavailable: {reason}")),
        other => Error::Inference(format!("OSOR: {other}")),
    }
}

fn load(reference: &Reference, backend: OsorBackend) -> Result<(Cached, f64)> {
    let identity = serde_json::to_vec(&reference.manifest).map_err(|e| Error::Integrity(e.to_string()))?;
    let backend_key = format!("{backend:?}");
    let old = crate::cache::take_osor();
    if let Some(old) = old {
        if old.identity == identity && old.backend == backend_key {
            return Ok((old, 0.));
        }
        drop(old);
    }
    let start = Instant::now();
    let model = Osor::load_with_options_and_manifest_sha256_and_limits(
        &reference.path,
        LoadOptions { backend, dtype: DType::F32, limits: InferenceLimits { max_batch: 1, max_pixels: 512 * 512 } },
        reference.manifest_sha256()?,
        &osor::artifact::ArtifactLimits {
            max_file_bytes: 16 * 1024 * 1024 * 1024,
            max_total_tensor_bytes: MAX_DECODED_WEIGHTS,
            ..osor::artifact::ArtifactLimits::default()
        },
    )
    .map_err(error)?;
    // The native loader hashes the exact bytes parsed into tensors against its actual
    // parsed manifest. Comparing that manifest to the pinned receipt before forwarding
    // rejects a replaced manifest even if its tensors were consistently substituted.
    if serde_json::to_vec(model.manifest()).map_err(|e| Error::Integrity(e.to_string()))? != identity {
        model.device().synchronize().map_err(|e| Error::Inference(e.to_string()))?;
        return Err(Error::Integrity("OSOR manifest no longer matches its imported reference".into()));
    }
    Ok((Cached { identity, backend: backend_key, model }, start.elapsed().as_secs_f64() * 1000.))
}

pub(crate) fn infer(reference: &Reference, backend: OsorBackend, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
    check(ctl)?;
    ctl.progress(0.08, "Loading local OSOR model");
    let (cached, load_ms) = load(reference, backend)?;
    let result = forward(&cached.model, input, ctl);
    // Metal may enqueue device work. Drain it even if a stage observer cancels or fails;
    // admission and the model remain owned until synchronization completes.
    let drained = cached.model.device().synchronize().map_err(|e| Error::Inference(e.to_string()));
    crate::cache::put_osor(cached);
    drained?;
    let mut output = result?;
    output.load_ms = load_ms;
    check(ctl)?;
    Ok(output)
}

struct CancelMonitorCompletion<'a> {
    finished: &'a AtomicBool,
    monitor: std::thread::Thread,
}

impl Drop for CancelMonitorCompletion<'_> {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.monitor.unpark();
    }
}

fn with_cancellation<T>(ctl: &dyn RemovalControl, run: impl FnOnce(&osor::CancellationToken) -> Result<T>) -> Result<T> {
    check(ctl)?;
    let cancellation = osor::CancellationToken::new();
    let finished = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // The app owns its cancellation flag; the library checks its token inside blocks.
        // Keep one bounded monitor for this forward, including while no observer is called.
        let monitor = std::thread::Builder::new().name("photocraft-osor-cancel".into()).spawn_scoped(scope, || {
            while !finished.load(Ordering::Acquire) {
                if ctl.cancelled() {
                    cancellation.cancel();
                    return;
                }
                std::thread::park_timeout(Duration::from_millis(10));
            }
        })?;
        // Also wake the monitor during unwinding, so a backend panic cannot strand scope's join.
        let completion = CancelMonitorCompletion { finished: &finished, monitor: monitor.thread().clone() };
        let result = run(&cancellation);
        drop(completion);
        monitor.join().map_err(|_| Error::Inference("the local removal cancellation monitor failed".into()))?;
        result
    })
}

fn forward(model: &Osor, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
    check(ctl)?;
    let geometry = Geometry::osor(input.width, input.height)?;
    let (rgb, mask) = geometry.prepare(input)?;
    let image = Tensor::from_vec(rgb, (1, 3, geometry.rows, geometry.side), model.device()).map_err(|e| Error::Inference(e.to_string()))?;
    let mask = Tensor::from_vec(mask, (1, 1, geometry.rows, geometry.side), model.device()).map_err(|e| Error::Inference(e.to_string()))?;
    let latent_count = 4 * (geometry.rows / 8) * (geometry.side / 8);
    // Frozen internal seed and distinct streams make repeats stable without reading a
    // process environment or changing the existing RemovalInput API.
    let posterior = Tensor::from_vec(noise(latent_count, 0x5048_4f54_4f43_5241), (1, 4, geometry.rows / 8, geometry.side / 8), model.device())
        .map_err(|e| Error::Inference(e.to_string()))?;
    let diffusion = Tensor::from_vec(noise(latent_count, 0x4654_4f53_4f52_0001), (1, 4, geometry.rows / 8, geometry.side / 8), model.device())
        .map_err(|e| Error::Inference(e.to_string()))?;
    let started = Instant::now();
    let mut stage = 0usize;
    let result = with_cancellation(ctl, |cancellation| {
        model
            .infer_with_cancellation(
                InferenceInput { image: &image, mask: &mask, posterior_noise: &posterior, diffusion_noise: &diffusion },
                cancellation,
                &mut |_, _| {
                    if ctl.cancelled() {
                        return Err(osor::Error::Cancelled);
                    }
                    stage = stage.saturating_add(1);
                    ctl.progress(0.2 + 0.7 * stage as f32 / osor::pipeline::STAGES.len() as f32, "Removing object and effects locally");
                    Ok(())
                },
            )
            .map_err(error)
    })?;
    model.device().synchronize().map_err(|e| Error::Inference(e.to_string()))?;
    let forward_ms = started.elapsed().as_secs_f64() * 1000.;
    check(ctl)?;
    if result.image.dims() != [1, 3, geometry.rows, geometry.side] || result.alpha.dims() != [1, 1, geometry.rows, geometry.side] {
        return Err(Error::Inference("OSOR returned unexpected RGB or edit-alpha dimensions".into()));
    }
    let rgb = result.image.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| Error::Inference(e.to_string()))?;
    let alpha = result.alpha.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| Error::Inference(e.to_string()))?;
    Ok(RemovalOutput {
        rgb: geometry.restore(&rgb)?,
        edit_alpha: Some(geometry.restore_alpha(&alpha)?),
        inference_scale: geometry.scale,
        load_ms: 0.,
        forward_ms,
    })
}

fn noise(count: usize, mut state: u64) -> Vec<f32> {
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // Open interval (0,1), including neither log(0) nor an exact endpoint.
        ((z >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    };
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let radius = (-2.0 * uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform();
        values.push((radius * angle.cos()) as f32);
        if values.len() < count {
            values.push((radius * angle.sin()) as f32);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_reaches_a_backend_block_without_an_observer_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Control(AtomicBool);
        impl RemovalControl for Control {
            fn cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
            fn progress(&self, _: f32, _: &str) {}
        }
        let control = Control(AtomicBool::new(false));
        let result = with_cancellation(&control, |token| {
            // Simulate a long block that checks the library token but emits no stage event.
            control.0.store(true, Ordering::Release);
            let deadline = Instant::now() + std::time::Duration::from_secs(1);
            while !token.is_cancelled() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            token.check().map_err(error)
        });
        assert!(matches!(result, Err(Error::Cancelled)), "a cancellation during a block must reach the library without a stage observer: {result:?}");
    }

    #[test]
    fn cancellation_monitor_finishes_on_success_and_backend_unwind() {
        assert_eq!(with_cancellation(&crate::NoopControl, |_| Ok(42)).unwrap(), 42);
        let result = std::panic::catch_unwind(|| {
            let _: Result<()> = with_cancellation(&crate::NoopControl, |_| panic!("synthetic backend panic"));
        });
        assert!(result.is_err(), "the scoped monitor must stop while a backend unwinds");
    }

    #[test]
    fn explicit_noise_repeats_and_uses_separate_finite_normal_streams() {
        let posterior = noise(16_384, 0x5048_4f54_4f43_5241);
        assert_eq!(posterior, noise(16_384, 0x5048_4f54_4f43_5241));
        assert_ne!(posterior, noise(16_384, 0x4654_4f53_4f52_0001));
        assert!(posterior.iter().all(|n| n.is_finite()));
        let mean = posterior.iter().sum::<f32>() / posterior.len() as f32;
        let variance = posterior.iter().map(|n| (*n - mean).powi(2)).sum::<f32>() / posterior.len() as f32;
        assert!(mean.abs() < 0.05 && (0.9..1.1).contains(&variance));
        assert!(noise(0, 1).is_empty());
        assert_eq!(noise(3, 1).len(), 3);
    }
}
