//! Exact pinned ONNX adapters. No model code or Python is executed by the application.
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use tract_onnx::prelude::*;

use crate::resample::Geometry;
use crate::store::Fingerprint;
use crate::{Error, ModelSpec, RemovalControl, RemovalInput, RemovalOutput, Result, check};

#[cfg(target_os = "macos")]
mod metal;

pub(crate) struct Cached {
    pub model_id: &'static str,
    fingerprint: Fingerprint,
    requested: crate::OnnxBackend,
    retry_at_bytes: Option<u64>,
    plan: Arc<dyn Runnable>,
    pub info: crate::ExecutionInfo,
    pub working_bytes: u64,
}

pub(crate) struct Request {
    pub backend: crate::OnnxBackend,
    pub available_bytes: u64,
    pub fallback: Option<String>,
}

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Inference(format!("{error:#}"))
}

fn contained<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).map_err(|_| Error::Inference("the local inference runtime failed internally".into()))?
}

impl Cached {
    fn reusable(&self, model: &ModelSpec, fingerprint: &Fingerprint, request: &Request) -> bool {
        self.model_id == model.id
            && self.fingerprint == *fingerprint
            && self.requested == request.backend
            && self.working_bytes <= request.available_bytes
            && self.retry_at_bytes.is_none_or(|required| required > request.available_bytes)
            && request.fallback.is_none()
    }
}

pub(crate) fn load(
    model: &'static ModelSpec,
    mut file: File,
    fingerprint: Fingerprint,
    old: Option<Cached>,
    request: Request,
    ctl: &dyn RemovalControl,
) -> Result<(Cached, f64)> {
    // Production Auto enables only qualified model/backend pairs. Explicit Metal
    // remains available for a separate MI-GAN qualification.
    let cpu_only = model.id == "migan-v1" && request.backend == crate::OnnxBackend::Auto;
    if let Some(old) = old {
        if !(cpu_only && old.info.uses_gpu) && old.reusable(model, &fingerprint, &request) {
            return Ok((old, 0.));
        }
        drop(old);
    }
    let started = Instant::now();
    check(ctl)?;
    ctl.progress(0.08, "Loading local model");
    // Cold parsing consumes the exact bounded snapshot that was hashed. The store
    // already rehashed the opened file before even considering a warm plan.
    let snapshot = crate::store::verified_snapshot(&mut file, model, ctl)?;
    check(ctl)?;
    let mut graph = contained(|| tract_onnx::onnx().model_for_read(&mut snapshot.as_slice()).map_err(model_error))?;
    drop(snapshot);
    check(ctl)?;
    let side = model.input_size as usize;
    if model.id == "migan-v1" {
        graph.set_input_fact(0, u8::fact([1, 3, side, side]).into()).map_err(model_error)?;
        graph.set_input_fact(1, u8::fact([1, 1, side, side]).into()).map_err(model_error)?;
    } else {
        graph.set_input_fact(0, f32::fact([1, 3, side, side]).into()).map_err(model_error)?;
        graph.set_input_fact(1, f32::fact([1, 1, side, side]).into()).map_err(model_error)?;
    }
    ctl.progress(0.18, "Preparing local model");
    let graph = contained(|| graph.into_typed().map_err(model_error))?;
    check(ctl)?;
    let base_bytes = u64::from(model.memory_mib) * 1024 * 1024;
    let mut fallback = request.fallback;
    if cpu_only && fallback.is_none() {
        fallback = Some("MI-GAN Metal execution is not qualified; using the same model on CPU".into());
    }
    let mut retry_at_bytes = None;
    if request.backend != crate::OnnxBackend::Cpu && fallback.is_none() {
        #[cfg(target_os = "macos")]
        let prepared = contained(|| metal::prepare(graph.clone(), model));
        #[cfg(not(target_os = "macos"))]
        let prepared: Result<(Arc<dyn Runnable>, crate::ExecutionInfo)> = Err(Error::Inference("Metal inference is only available on macOS".into()));
        #[cfg(target_os = "macos")]
        if prepared.is_ok() {
            contained(metal::drain)?;
        }
        match prepared {
            Ok((plan, info)) => {
                let working_bytes = base_bytes
                    .checked_add(info.graph.as_ref().map_or(0, |graph| graph.gpu_arena_bytes))
                    .ok_or_else(|| Error::Inference("Metal working set overflow".into()))?;
                if working_bytes <= request.available_bytes {
                    check(ctl)?;
                    return Ok((
                        Cached { model_id: model.id, fingerprint, requested: request.backend, retry_at_bytes: None, plan, info, working_bytes },
                        started.elapsed().as_secs_f64() * 1000.,
                    ));
                }
                retry_at_bytes = Some(working_bytes);
                fallback = Some(format!("Metal needs an estimated {working_bytes} model bytes; {} are available", request.available_bytes));
            }
            Err(error) => {
                // Initialization/lowering may have submitted constant evaluation work.
                #[cfg(target_os = "macos")]
                if tract_gpu::device::get_context().is_ok_and(|context| context.is::<tract_metal::MetalContext>()) {
                    contained(metal::drain)?;
                }
                fallback = Some(error.to_string());
            }
        }
        if request.backend == crate::OnnxBackend::Metal {
            return Err(Error::Inference(fallback.unwrap_or_else(|| "Metal unavailable".into())));
        }
    }
    check(ctl)?;
    let plan: Arc<dyn Runnable> = Arc::from(contained(|| {
        let runtime = runtime_for_name("cpu").map_err(model_error)?.ok_or(Error::Unavailable)?;
        runtime.prepare(graph).map_err(model_error)
    })?);
    let cpu_nodes = if let Some(graph) = plan.typed_model() {
        graph.eval_order().map_err(model_error)?.iter().filter(|id| graph.nodes().get(**id).is_some_and(|node| !node.inputs.is_empty())).count()
    } else {
        0
    };
    let info = crate::ExecutionInfo {
        model_id: model.id.into(),
        runtime: "tract CPU".into(),
        device: "CPU".into(),
        uses_gpu: false,
        graph: Some(crate::GraphPlacement { gpu_nodes: 0, cpu_nodes, transfer_nodes: 0, gpu_arena_bytes: 0 }),
        fallback_reason: fallback,
    };
    check(ctl)?;
    Ok((
        Cached { model_id: model.id, fingerprint, requested: request.backend, retry_at_bytes, plan, info, working_bytes: base_bytes },
        started.elapsed().as_secs_f64() * 1000.,
    ))
}

/// A failed synchronization never triggers a CPU retry while device work may remain.
pub(crate) struct Forward {
    pub result: Result<RemovalOutput>,
    pub drained: bool,
}

pub(crate) fn forward(model: &ModelSpec, cached: &Cached, input: &RemovalInput, ctl: &dyn RemovalControl) -> Forward {
    let result = contained(|| infer(model, cached, input, ctl));
    #[cfg(target_os = "macos")]
    if cached.info.uses_gpu
        && let Err(error) = contained(metal::drain)
    {
        return Forward { result: Err(error), drained: false };
    }
    Forward { result, drained: true }
}

fn infer(model: &ModelSpec, cached: &Cached, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
    check(ctl)?;
    ctl.progress(0.28, "Preparing removal crop");
    let geometry = Geometry::new(input.width, input.height, model.input_size)?;
    let (rgb, mask) = geometry.prepare(input)?;
    check(ctl)?;
    let side = model.input_size as usize;
    let inputs = if model.id == "migan-v1" {
        let image: Vec<u8> = rgb.iter().map(|v| (*v * 255.).clamp(0., 255.).round() as u8).collect();
        let mask: Vec<u8> = mask.iter().map(|v| if *v > 0. { 0 } else { 255 }).collect();
        tvec![
            Tensor::from_shape(&[1, 3, side, side], &image).map_err(model_error)?.into(),
            Tensor::from_shape(&[1, 1, side, side], &mask).map_err(model_error)?.into()
        ]
    } else {
        tvec![
            Tensor::from_shape(&[1, 3, side, side], &rgb).map_err(model_error)?.into(),
            Tensor::from_shape(&[1, 1, side, side], &mask).map_err(model_error)?.into()
        ]
    };
    ctl.progress(0.4, "Reconstructing background locally");
    let started = Instant::now();
    // tract's forward is not cooperatively interruptible. The owner keeps its permit while this
    // returns; cancellation is acknowledged by the engine without releasing inference memory.
    let outputs = contained(|| cached.plan.run(inputs).map_err(model_error));
    #[cfg(target_os = "macos")]
    if cached.info.uses_gpu {
        contained(metal::drain)?;
    }
    let outputs = outputs?;
    let forward_ms = started.elapsed().as_secs_f64() * 1000.;
    check(ctl)?;
    let output = outputs.first().ok_or_else(|| Error::Inference("model returned no image".into()))?;
    if output.shape() != [1, 3, side, side] {
        return Err(Error::Inference("model returned an unexpected image shape".into()));
    }
    let planar: Vec<f32> = if model.id == "migan-v1" {
        output.to_plain_array_view::<u8>().map_err(model_error)?.iter().map(|v| f32::from(*v) / 255.).collect()
    } else {
        // This exact LaMa export returns float samples in 0..255, independently of content.
        // Never infer output normalization from a particular image's min/max.
        output.to_plain_array_view::<f32>().map_err(model_error)?.iter().map(|v| *v / 255.).collect()
    };
    if planar.iter().any(|v| !v.is_finite()) {
        return Err(Error::Inference("model returned non-finite image samples".into()));
    }
    ctl.progress(0.92, "Restoring crop detail");
    let rgb = geometry.restore(&planar)?;
    check(ctl)?;
    Ok(RemovalOutput { edit_alpha: None, rgb, inference_scale: geometry.scale, load_ms: 0., forward_ms })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upstream_runtime_panics_become_actionable_errors() {
        let result: Result<()> = contained(|| panic!("simulated third-party runtime failure"));
        assert!(matches!(result, Err(Error::Inference(message)) if message.contains("runtime failed internally")));
    }
}
