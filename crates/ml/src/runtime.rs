//! Exact pinned ONNX adapters. No model code or Python is executed by the application.
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use tract_onnx::prelude::*;

use crate::resample::Geometry;
use crate::store::Fingerprint;
use crate::{Error, ModelSpec, RemovalControl, RemovalInput, RemovalOutput, Result, check};

pub(crate) struct Cached {
    pub model_id: &'static str,
    fingerprint: Fingerprint,
    plan: Arc<TypedRunnableModel>,
}

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Inference(error.to_string())
}

pub(crate) fn load(
    model: &'static ModelSpec,
    mut file: File,
    fingerprint: Fingerprint,
    old: Option<Cached>,
    ctl: &dyn RemovalControl,
) -> Result<(Cached, f64)> {
    if let Some(old) = old {
        if old.model_id == model.id && old.fingerprint == fingerprint {
            return Ok((old, 0.));
        }
        drop(old);
    }
    let started = Instant::now();
    check(ctl)?;
    ctl.progress(0.08, "Loading local model");
    // Parse exactly the bounded byte snapshot whose digest was checked. The on-disk file can
    // be rewritten after verification, but that cannot change this graph's input bytes.
    let snapshot = crate::store::verified_snapshot(&mut file, model, ctl)?;
    check(ctl)?;
    let mut graph = tract_onnx::onnx().model_for_read(&mut snapshot.as_slice()).map_err(model_error)?;
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
    let graph = graph.into_optimized().map_err(model_error)?;
    check(ctl)?;
    let plan = graph.into_runnable().map_err(model_error)?;
    check(ctl)?;
    Ok((Cached { model_id: model.id, fingerprint, plan }, started.elapsed().as_secs_f64() * 1000.))
}

pub(crate) fn infer(model: &ModelSpec, cached: &Cached, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
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
    let outputs = cached.plan.run(inputs).map_err(model_error)?;
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
