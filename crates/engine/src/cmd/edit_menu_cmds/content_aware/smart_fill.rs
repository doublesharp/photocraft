//! LaMa fills the selected area from bounded rectangular, opaque context.
use super::{CafParams, Document, EngineError, FillEdit, FilledWindow, LayerId, Rect, Result, Session, Value, bad, caf_params, content_aware_plan_with, json};
use std::sync::Arc;
const CMD: &str = "edit.contentAwareFill";
fn error(e: photocraft_ml::Error) -> EngineError {
    if matches!(e, photocraft_ml::Error::Cancelled) { EngineError::Cancelled } else { bad(CMD, e.to_string()) }
}

/// Runs before floating-selection handling or any edit/allocation side effects.
pub(crate) fn validate(s: &Session, p: &Value) -> Result<()> {
    let method = match p.get("method") {
        None => "classical",
        Some(Value::String(v)) if matches!(v.as_str(), "classical" | "ai") => v.as_str(),
        _ => return Err(bad(CMD, "method must be classical or ai")),
    };
    if method == "classical" {
        if ["modelId", "modelSha256", "preview"].iter().any(|k| p.get(*k).is_some()) {
            return Err(bad(CMD, "model and preview options require method: ai"));
        }
        return Ok(());
    }
    let object = p.as_object().ok_or_else(|| bad(CMD, "params must be an object"))?;
    const KEYS: &[&str] = &[
        "method",
        "modelId",
        "modelSha256",
        "preview",
        "sampling",
        "margin",
        "area",
        "channel",
        "samplingStrokes",
        "fillEdits",
        "sampleAllLayers",
        "colorAdaptation",
        "rotationAdaptation",
        "scale",
        "mirror",
        "output",
        "layer",
        "target",
        "seed",
    ];
    if let Some(k) = object.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(bad(CMD, format!("unknown Smart Fill option {}", k.chars().take(64).collect::<String>())));
    }
    if p.get("modelId").is_some_and(|v| v.as_str() != Some("lama")) {
        return Err(bad(CMD, "Smart Fill supports modelId: lama only"));
    }
    for key in ["preview", "sampleAllLayers", "scale", "mirror"] {
        if p.get(key).is_some_and(|v| !v.is_boolean()) {
            return Err(bad(CMD, format!("{key} must be a boolean")));
        }
    }
    if p.get("sampling").is_some_and(|v| v.as_str() != Some("rectangular"))
        || ["area", "channel", "seed"].iter().any(|k| p.get(*k).is_some())
        || p.get("samplingStrokes").is_some_and(|v| !v.as_array().is_some_and(|a| a.is_empty()))
        || p.get("colorAdaptation").is_some_and(|v| v.as_str() != Some("none"))
        || p.get("rotationAdaptation").is_some_and(|v| v.as_str() != Some("none"))
        || p.get("scale").and_then(Value::as_bool) == Some(true)
        || p.get("mirror").and_then(Value::as_bool) == Some(true)
    {
        return Err(bad(
            CMD,
            "LaMa uses rectangular image context; choose Classical for sampling-brush, custom source areas, adaptation, rotation, scale, mirror or seed controls",
        ));
    }
    if p.get("output").is_some_and(|v| !v.as_str().is_some_and(|s| matches!(s, "current" | "new" | "duplicate"))) {
        return Err(bad(CMD, "output must be current, new or duplicate"));
    }
    if p.get("layer").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad(CMD, "layer must be a nonnegative layer id"));
    }
    if p.get("target").is_some_and(|v| v.as_str() != Some("pixels")) {
        return Err(bad(CMD, "AI Smart Fill requires the full-colour pixel target"));
    }
    if p.get("margin").is_some_and(|v| !v.as_i64().is_some_and(|n| (0..=100_000).contains(&n))) {
        return Err(bad(CMD, "margin must be 0..100000 pixels"));
    }
    if p.get("preview").and_then(Value::as_bool) == Some(true) && (s.actions.playing > 0 || s.actions.script_depth > 0) {
        return Err(bad(CMD, "Smart Fill previews require explicit review outside action or script playback"));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if crate::float_cmds::floating(st).is_some() {
        return Err(bad(CMD, "anchor the floating selection before AI Smart Fill"));
    }
    if !s.prefs().generative_remove.enabled {
        return Err(bad(CMD, "local AI processing is disabled; enable LaMa in Models"));
    }
    if !s.prefs().generative_remove.enabled_models.iter().any(|id| id == "lama") {
        return Err(bad(CMD, "LaMa is disabled; enable it in Models"));
    }
    if !s.removal_service().model_runtime_available("lama") {
        return Err(bad(CMD, "LaMa inference is unavailable in this build"));
    }
    if let Some(reason) = crate::retouch_cmds::removal_context_reason(s, p) {
        return Err(bad(CMD, reason.replace("AI Remove", "AI Smart Fill")));
    }
    if let Some(pin) = p.get("modelSha256")
        && pin.as_str() != Some(s.removal_service().model_sha256("lama").map_err(error)?.as_str())
    {
        return Err(bad(CMD, "the installed LaMa model differs from the captured modelSha256"));
    }
    Ok(())
}

/// Conservative geometry bound before reading a mask, composite or pixel window.
fn bound(doc: &Document, id: LayerId, p: &Value) -> Result<u64> {
    let CafParams { sb, edits, .. } = caf_params(doc, id, p, CMD)?;
    let grow = edits.iter().try_fold(0i32, |n, e| {
        if let FillEdit::Grow(by) = e { n.checked_add((*by).max(0)).ok_or_else(|| bad(CMD, "fill expansion overflows")) } else { Ok(n) }
    })?;
    let reach = edits
        .iter()
        .fold(sb, |r, e| match e {
            FillEdit::Lasso { points, .. } => {
                points.iter().fold(r, |r, q| r.union(&Rect::new(q[0].floor() as i32, q[1].floor() as i32, q[0].floor() as i32 + 1, q[1].floor() as i32 + 1)))
            }
            FillEdit::Grow(_) => r,
        })
        .inflate(grow.saturating_add(1))
        .intersect(&doc.bounds());
    let sampling = p
        .get("margin")
        .and_then(Value::as_i64)
        .map_or_else(|| crate::fill_cmds::sampling_window(reach, doc.bounds()), |m| reach.inflate(m as i32).intersect(&doc.bounds()));
    let rect = reach.union(&sampling);
    let pixels = u64::from(rect.width())
        .checked_mul(u64::from(rect.height()))
        .filter(|n| *n > 0 && *n <= photocraft_ml::MAX_INPUT_PIXELS)
        .ok_or_else(|| bad(CMD, "AI Smart Fill context is too large; use a smaller selection or Classical"))?;
    // Covers edit-mask scratch, source/composite, float RGB and masks, filled/original
    // windows, candidate output tiles and output=new's temporary RGBA buffer.
    pixels.checked_mul(160).ok_or_else(|| bad(CMD, "Smart Fill memory estimate overflow"))
}

pub(super) fn preflight(s: &Session, p: &Value) -> Result<()> {
    if p.get("method").and_then(Value::as_str) != Some("ai") {
        return Err(bad(CMD, "LaMa preflight requires method: ai"));
    }
    validate(s, p)?;
    let state = s.active().ok_or(EngineError::NoDocument)?;
    let id = p.get("layer").and_then(Value::as_u64).map(LayerId).or(state.active_layer).ok_or(EngineError::NoDocument)?;
    let transient = bound(&state.doc, id, p)?;
    let working = s
        .removal_service()
        .catalog()
        .iter()
        .find(|m| m.id == "lama")
        .map(|m| u64::from(m.memory_mib) * 1024 * 1024)
        .ok_or_else(|| bad(CMD, "LaMa is unavailable"))?;
    let budget = u64::from(s.prefs().performance.memory_usage_mb) * 1024 * 1024;
    if working.checked_add(s.removal_service().admitted_transient_bytes(transient).map_err(error)?).is_none_or(|n| n > budget) {
        return Err(bad(CMD, "LaMa plus Smart Fill image buffers exceed the configured memory allowance"));
    }
    Ok(())
}

pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    validate(s, p)?;
    let id = p.get("layer").and_then(Value::as_u64).map(LayerId).or_else(|| s.active().and_then(|st| st.active_layer)).ok_or(EngineError::NoDocument)?;
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let active_layer = s.active().and_then(|st| st.active_layer);
    let mut p = p.clone();
    p["sampling"] = json!("rectangular");
    p["colorAdaptation"] = json!("none");
    if p.get("sampleAllLayers").is_none() {
        p["sampleAllLayers"] = json!(false);
    }
    let transient = bound(&doc, id, &p)?;
    let budget = u64::from(s.prefs().performance.memory_usage_mb) * 1024 * 1024;
    let service = s.removal_service();
    let working =
        service.catalog().iter().find(|m| m.id == "lama").map(|m| u64::from(m.memory_mib) * 1024 * 1024).ok_or_else(|| bad(CMD, "LaMa is unavailable"))?;
    if working.checked_add(service.admitted_transient_bytes(transient).map_err(error)?).is_none_or(|required| required > budget) {
        return Err(bad(CMD, "LaMa plus Smart Fill image buffers exceed the configured memory allowance"));
    }
    // Replacing this workspace's previous candidate releases its reservation, while
    // other sessions/Remove candidates remain protected by process-wide admission.
    s.fill_preview = None;
    let status = service.status_for_fill().into_iter().find(|st| st.id == "lama").ok_or_else(|| bad(CMD, "LaMa is unavailable"))?;
    if status.busy {
        return Err(bad(CMD, "local AI is busy; wait for its operation or discard its preview"));
    }
    if !status.available {
        return Err(bad(CMD, format!("LaMa: {}; open Models to install it", status.reason.as_deref().unwrap_or("not installed"))));
    }
    let sha = service.model_sha256("lama").map_err(error)?;
    let preview = p.get("preview").and_then(Value::as_bool) == Some(true);
    let capture = crate::smart_fill_preview::Capture::new(s)?;
    let mut accepted = p.clone();
    accepted["preview"] = json!(false);
    accepted["modelId"] = json!("lama");
    accepted["modelSha256"] = json!(sha);
    crate::jobs::run(
        s,
        "Smart Fill (LaMa)",
        true,
        move |ctx| {
            ctx.check()?;
            let mut permit = service.admit_fill(transient, budget).map_err(error)?;
            if service.model_sha256("lama").map_err(error)? != sha {
                return Err(bad(CMD, "the LaMa model changed before inference"));
            }
            let cancel = || ctx.cancelled();
            let plan = content_aware_plan_with(&doc, id, &p, CMD, &photocraft_raster::Interrupt::cancel_only(&cancel))?;
            let n = plan.fmt.channels();
            if !matches!(n, 3 | 4) || plan.cover.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err(bad(CMD, "invalid Smart Fill pixels or selection coverage"));
            }
            let count = plan.hole.len();
            let mut rgb = Vec::with_capacity(count.saturating_mul(3));
            for (pixel, hole) in plan.img.chunks_exact(n).zip(&plan.hole) {
                if !hole && plan.fmt.alpha && pixel.get(3).copied() != Some(1.) {
                    return Err(bad(CMD, "LaMa needs opaque known context; use Sample All Layers or Classical"));
                }
                let color = photocraft_raster::to_rgba(&plan.fmt, pixel);
                rgb.extend_from_slice(&color[..3]);
            }
            let mask = plan.hole.iter().map(|v| if *v { 1. } else { 0. }).collect();
            let input = photocraft_ml::RemovalInput { width: plan.rect.width(), height: plan.rect.height(), rgb, mask };
            let output = permit.infer(&input, ctx).map_err(error)?;
            ctx.check()?;
            let rect = plan.rect;
            let fmt = plan.fmt;
            let own = plan.layer_pixels();
            let mut filled = own.to_vec();
            let keep_alpha = plan.output != "new" && doc.effective_locks(id).transparency;
            for ((dest, color), hole) in filled.chunks_exact_mut(n).zip(output.rgb.as_chunks::<3>().0.iter()).zip(&plan.hole) {
                if *hole {
                    if let Some(target) = dest.get_mut(..3) {
                        target.copy_from_slice(color);
                    }
                    if fmt.alpha
                        && !keep_alpha
                        && let Some(a) = dest.get_mut(3)
                    {
                        *a = 1.;
                    }
                }
            }
            let (w, h) = (rect.width() as usize, rect.height() as usize);
            let bounds = super::mask_bounds(w, h, &plan.hole);
            let (x0, y0, x1, y1) = bounds;
            let mut original = Vec::new();
            let mut cover = Vec::new();
            for y in y0..y1 {
                let row = y.checked_mul(w).ok_or_else(|| bad(CMD, "Smart Fill row overflow"))?;
                let start = row.checked_add(x0).ok_or_else(|| bad(CMD, "Smart Fill column overflow"))?;
                let end = row.checked_add(x1).ok_or_else(|| bad(CMD, "Smart Fill column overflow"))?;
                let a = start.checked_mul(n).ok_or_else(|| bad(CMD, "Smart Fill pixel offset overflow"))?;
                let b = end.checked_mul(n).ok_or_else(|| bad(CMD, "Smart Fill pixel offset overflow"))?;
                original.extend_from_slice(own.get(a..b).ok_or_else(|| bad(CMD, "invalid Smart Fill original window"))?);
                cover.extend_from_slice(plan.cover.get(start..end).ok_or_else(|| bad(CMD, "invalid Smart Fill coverage window"))?);
            }
            let n_hole = cover.iter().filter(|v| **v > 0.).count();
            let mut candidate = (*doc).clone();
            let mut active = active_layer;
            let layer = super::apply_filled_document(
                &mut candidate,
                &mut active,
                id,
                &plan.output,
                fmt,
                rect,
                FilledWindow { filled, bounds, original, cover, n_hole, alpha_aware: true },
            )?;
            let result = json!({"layer":layer.0,"filled":n_hole,"window":[rect.x0,rect.y0,rect.width(),rect.height()],"crop":[rect.x0,rect.y0,rect.x1,rect.y1],"effectiveMethod":"ai","modelId":"lama","modelSha256":sha,"inferenceScale":output.inference_scale,"loadMs":output.load_ms,"forwardMs":output.forward_ms,"estimatedTransientBytes":transient,"memoryBudgetBytes":budget});
            ctx.check()?;
            // Source/input/adapter buffers drop before candidate's reservation is returned.
            drop((input, output, plan));
            let candidate = crate::smart_fill_preview::Candidate {
                doc: Arc::new(candidate),
                active_layer: active,
                crop: rect,
                params: accepted,
                result,
                _reservation: permit.into_reservation(),
            };
            Ok(candidate)
        },
        move |s, candidate| {
            if preview { crate::smart_fill_preview::publish(s, capture, candidate) } else { crate::smart_fill_preview::apply_candidate(s, capture, candidate) }
        },
    )
}
