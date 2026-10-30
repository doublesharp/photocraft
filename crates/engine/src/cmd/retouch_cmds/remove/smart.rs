//! The opt-in learned path, bounded to the captured Remove window.
use super::{
    BlendMode, CMD, Document, EngineError, Layer, LayerContent, LayerId, PixelFormat, Plan, Rect, Region, Result, SEED, SampleType, Session, Value, bad, brush,
    damage_json, hole, json, run_remove, stroke_bounds, targets_pixels,
};
use photocraft_color::ColorMode;
use photocraft_ml::{RemovalInput, RemovalOutput};
use photocraft_raster::from_rgba_into;
use std::sync::Arc;

#[derive(Clone)]
pub(super) enum Method {
    Legacy,
    Hybrid,
    Classical { fallback: Option<String> },
    Ai { id: String, sha256: String, automatic: bool },
}

/// Validate all new options before stroke parsing, floating-selection handling or pixel allocation.
pub(super) fn validate(s: &Session, p: &Value) -> Result<()> {
    let method = match p.get("method") {
        None => {
            if ["modelId", "modelSha256", "removeEffects", "preview", "strokeSpecs", "seed", "repairType", "regionalPlan", "modelRevisions"]
                .iter()
                .any(|k| p.get(*k).is_some())
            {
                return Err(bad(CMD, "new Remove options require an explicit method"));
            }
            return Ok(());
        }
        Some(Value::String(method)) if matches!(method.as_str(), "classical" | "auto" | "ai") => method.as_str(),
        _ => return Err(bad(CMD, "`method` must be classical, auto or ai")),
    };
    let Some(obj) = p.as_object() else { return Err(bad(CMD, "params must be an object")) };
    const KEYS: &[&str] = &[
        "method",
        "brush",
        "modelId",
        "modelSha256",
        "seed",
        "removeEffects",
        "preview",
        "points",
        "strokes",
        "size",
        "hardness",
        "opacity",
        "flow",
        "spacing",
        "layer",
        "target",
        "closeLoops",
        "sampleAllLayers",
        "newLayer",
        "coalesce",
        "repairType",
        "regionalPlan",
        "modelRevisions",
    ];
    if let Some(key) = obj.keys().find(|key| !KEYS.contains(&key.as_str())) {
        let shown: String = key.chars().take(64).collect();
        return Err(bad(CMD, format!("unknown Remove option `{shown}`")));
    }
    if p.get("points").is_some() && p.get("strokes").is_some() {
        return Err(bad(CMD, "use `points` or `strokes`, not both"));
    }
    // Reject hostile point collections and malformed mixed-stroke entries before cloning them.
    let check_points = |value: &Value| -> Result<usize> {
        let points = value.as_array().filter(|v| !v.is_empty() && v.len() <= 100_000).ok_or_else(|| bad(CMD, "a stroke needs 1..100000 points"))?;
        for point in points {
            let valid = point.as_array().is_some_and(|v| {
                (2..=3).contains(&v.len())
                    && v.iter().all(|n| n.as_f64().is_some_and(f64::is_finite))
                    && v.first().and_then(Value::as_f64).is_some_and(|x| x.abs() <= 1_000_000.0)
                    && v.get(1).and_then(Value::as_f64).is_some_and(|y| y.abs() <= 1_000_000.0)
                    && v.get(2).is_none_or(|pressure| pressure.as_f64().is_some_and(|x| (0.0..=1.0).contains(&x)))
            });
            if !valid {
                return Err(bad(CMD, "stroke points must be finite [x,y,pressure?] values"));
            }
        }
        Ok(points.len())
    };
    if let Some(strokes) = p.get("strokes") {
        let strokes = strokes.as_array().filter(|v| !v.is_empty() && v.len() <= 1024).ok_or_else(|| bad(CMD, "`strokes` must contain 1..1024 strokes"))?;
        let mut total = 0usize;
        for stroke in strokes {
            let points = if let Some(obj) = stroke.as_object() {
                if obj.keys().any(|key| !["points", "brush", "size", "hardness", "spacing", "repairType"].contains(&key.as_str())) {
                    return Err(bad(CMD, "unknown stroke option"));
                }
                for (key, lo, hi) in [("size", 1.0, 5000.0), ("hardness", 0.0, 100.0), ("spacing", 1.0, 1000.0)] {
                    if obj.get(key).is_some_and(|v| v.as_f64().is_none_or(|n| !n.is_finite() || !(lo..=hi).contains(&n))) {
                        return Err(bad(CMD, format!("invalid stroke `{key}`")));
                    }
                }
                obj.get("points").ok_or_else(|| bad(CMD, "stroke is missing `points`"))?
            } else {
                stroke
            };
            total = total.saturating_add(check_points(points)?);
            if total > 100_000 {
                return Err(bad(CMD, "too many stroke points"));
            }
        }
    } else if let Some(points) = p.get("points") {
        check_points(points)?;
    }
    for key in ["closeLoops", "sampleAllLayers", "newLayer", "removeEffects", "preview"] {
        if p.get(key).is_some_and(|v| !v.is_boolean()) {
            return Err(bad(CMD, format!("`{key}` must be a boolean")));
        }
    }
    super::hybrid::validate_hints(p)?;
    let configured = s.prefs().generative_remove.workflow.as_str();
    let hybrid = method == "auto" && configured == "recommended";
    if !hybrid && (p.get("regionalPlan").is_some() || p.get("modelRevisions").is_some()) {
        return Err(bad(CMD, "frozen regional metadata requires the recommended Auto workflow"));
    }
    let profile_ai = method == "auto" && matches!(configured, "recommended" | "lama" | "compact");
    let preview = p.get("preview").and_then(Value::as_bool).unwrap_or(hybrid);
    let effects = p.get("removeEffects").and_then(Value::as_bool) == Some(true);
    if preview && method != "ai" && !profile_ai {
        return Err(bad(CMD, "Remove previews require the explicit ai method"));
    }
    if effects && hybrid && !preview {
        return Err(bad(CMD, "effect removal requires an OSOR-SDXL preview"));
    }
    if effects
        && !hybrid
        && (method != "ai" || !preview || p.get("modelId").and_then(Value::as_str).unwrap_or(&s.prefs().generative_remove.selected_model) != "osor-sdxl")
    {
        return Err(bad(CMD, "effect removal requires an OSOR-SDXL preview"));
    }
    if (preview || hybrid) && (s.actions.playing > 0 || s.actions.script_depth > 0) {
        return Err(bad(CMD, "Remove previews require explicit review and are unavailable during action or script playback"));
    }
    if preview && s.active().is_some_and(|st| crate::float_cmds::floating(st).is_some()) {
        return Err(bad(CMD, "finish the floating selection before generating a Remove preview"));
    }
    if p.get("seed").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad(CMD, "`seed` must be a nonnegative integer"));
    }
    if (method == "ai" || profile_ai) && p.get("seed").and_then(Value::as_u64).is_some_and(|seed| seed != SEED) {
        return Err(bad(CMD, "this deterministic AI adapter does not support a custom seed"));
    }
    if p.get("coalesce").is_some_and(|value| value.as_str().is_none_or(|key| key.len() > 256)) {
        return Err(bad(CMD, "`coalesce` must be a string of at most 256 bytes"));
    }
    if p.get("layer").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad(CMD, "`layer` must be a layer id"));
    }
    if let Some(target) = p.get("target") {
        let valid = target.as_str().is_some_and(|v| matches!(v, "pixels" | "mask" | "filterMask" | "quickMask"))
            || target.as_object().is_some_and(|o| o.len() == 1 && o.get("channel").and_then(Value::as_u64).is_some());
        if !valid {
            return Err(bad(CMD, "unknown `target`"));
        }
    }
    for (key, min, max) in [("size", 1.0, 5000.0), ("hardness", 0.0, 100.0), ("opacity", 1.0, 100.0), ("flow", 1.0, 100.0), ("spacing", 1.0, 1000.0)] {
        if p.get(key).is_some_and(|v| v.as_f64().is_none_or(|x| !x.is_finite() || !(min..=max).contains(&x))) {
            return Err(bad(CMD, format!("`{key}` must be within {min}..{max}")));
        }
    }
    if let Some(model) = p.get("modelId") {
        let id = model.as_str().filter(|id| !id.is_empty() && id.len() <= 64).ok_or_else(|| bad(CMD, "`modelId` must be a known model id"))?;
        let service = s.removal_service();
        service.catalog().iter().find(|m| m.id == id).ok_or_else(|| bad(CMD, format!("unknown removal model `{id}`")))?;
        if method == "classical" {
            return Err(bad(CMD, "`modelId` cannot be used with the classical method"));
        }
        if let Some(value) = p.get("modelSha256") {
            let hash = service.model_sha256(id).map_err(|error| bad(CMD, error.to_string()))?;
            if value.as_str() != Some(hash.as_str()) {
                return Err(bad(CMD, "model asset revision does not match the installed model"));
            }
        }
    } else if p.get("modelSha256").is_some() {
        return Err(bad(CMD, "`modelSha256` requires `modelId`"));
    }
    brush::validate_request(p)?;
    Ok(())
}

/// Sampling a composite with unbounded effects, smart objects, masks or pattern preparation can
/// allocate work outside the admitted crop. Until those paths have their own bounds, AI sampling
/// accepts only simple raster layers and pass-through groups.
fn simple_composite(layers: &[Layer], depth: usize, sample: SampleType) -> bool {
    if depth > photocraft_doc::MAX_GROUP_DEPTH {
        return false;
    }
    layers.iter().filter(|layer| layer.visible).all(|layer| {
        if layer.opacity != 1.0
            || layer.fill_opacity != 1.0
            || layer.clipped
            || layer.mask.is_some()
            || layer.vector_mask.is_some()
            || !layer.effects.items.is_empty()
            || layer.effects.psd_raw.is_some()
            || layer.excluded_channels != 0
            || !layer.blend_if.is_default()
            || !layer.advanced.is_default()
        {
            return false;
        }
        match &layer.content {
            LayerContent::Raster(surface) => layer.blend == BlendMode::Normal && surface.format().mode == ColorMode::Rgb && surface.format().sample == sample,
            LayerContent::Group(group) => {
                layer.blend == BlendMode::PassThrough && group.artboard.is_none() && simple_composite(&group.children, depth + 1, sample)
            }
            _ => false,
        }
    })
}

/// The format and destination contract which can be decided without reading the crop.
pub(super) fn context_reason(s: &Session, p: &Value) -> Option<String> {
    let Some(st) = s.active() else { return Some("no document is open".into()) };
    let doc = &st.doc;
    if doc.mode != ColorMode::Rgb {
        return Some("AI Remove supports RGB documents only".into());
    }
    if !matches!(doc.depth, SampleType::U8 | SampleType::U16) {
        return Some("AI Remove supports 8-bit and 16-bit depth only".into());
    }
    let srgb = photocraft_cms::Builtin::Srgb.profile();
    let profile_ok = match &doc.icc_profile {
        Some(bytes) if bytes.as_slice() == srgb.to_bytes().as_slice() => true,
        Some(bytes) => photocraft_cms::Profile::parse(bytes).is_ok_and(|profile| profile.same_colors(srgb)),
        None => s.color.settings.working_rgb == photocraft_cms::Builtin::Srgb.id(),
    };
    if !profile_ok {
        return Some("AI Remove requires an sRGB colour profile".into());
    }
    if p.get("sampleAllLayers").and_then(Value::as_bool) == Some(true) && !simple_composite(&doc.layers, 0, doc.depth) {
        return Some("AI Sample All Layers requires a simple raster composite without effects, masks or clipping".into());
    }
    let routed = crate::channel_cmds::inject_target(s, CMD, p.clone());
    if !targets_pixels(&routed) || crate::channel_cmds::color_restriction(s, CMD, &routed).is_some() {
        return Some("AI Remove requires the full-colour pixel target".into());
    }
    let id = match routed.get("layer").and_then(Value::as_u64).map(LayerId).or(st.active_layer) {
        Some(id) => id,
        None => return Some("AI Remove requires an active pixel layer".into()),
    };
    let Some(path) = doc.path_of(id) else { return Some("AI Remove target layer does not exist".into()) };
    let locks = doc.effective_locks(id);
    if locks.all || locks.pixels {
        return Some("AI Remove cannot edit a locked pixel layer".into());
    }
    for depth in 1..=path.len() {
        let Some(prefix) = path.get(..depth) else { return Some("invalid layer path".into()) };
        let Some(layer) = doc.layer_at(prefix) else { return Some("invalid layer path".into()) };
        let is_target = depth == path.len();
        let blend_ok = if is_target { layer.blend == BlendMode::Normal } else { matches!(layer.blend, BlendMode::Normal | BlendMode::PassThrough) };
        if !layer.visible
            || !blend_ok
            || layer.opacity != 1.0
            || layer.fill_opacity != 1.0
            || layer.clipped
            || layer.mask.is_some()
            || layer.vector_mask.is_some()
            || !layer.effects.items.is_empty()
            || layer.effects.psd_raw.is_some()
            || layer.excluded_channels != 0
            || !layer.blend_if.is_default()
            || !layer.advanced.is_default()
        {
            return Some("AI Remove requires a visible Normal raster layer and simple unmasked, unclipped ancestors".into());
        }
        if is_target && !matches!(layer.content, LayerContent::Raster(_)) {
            return Some("AI Remove requires a raster pixel layer".into());
        }
        if is_target && layer.surface().is_none_or(|surface| surface.format().mode != ColorMode::Rgb || surface.format().sample != doc.depth) {
            return Some("AI Remove requires pixels at the document RGB depth".into());
        }
        if matches!(&layer.content, LayerContent::Group(group) if group.artboard.is_some()) {
            return Some("AI Remove does not yet support artboard ancestors".into());
        }
        let Some((&index, parent)) = prefix.split_last() else { return Some("invalid layer path".into()) };
        let siblings = if parent.is_empty() { Some(doc.layers.as_slice()) } else { doc.layer_at(parent).and_then(|l| l.children()) };
        if siblings.and_then(|layers| layers.get(index.saturating_add(1))).is_some_and(|above| above.clipped) {
            return Some("AI Remove cannot change a clipping base".into());
        }
    }
    None
}

fn require_model(s: &Session, p: &Value, id: &str) -> Result<()> {
    let prefs = &s.prefs().generative_remove;
    let service = s.removal_service();
    if !prefs.enabled {
        return Err(bad(CMD, "local AI processing is disabled"));
    }
    if !prefs.enabled_models.iter().any(|model| model == id) {
        return Err(bad(CMD, format!("removal model `{id}` is disabled")));
    }
    if !service.model_runtime_available(id) {
        return Err(bad(CMD, format!("local AI inference for `{id}` is unavailable in this build")));
    }
    if let Some(reason) = context_reason(s, p) {
        return Err(bad(CMD, reason));
    }
    match service.status().into_iter().find(|status| status.id == id) {
        Some(status) if status.busy => Err(bad(CMD, "local AI is busy")),
        Some(status) if status.available => Ok(()),
        Some(status) => Err(bad(CMD, format!("{id}: {}", status.reason.as_deref().unwrap_or("model is not installed; explicitly install it first")))),
        None => Err(bad(CMD, format!("removal model `{id}` is unavailable"))),
    }
}

pub(super) fn decide(s: &Session, p: &Value) -> Result<Method> {
    validate(s, p)?;
    let Some(method) = p.get("method").and_then(Value::as_str) else { return Ok(Method::Legacy) };
    if method == "classical" {
        return Ok(Method::Classical { fallback: None });
    }
    let automatic = method == "auto";
    let prefs = &s.prefs().generative_remove;
    if automatic && prefs.workflow == "classical" {
        return Ok(Method::Classical { fallback: None });
    }
    if automatic && prefs.workflow == "recommended" {
        for id in ["lama", "osor-sdxl"] {
            require_model(s, p, id)?;
        }
        return Ok(Method::Hybrid);
    }
    if automatic && matches!(prefs.workflow.as_str(), "lama" | "compact") {
        let id = if prefs.workflow == "compact" { "migan-v1" } else { "lama" };
        require_model(s, p, id)?;
        let sha256 = s.removal_service().model_sha256(id).map_err(|error| bad(CMD, error.to_string()))?;
        return Ok(Method::Ai { id: id.into(), sha256, automatic: false });
    }
    let id = p.get("modelId").and_then(Value::as_str).unwrap_or(&prefs.selected_model);
    let service = s.removal_service();
    let spec = service.catalog().iter().find(|spec| spec.id == id);
    let reason = if !prefs.enabled {
        Some("local AI processing is disabled".to_string())
    } else if spec.is_none() {
        Some("the selected removal model is unknown".to_string())
    } else if !prefs.enabled_models.iter().any(|model| model == id) {
        Some("the selected removal model is disabled".to_string())
    } else if !service.model_runtime_available(id) {
        Some("local AI inference is unavailable in this build".to_string())
    } else if let Some(reason) = context_reason(s, p) {
        Some(reason)
    } else if let Some(status) = service.status().into_iter().find(|status| status.id == id) {
        if status.busy {
            Some("local AI is busy".into())
        } else if !status.available {
            Some(status.reason.unwrap_or_else(|| "model is not installed".into()))
        } else {
            None
        }
    } else {
        Some("the selected removal model is unavailable".into())
    };
    if let Some(reason) = reason {
        return if automatic { Ok(Method::Classical { fallback: Some(reason) }) } else { Err(bad(CMD, reason)) };
    }
    // No catalog entry has passed the frozen quality and performance gates for Auto yet.
    if automatic {
        return Ok(Method::Classical { fallback: Some("the selected model is not yet qualified for Automatic Remove".into()) });
    }
    let sha256 = service.model_sha256(id).map_err(|error| bad(CMD, error.to_string()))?;
    Ok(Method::Ai { id: id.to_string(), sha256, automatic })
}

/// Conservative allowance for all simultaneously live caller buffers. It is a per-operation
/// admission policy, not a process RSS cap. The model's own allowance comes from the catalog.
#[derive(Clone, Copy)]
pub(super) struct Admission {
    pub(super) transient_bytes: u64,
    pub(super) budget_bytes: u64,
    pub(super) required_bytes: u64,
}

pub(super) fn admission(plan: &Plan, method: &Method, service: &photocraft_ml::RemovalService, budget_bytes: u64) -> Result<Option<Admission>> {
    let ids = match method {
        Method::Ai { id, .. } => vec![id.as_str()],
        Method::Hybrid => vec!["lama", "osor-sdxl"],
        _ => return Ok(None),
    };
    let crop_pixels = if matches!(method, Method::Hybrid) {
        plan.regions.iter().try_fold(0u64, |sum, region| {
            u64::from(region.window.width())
                .checked_mul(u64::from(region.window.height()))
                .and_then(|n| n.checked_mul(region.models.len() as u64))
                .and_then(|n| sum.checked_add(n))
        })
    } else {
        u64::from(plan.window.width()).checked_mul(u64::from(plan.window.height()))
    };
    let pixels = crop_pixels
        .filter(|pixels| *pixels > 0 && *pixels <= photocraft_ml::MAX_INPUT_PIXELS)
        .ok_or_else(|| bad(CMD, "the AI sampling window is too large"))?;
    // Coverage renders strokes one at a time, and its tiled extent can extend outside the
    // canvas. Include its largest bounded footprint as well as every crop/output buffer.
    let stroke_pixels =
        plan.strokes.iter().map(stroke_bounds).map(|bounds| u64::from(bounds.width()).saturating_mul(u64::from(bounds.height()))).max().unwrap_or(0);
    const FIXED_BYTES: u64 = 128 * 1024 * 1024;
    const CROP_BYTES_PER_PIXEL: u64 = 128; // selection, hole, read/composite, input RGB+mask, output and write buffers
    const STROKE_BYTES_PER_PIXEL: u64 = 32; // dense coverage plus tiled renderer allowance
    let transient_bytes = pixels
        .checked_mul(CROP_BYTES_PER_PIXEL)
        .and_then(|bytes| stroke_pixels.checked_mul(STROKE_BYTES_PER_PIXEL).and_then(|stroke| bytes.checked_add(stroke)))
        .and_then(|bytes| bytes.checked_add(plan.stroke_work_bytes))
        .and_then(|bytes| bytes.checked_add(FIXED_BYTES))
        .ok_or_else(|| bad(CMD, "AI Remove memory estimate is too large"))?;
    let memory_mib = ids
        .iter()
        .map(|id| service.catalog().iter().find(|model| model.id == *id).map(|model| model.memory_mib).ok_or_else(|| bad(CMD, "unknown removal model")))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let required_bytes = u64::from(memory_mib)
        .checked_mul(1024 * 1024)
        .and_then(|bytes| bytes.checked_add(transient_bytes))
        .ok_or_else(|| bad(CMD, "AI Remove memory estimate is too large"))?;
    if required_bytes > budget_bytes {
        return Err(bad(
            CMD,
            format!(
                "AI Remove needs an estimated {} MiB, exceeding the {} MiB Memory Usage budget",
                required_bytes.div_ceil(1024 * 1024),
                budget_bytes / (1024 * 1024)
            ),
        ));
    }
    Ok(Some(Admission { transient_bytes, budget_bytes, required_bytes }))
}

pub(super) enum Content {
    Classical(Option<(Region, Vec<bool>)>),
    Ai(AiContent),
}

pub(super) struct AiPatch {
    pub rgb: Vec<f32>,
    pub coverage: Vec<f32>,
    pub rect: Rect,
}

/// Output patches drop before their process-wide reservation.
pub(super) struct AiContent {
    pub patches: Vec<AiPatch>,
    pub _reservation: photocraft_ml::RemovalReservation,
}

pub(super) struct Outcome {
    pub content: Content,
    pub metadata: Option<Value>,
}

fn classical(doc: &Document, plan: &Plan, p: &Value, ctx: &crate::jobs::JobCtx, fallback: Option<String>, legacy: bool) -> Result<Outcome> {
    let mut snapshot = doc.clone();
    let filled = run_remove(&mut snapshot, plan, p, ctx, "Remove")?;
    let metadata = (!legacy).then(|| json!({"effectiveMethod":"classical", "fallbackReason":fallback}));
    Ok(Outcome { content: Content::Classical(filled), metadata })
}

pub(super) fn work(
    doc: &Document,
    plan: &Plan,
    p: &Value,
    method: Method,
    service: Arc<photocraft_ml::RemovalService>,
    admission: Option<Admission>,
    ctx: &crate::jobs::JobCtx,
) -> Result<Outcome> {
    if matches!(method, Method::Hybrid) {
        return super::hybrid::work(doc, plan, p, service, admission.ok_or_else(|| bad(CMD, "hybrid Remove was not admitted"))?, ctx);
    }
    let (id, sha256, automatic) = match method {
        Method::Hybrid => return Err(bad(CMD, "invalid hybrid dispatch")),
        Method::Legacy => return classical(doc, plan, p, ctx, None, true),
        Method::Classical { fallback } => return classical(doc, plan, p, ctx, fallback, false),
        Method::Ai { id, sha256, automatic } => (id, sha256, automatic),
    };
    let admission = admission.ok_or_else(|| bad(CMD, "AI Remove was not admitted"))?;
    ctx.check()?;
    // The process-wide lease covers the model and every caller-owned crop/output buffer. It is
    // acquired in the worker before selection, hole, source-region or RGB allocations.
    let mut permit = service
        .admit(&id, admission.transient_bytes, admission.budget_bytes)
        .map_err(|error| if matches!(error, photocraft_ml::Error::Cancelled) { EngineError::Cancelled } else { EngineError::Other(error.to_string()) })?;
    ctx.check()?;
    let installed_hash = service.model_sha256(&id).map_err(|error| bad(CMD, error.to_string()))?;
    if installed_hash != sha256 {
        return Err(bad(CMD, "the installed model changed before AI Remove acquired admission"));
    }
    let selection: Option<Vec<f32>> = doc.selection.as_ref().map(|selection| {
        let channels = selection.channels().max(1);
        selection.read_region(plan.window).chunks_exact(channels).filter_map(|pixel| pixel.first().copied()).collect()
    });
    if selection.as_ref().is_some_and(|values| values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))) {
        return Err(bad(CMD, "selection coverage must be finite and between 0 and 1"));
    }
    let hole = hole(plan, doc.selection.as_ref())?;
    if !hole.iter().any(|v| *v) {
        return Err(bad(CMD, "the stroke doesn't cover anything that can change (check the selection)"));
    }
    if hole.iter().all(|v| *v) {
        return Err(bad(CMD, "nothing is left around the stroke to fill it from"));
    }
    ctx.check()?;
    let (image, fmt) = {
        let surf = crate::channel_cmds::read_target_surface(doc, plan.id, p)?;
        let fmt = if plan.all_layers { PixelFormat::new(surf.format().mode, surf.format().sample, true) } else { surf.format() };
        let image = if plan.all_layers {
            // Render this region directly from the immutable snapshot. Cloning the whole document
            // merely to sample all layers would allocate a tile map for every source layer.
            let composite = photocraft_compose::render(doc, plan.window);
            let mut image = Region::new(plan.window, fmt.channels());
            if composite.px.len() != image.data.len() / fmt.channels() {
                return Err(bad(CMD, "could not render the AI sampling window"));
            }
            let mut encoded = [0.0f32; 8];
            for (dest, pixel) in image.data.chunks_exact_mut(fmt.channels()).zip(composite.px) {
                let n = from_rgba_into(&fmt, pixel, &mut encoded);
                let samples = encoded.get(..n).ok_or_else(|| bad(CMD, "invalid composite pixel format"))?;
                if samples.len() != dest.len() {
                    return Err(bad(CMD, "invalid composite pixel channels"));
                }
                dest.copy_from_slice(samples);
            }
            image
        } else {
            Region::read(surf, plan.window)
        };
        (image, fmt)
    };
    if fmt.mode != ColorMode::Rgb || image.ch != fmt.channels() {
        return Err(bad(CMD, "invalid AI sampling format"));
    }
    let opaque = !fmt.alpha || image.data.chunks_exact(image.ch).zip(&hole).all(|(px, hole)| *hole || px.get(3).is_some_and(|a| *a == 1.0));
    if !opaque {
        let reason = "AI Remove requires opaque sampled context";
        if automatic {
            drop(permit);
            return classical(doc, plan, p, ctx, Some(reason.into()), false);
        }
        return Err(bad(CMD, reason));
    }
    let mut rgb = Vec::with_capacity(hole.len().saturating_mul(3));
    for px in image.data.chunks_exact(image.ch) {
        for value in px.iter().take(3) {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(bad(CMD, "invalid sRGB sample in AI context"));
            }
            rgb.push(*value);
        }
    }
    let mask = hole.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
    let input = RemovalInput { width: plan.window.width(), height: plan.window.height(), rgb, mask };
    let RemovalOutput { rgb, edit_alpha, inference_scale, load_ms, forward_ms } = permit
        .infer(&input, ctx)
        .map_err(|e| if matches!(e, photocraft_ml::Error::Cancelled) { EngineError::Cancelled } else { EngineError::Other(e.to_string()) })?;
    ctx.check()?;
    if rgb.len() != input.rgb.len()
        || rgb.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || edit_alpha.as_ref().is_some_and(|alpha| alpha.len() != hole.len() || alpha.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
        || (id == "osor-sdxl" && edit_alpha.is_none())
        || !inference_scale.is_finite()
        || inference_scale <= 0.0
        || !load_ms.is_finite()
        || load_ms < 0.0
        || !forward_ms.is_finite()
        || forward_ms < 0.0
    {
        return Err(EngineError::Other("AI adapter returned invalid image data".into()));
    }
    // OSOR RGB is already blended by its latent edit alpha. Selection coverage is the only
    // extra blend: effects may change the full crop; ordinary removal stays inside the hole.
    let remove_effects = p.get("removeEffects").and_then(Value::as_bool) == Some(true);
    let coverage = hole
        .iter()
        .enumerate()
        .map(|(i, h)| if remove_effects || *h { selection.as_ref().and_then(|s| s.get(i)).copied().unwrap_or(1.0).clamp(0.0, 1.0) } else { 0.0 })
        .collect();
    let metadata = json!({"effectiveMethod":"ai", "modelId":id, "modelSha256":sha256, "inferenceScale":inference_scale,
        "loadMs":load_ms, "forwardMs":forward_ms, "crop":damage_json(plan.window), "fallbackReason":Value::Null,
        "estimatedTransientBytes":admission.transient_bytes, "estimatedWorkingSetBytes":admission.required_bytes,
        "memoryBudgetBytes":admission.budget_bytes});
    ctx.check()?;
    let reservation = permit.into_reservation();
    Ok(Outcome {
        content: Content::Ai(AiContent { patches: vec![AiPatch { rgb, coverage, rect: plan.window }], _reservation: reservation }),
        metadata: Some(metadata),
    })
}

pub(super) fn apply_content(doc: &mut Document, id: Option<LayerId>, p: &Value, content: &AiContent, new_layer: bool) -> Result<(Rect, Option<LayerId>)> {
    let mut target = id;
    let mut created = None;
    let mut damage = None::<Rect>;
    for patch in &content.patches {
        let (rect, new_id) = apply(doc, target, p, patch.rect, &patch.rgb, &patch.coverage, new_layer && created.is_none())?;
        if let Some(new_id) = new_id {
            created = Some(new_id);
            target = Some(new_id);
        }
        damage = Some(damage.map_or(rect, |old| old.union(&rect)));
    }
    Ok((damage.ok_or_else(|| bad(CMD, "empty AI Remove candidate"))?, created))
}

pub(super) fn apply(
    doc: &mut Document,
    id: Option<LayerId>,
    p: &Value,
    rect: Rect,
    rgb: &[f32],
    coverage: &[f32],
    new_layer: bool,
) -> Result<(Rect, Option<LayerId>)> {
    if rgb.len() != coverage.len().saturating_mul(3) {
        return Err(EngineError::Other("AI output dimensions disagree".into()));
    }
    if new_layer {
        let fmt = PixelFormat::new(ColorMode::Rgb, doc.depth, true);
        let mut data = vec![0.0; coverage.len().saturating_mul(4)];
        for ((dest, src), &w) in data.as_chunks_mut::<4>().0.iter_mut().zip(rgb.as_chunks::<3>().0.iter()).zip(coverage) {
            if w > 0.0 {
                dest[..3].copy_from_slice(src);
                dest[3] = w;
            }
        }
        let mut layer = Layer::raster(doc.next_layer_name("Layer"), fmt);
        crate::pixels_mut(&mut layer)?.try_write_region(rect, &data)?;
        let new_id = crate::layer_place::insert_new(doc, id, layer, crate::layer_place::Place::Above);
        return Ok((rect, Some(new_id)));
    }
    let (surface, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
    let fmt = surface.format();
    let n = fmt.channels();
    if fmt.mode != ColorMode::Rgb || !matches!(n, 3 | 4) {
        return Err(bad(CMD, "invalid AI target format"));
    }
    let mut dest = surface.read_region(rect);
    if dest.len() != coverage.len().saturating_mul(n) {
        return Err(bad(CMD, "AI target dimensions disagree"));
    }
    for ((pixel, src), &w) in dest.chunks_exact_mut(n).zip(rgb.as_chunks::<3>().0.iter()).zip(coverage) {
        if w <= 0.0 {
            continue;
        }
        let a = if fmt.alpha { pixel.get(3).copied().unwrap_or(0.0) } else { 1.0 };
        if lock && a <= 0.0 {
            continue;
        }
        let out_alpha = if lock { a } else { a + (1.0 - a) * w };
        for (d, s) in pixel.iter_mut().take(3).zip(src) {
            *d = if lock {
                *d + (*s - *d) * w
            } else if out_alpha > 0.0 {
                (*d * a * (1.0 - w) + *s * w) / out_alpha
            } else {
                0.0
            };
        }
        if fmt.alpha
            && !lock
            && let Some(alpha) = pixel.get_mut(3)
        {
            *alpha = out_alpha;
        }
    }
    surface.try_write_region(rect, &dest)?;
    Ok((rect, None))
}
