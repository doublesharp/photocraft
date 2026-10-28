//! Regional Remove: every forward samples the same immutable source. Only disjoint final
//! ownership masks are composited, so fractional selection and learned alpha are never doubled.
use super::smart::{Admission, AiContent, AiPatch, Content, Outcome};
use super::{CMD, EngineError, PixelFormat, Plan, Rect, Region, Result, Stroke, Value, bad, damage_json, hole_for, json, sampling_margin, stroke_bounds};
use photocraft_color::ColorMode;
use photocraft_ml::{RemovalInput, RemovalOutput};
use photocraft_raster::from_rgba_into;
use std::sync::Arc;

const MAX_REGIONS: usize = 32;
const MAX_FORWARDS: usize = 32;

pub(super) struct ModelRegion {
    id: &'static str,
    indices: Vec<usize>,
}
pub(super) struct RegionPlan {
    pub window: Rect,
    pub models: Vec<ModelRegion>,
}

fn hint(value: Option<&Value>) -> Result<&str> {
    match value {
        None => Ok("auto"),
        Some(Value::String(value)) if matches!(value.as_str(), "auto" | "detail" | "object") => Ok(value),
        _ => Err(bad(CMD, "repairType must be auto, detail or object")),
    }
}

pub(super) fn validate_hints(p: &Value) -> Result<()> {
    hint(p.get("repairType"))?;
    if let Some(strokes) = p.get("strokes").and_then(Value::as_array) {
        for stroke in strokes {
            hint(stroke.get("repairType"))?;
        }
    }
    if let Some(value) = p.get("regionalPlan") {
        let regions = value
            .as_array()
            .filter(|regions| !regions.is_empty() && regions.len() <= MAX_REGIONS)
            .ok_or_else(|| bad(CMD, "regionalPlan must contain 1..32 frozen regions"))?;
        let mut indices = 0usize;
        for region in regions {
            let region = region
                .as_object()
                .filter(|region| region.len() == 2 && region.contains_key("crop") && region.contains_key("components"))
                .ok_or_else(|| bad(CMD, "invalid frozen region fields"))?;
            let crop = region
                .get("crop")
                .and_then(Value::as_array)
                .filter(|crop| crop.len() == 4 && crop.iter().all(|n| n.as_i64().is_some_and(|n| i32::try_from(n).is_ok())));
            if crop.is_none() {
                return Err(bad(CMD, "invalid frozen region crop"));
            }
            let components = region
                .get("components")
                .and_then(Value::as_array)
                .filter(|parts| !parts.is_empty() && parts.len() <= 2)
                .ok_or_else(|| bad(CMD, "invalid frozen region components"))?;
            for component in components {
                let component = component
                    .as_object()
                    .filter(|part| part.len() == 2 && part.contains_key("modelId") && part.contains_key("strokeIndices"))
                    .ok_or_else(|| bad(CMD, "invalid frozen component fields"))?;
                if !component.get("modelId").and_then(Value::as_str).is_some_and(|id| matches!(id, "lama" | "osor-sdxl")) {
                    return Err(bad(CMD, "unknown frozen region model"));
                }
                let strokes = component
                    .get("strokeIndices")
                    .and_then(Value::as_array)
                    .filter(|strokes| {
                        !strokes.is_empty() && strokes.len() <= 1024 && strokes.iter().all(|index| index.as_u64().is_some_and(|index| index < 1024))
                    })
                    .ok_or_else(|| bad(CMD, "invalid frozen region stroke indices"))?;
                indices = indices.saturating_add(strokes.len());
                if indices > 1024 {
                    return Err(bad(CMD, "too many frozen stroke indices"));
                }
            }
        }
    }
    if p.get("modelRevisions").is_some_and(|value| {
        value.as_object().is_none_or(|revisions| {
            revisions.len() != 2
                || revisions.iter().any(|(id, hash)| {
                    !matches!(id.as_str(), "lama" | "osor-sdxl")
                        || hash.as_str().is_none_or(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
                })
        })
    }) {
        return Err(bad(CMD, "invalid frozen model revisions"));
    }
    Ok(())
}

pub(super) fn regions(strokes: &[Stroke], p: &Value, canvas: Rect) -> Result<Vec<RegionPlan>> {
    let mut groups: Vec<(Rect, Vec<usize>)> = Vec::new();
    // Actual stroke footprints define connected components. A group's bounding rectangle may
    // contain untouched space; that empty space must not join otherwise independent marks.
    for (index, stroke) in strokes.iter().enumerate() {
        let mut bounds = stroke_bounds(stroke).intersect(&canvas);
        if bounds.is_empty() {
            continue;
        }
        let mut members = vec![index];
        let mut at = 0;
        while at < groups.len() {
            let touching = groups.get(at).is_some_and(|(_, indices)| {
                indices
                    .iter()
                    .filter_map(|index| strokes.get(*index))
                    .any(|other| !stroke_bounds(stroke).inflate(1).intersect(&stroke_bounds(other)).is_empty())
            });
            if touching {
                let (other, mut indices) = groups.remove(at);
                bounds = bounds.union(&other);
                members.append(&mut indices);
                at = 0;
            } else {
                at += 1;
            }
        }
        members.sort_unstable();
        groups.push((bounds, members));
    }
    groups.sort_by_key(|(_, members)| members.first().copied().unwrap_or(usize::MAX));
    if groups.is_empty() || groups.len() > MAX_REGIONS {
        return Err(bad(CMD, "regional Remove needs 1..32 independent regions; use fewer strokes"));
    }
    let top = hint(p.get("repairType"))?;
    let mut result = Vec::with_capacity(groups.len());
    let mut total_pixels = 0u64;
    let mut forwards = 0usize;
    for (bounds, members) in groups {
        let window = bounds.inflate(sampling_margin(bounds).min(super::MAX_MARGIN)).intersect(&canvas);
        let mut detail = Vec::new();
        let mut object = Vec::new();
        for index in members {
            let stroke = strokes.get(index).ok_or_else(|| bad(CMD, "invalid regional stroke index"))?;
            let value = p.get("strokes").and_then(Value::as_array).and_then(|strokes| strokes.get(index)).and_then(|stroke| stroke.get("repairType"));
            let repair = if value.is_some() { hint(value)? } else { top };
            let footprint = stroke_bounds(stroke);
            let minor = footprint.width().min(footprint.height());
            let major = footprint.width().max(footprint.height());
            // This is a bounded geometry choice, not semantic object recognition. Thin elongated
            // footprints use texture repair; compact footprints use the object-removal family.
            let use_detail = repair == "detail" || (repair == "auto" && minor <= 64 && u64::from(major) >= u64::from(minor).saturating_mul(3));
            if use_detail {
                detail.push(index);
            } else {
                object.push(index);
            }
        }
        let mut models = Vec::new();
        if !detail.is_empty() {
            models.push(ModelRegion { id: "lama", indices: detail });
        }
        if !object.is_empty() {
            models.push(ModelRegion { id: "osor-sdxl", indices: object });
        }
        forwards = forwards.saturating_add(models.len());
        total_pixels = total_pixels
            .checked_add(u64::from(window.width()).saturating_mul(u64::from(window.height())).saturating_mul(models.len() as u64))
            .ok_or_else(|| bad(CMD, "regional crop estimate overflow"))?;
        if forwards > MAX_FORWARDS || total_pixels > super::MAX_PIXELS {
            return Err(bad(CMD, "regional Remove exceeds 32 model forwards or 16 million aggregate crop pixels; use fewer or smaller strokes"));
        }
        result.push(RegionPlan { window, models });
    }
    Ok(result)
}

fn metadata(plan: &Plan) -> Value {
    Value::Array(
        plan.regions
            .iter()
            .map(|region| {
                json!({"crop":damage_json(region.window),"components":region.models.iter().map(|model|
        json!({"modelId":model.id,"strokeIndices":model.indices})).collect::<Vec<_>>() })
            })
            .collect(),
    )
}

fn ml_error(error: photocraft_ml::Error) -> EngineError {
    if matches!(error, photocraft_ml::Error::Cancelled) { EngineError::Cancelled } else { EngineError::Other(error.to_string()) }
}

struct Patch {
    data: AiPatch,
    hole: Vec<bool>,
    osor: bool,
}
impl Patch {
    fn rank(&self, x: i32, y: i32, effects: bool) -> u8 {
        if !self.data.rect.contains(x, y) {
            return 0;
        }
        let i = (y - self.data.rect.y0) as usize * self.data.rect.width() as usize + (x - self.data.rect.x0) as usize;
        if self.hole.get(i).copied().unwrap_or(false) {
            if self.osor { 3 } else { 2 }
        } else if self.osor && effects {
            1
        } else {
            0
        }
    }
}

pub(super) fn work(
    doc: &photocraft_doc::Document,
    plan: &Plan,
    p: &Value,
    service: Arc<photocraft_ml::RemovalService>,
    admission: Admission,
    ctx: &crate::jobs::JobCtx,
) -> Result<Outcome> {
    ctx.check()?;
    let mut permit = service.admit_models(&["lama", "osor-sdxl"], admission.transient_bytes, admission.budget_bytes).map_err(ml_error)?;
    let revisions = json!({"lama":permit.model_sha256("lama").map_err(ml_error)?,"osor-sdxl":permit.model_sha256("osor-sdxl").map_err(ml_error)?});
    let frozen = metadata(plan);
    if p.get("regionalPlan").is_some_and(|value| value != &frozen) || p.get("modelRevisions").is_some_and(|value| value != &revisions) {
        return Err(bad(CMD, "the frozen regional plan or installed model revisions changed"));
    }
    let mut patches = Vec::new();
    let mut timings = Vec::new();
    // Models are grouped so only one warm family is needed, including when context windows
    // overlap. Every input is independently extracted from doc, never from a preceding output.
    for id in ["lama", "osor-sdxl"] {
        for (region_index, region) in plan.regions.iter().enumerate() {
            for model in region.models.iter().filter(|model| model.id == id) {
                ctx.check()?;
                let rect = region.window;
                let selection = doc.selection.as_ref().map(|selection| {
                    let channels = selection.channels().max(1);
                    selection.read_region(rect).chunks_exact(channels).filter_map(|pixel| pixel.first().copied()).collect::<Vec<_>>()
                });
                if selection.as_ref().is_some_and(|values| values.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))) {
                    return Err(bad(CMD, "selection coverage must be finite and between 0 and 1"));
                }
                let hole = hole_for(model.indices.iter().filter_map(|index| plan.strokes.get(*index)), rect, plan.close_loops, doc.selection.as_ref())?;
                if !hole.iter().any(|h| *h) {
                    continue;
                }
                if hole.iter().all(|h| *h) {
                    return Err(bad(CMD, "nothing is left around a region to fill it from"));
                }
                let surface = crate::channel_cmds::read_target_surface(doc, plan.id, p)?;
                let format = if plan.all_layers { PixelFormat::new(surface.format().mode, surface.format().sample, true) } else { surface.format() };
                let image = if plan.all_layers {
                    let composite = photocraft_compose::render(doc, rect);
                    let mut image = Region::new(rect, format.channels());
                    if composite.px.len() != image.data.len() / format.channels() {
                        return Err(bad(CMD, "invalid regional composite"));
                    }
                    let mut encoded = [0.0f32; 8];
                    for (dest, pixel) in image.data.chunks_exact_mut(format.channels()).zip(composite.px) {
                        let n = from_rgba_into(&format, pixel, &mut encoded);
                        let source =
                            encoded.get(..n).filter(|source| source.len() == dest.len()).ok_or_else(|| bad(CMD, "invalid regional composite channels"))?;
                        dest.copy_from_slice(source);
                    }
                    image
                } else {
                    Region::read(surface, rect)
                };
                if format.mode != ColorMode::Rgb || image.ch != format.channels() {
                    return Err(bad(CMD, "invalid regional sampling format"));
                }
                if format.alpha && image.data.chunks_exact(image.ch).zip(&hole).any(|(pixel, hole)| !*hole && pixel.get(3).is_none_or(|alpha| *alpha != 1.0)) {
                    return Err(bad(CMD, "AI Remove requires opaque sampled context"));
                }
                let mut rgb = Vec::with_capacity(hole.len().saturating_mul(3));
                for pixel in image.data.chunks_exact(image.ch) {
                    for value in pixel.iter().take(3) {
                        if !value.is_finite() || !(0.0..=1.0).contains(value) {
                            return Err(bad(CMD, "invalid sRGB sample in regional context"));
                        }
                        rgb.push(*value);
                    }
                }
                let input = RemovalInput { width: rect.width(), height: rect.height(), rgb, mask: hole.iter().map(|h| if *h { 1.0 } else { 0.0 }).collect() };
                let RemovalOutput { rgb, edit_alpha, inference_scale, load_ms, forward_ms } = permit.infer(id, &input, ctx).map_err(ml_error)?;
                ctx.check()?;
                if rgb.len() != input.rgb.len()
                    || rgb.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    || edit_alpha.as_ref().is_some_and(|values| values.len() != hole.len() || values.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
                    || (id == "osor-sdxl" && edit_alpha.is_none())
                    || !inference_scale.is_finite()
                    || inference_scale <= 0.0
                    || !load_ms.is_finite()
                    || load_ms < 0.0
                    || !forward_ms.is_finite()
                    || forward_ms < 0.0
                {
                    return Err(EngineError::Other("AI adapter returned invalid regional image data".into()));
                }
                // OSOR RGB is already latent blended. Retain only selection coverage; final
                // ownership below decides which source region may write each pixel.
                let coverage = (0..hole.len()).map(|i| selection.as_ref().and_then(|s| s.get(i)).copied().unwrap_or(1.0)).collect();
                patches.push(Patch { data: AiPatch { rgb, coverage, rect }, hole, osor: id == "osor-sdxl" });
                timings.push(json!({"region":region_index,"modelId":id,"inferenceScale":inference_scale,"loadMs":load_ms,"forwardMs":forward_ms}));
            }
        }
    }
    if patches.is_empty() {
        return Err(bad(CMD, "the strokes do not cover anything that can change; check the selection"));
    }
    let effects = p.get("removeEffects").and_then(Value::as_bool) == Some(true);
    // Freeze disjoint ownership before touching any destination. Core defects outrank all
    // effect bands; OSOR core wins model overlap. Earlier planned regions break equal ranks.
    for index in 0..patches.len() {
        let rect = patches.get(index).map(|patch| patch.data.rect).ok_or_else(|| bad(CMD, "invalid regional patch"))?;
        let overlaps = patches
            .iter()
            .enumerate()
            .filter(|(other, patch)| *other != index && !patch.data.rect.intersect(&rect).is_empty())
            .map(|(other, _)| other)
            .collect::<Vec<_>>();
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let own_rank = patches.get(index).map(|patch| patch.rank(x, y, effects)).unwrap_or(0);
                let loses = own_rank == 0
                    || overlaps.iter().any(|other| {
                        patches.get(*other).is_some_and(|patch| {
                            let rank = patch.rank(x, y, effects);
                            rank > own_rank || (rank == own_rank && rank > 0 && *other < index)
                        })
                    });
                if loses {
                    let at = (y - rect.y0) as usize * rect.width() as usize + (x - rect.x0) as usize;
                    if let Some(value) = patches.get_mut(index).and_then(|patch| patch.data.coverage.get_mut(at)) {
                        *value = 0.0;
                    }
                }
            }
        }
        ctx.check()?;
    }
    let result = json!({"effectiveMethod":"hybrid","workflow":"recommended","regionalPlan":frozen,"modelRevisions":revisions,"regionResults":timings,
        "fallbackReason":Value::Null,"preview":true,"estimatedTransientBytes":admission.transient_bytes,"estimatedWorkingSetBytes":admission.required_bytes,
        "memoryBudgetBytes":admission.budget_bytes});
    ctx.check()?;
    Ok(Outcome {
        content: Content::Ai(AiContent { patches: patches.into_iter().map(|patch| patch.data).collect(), _reservation: permit.into_reservation() }),
        metadata: Some(result),
    })
}
