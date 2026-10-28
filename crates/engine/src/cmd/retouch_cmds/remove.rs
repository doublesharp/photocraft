//! The Remove Tool as a command: paint over something and it is replaced by its surroundings.
//!
//! The stroke's footprint is the hole. A stroke that closes around an object also removes what it
//! encloses (`closeLoops`), and the hole grows by [`GROW`] pixels so the object's soft edge and
//! halo go with it. The hole is completed from a window around the stroke by patch-based
//! completion with texture features (`photocraft_algo::remove`, `photocraft_algo::nonlocal`). The
//! selection, when there is one, limits what changes. Large holes take seconds, so the command
//! runs as a background job (#210): with progress, cancellable, and the document unchanged until
//! it applies, in one undo step.
//!
//! Several strokes can go in one removal (`strokes`): Photoshop's *Remove after each stroke*
//! off, where the strokes collect and are applied together, so the fill sees every hole at once
//! instead of sampling from an area the next stroke is about to remove. With `newLayer` the fill
//! lands on a new layer above the target, only where the hole is, and the target stays as it was
//! (Photoshop's *Create a new layer*; with Sample All Layers it fills from the composite).

use super::content_aware_move::{Target, replace};
use super::patch::coverage;
use super::{
    BlendMode, BrushSettings, Document, EngineError, LayerContent, LayerId, PixelFormat, Rect, Region, Result, SampleLayers, SampleType, Session, Stroke,
    StrokePoint, Surface, Value, alpha_index, bad, clamp_samples, composite_region, damage_json, flag, json, parse_brush_with, reject_huge_stroke,
    targets_pixels,
};
use photocraft_doc::Layer;

mod brush;
mod hybrid;
mod smart;
mod work;

const CMD: &str = "paint.remove";

/// Largest stroke, in pixels of its bounding box (as the Patch Tool's and Content-Aware Move's).
/// Checked before the stroke's coverage is rendered, which allocates its whole bounding box, also
/// where it leaves the canvas.
const MAX_PIXELS: u64 = 16 << 20;

/// Cap on the sampling margin around the stroke (the window grows with the stroke).
const MAX_MARGIN: i32 = 512;

/// How far the hole grows beyond the stroke, in pixels.
const GROW: usize = 2;

/// Seed of the completion, so a removal is reproducible.
const SEED: u64 = 0x0052_e30e;

/// The sampling margin around the stroke: three quarters of its larger side, at least 32 pixels
/// (in u64, so a huge extent can't overflow; a margin beyond i32 saturates).
fn sampling_margin(area: Rect) -> i32 {
    let ext = u64::from(area.width().max(area.height()));
    i32::try_from(ext * 3 / 4).unwrap_or(i32::MAX).max(32)
}

/// What a removal does, checked before any work starts.
struct Plan {
    regions: Vec<hybrid::RegionPlan>,
    id: Option<LayerId>,
    /// One stroke (`points`), or several applied together (`strokes`).
    strokes: Vec<Stroke>,
    /// Largest simultaneously live generated-dab, walker and smoothing buffers.
    stroke_work_bytes: u64,
    /// Where the fill samples from: the strokes plus a margin, on the canvas.
    window: Rect,
    close_loops: bool,
    all_layers: bool,
    /// The fill goes to a new layer above the target instead of into it.
    new_layer: bool,
}

/// Conservative bounds of every rendered dab, before coverage is allocated. Keep the tip
/// reach in step with `BrushContext::dab_rect`: soft tips extend past the nominal radius and
/// rotated sampled tips reach their corners. Dynamics can shrink the tip and scatter its centre.
fn stroke_bounds(stroke: &Stroke) -> Rect {
    let b = stroke.brush.bounded_for_render();
    let reach = |size: f32, hardness: f32, tip: &photocraft_paint::TipShape, aliased: bool| {
        let radius = (size / 2.0).max(0.5);
        f64::from(if tip.bitmap().is_some_and(|bitmap| bitmap.is_valid()) {
            radius * std::f32::consts::SQRT_2
        } else if aliased {
            radius
        } else {
            photocraft_paint::tip_reach(radius, hardness).max(radius)
        })
    };
    let scatter = if b.scattering.enabled { f64::from((b.size / 2.0).max(0.5)) * f64::from(b.scattering.scatter.jitter) } else { 0.0 };
    let primary = reach(b.size, b.hardness, &b.tip, b.aliased) + scatter;
    let dual = if b.dual_brush.enabled {
        let d = &b.dual_brush;
        reach(d.size, d.hardness, &d.tip, false) + f64::from((d.size / 2.0).max(0.5)) * f64::from(d.scatter)
    } else {
        0.0
    };
    // Dab rectangles round outward and include an antialiasing pixel; the hole grows by GROW.
    let r = primary.max(dual) + GROW as f64 + 3.0;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for q in &stroke.points {
        (x0, y0, x1, y1) = (x0.min(q.x), y0.min(q.y), x1.max(q.x), y1.max(q.y));
    }
    // `check_coords` keeps points within ±MAX_COORD and sizes are capped, so these fit an i32.
    Rect::new((x0 - r).floor() as i32, (y0 - r).floor() as i32, (x1 + r).ceil() as i32, (y1 + r).ceil() as i32)
}

/// The strokes of the request: `strokes` (several, each a list of points, sharing the brush
/// and the target) or the one in `points`.
fn parse_strokes(s: &Session, p: &Value) -> Result<(Vec<Stroke>, Option<LayerId>)> {
    // The new explicit-method contract uses a complete, frozen brush. Legacy requests keep
    // resolving their brush exactly as they did before this command gained the new options.
    let common = if p.get("method").is_some() { p.get("brush").map(brush::decode).transpose()? } else { None };
    let Some(list) = p.get("strokes") else {
        let (stroke, id) = parse_brush_with(s, p, CMD, common)?;
        reject_huge_stroke(CMD, &stroke)?;
        return Ok((vec![stroke], id));
    };
    const ENTRY: &str = "each `strokes` entry is a point list or an object with points and optional brush settings";
    let list = list.as_array().ok_or_else(|| bad(CMD, ENTRY))?;
    if list.is_empty() {
        return Err(bad(CMD, "`strokes` is empty"));
    }
    let mut strokes = Vec::with_capacity(list.len());
    let mut id = None;
    // Do not clone the entire list for each stroke: only the shared options and this stroke
    // are needed. This keeps multi-stroke requests linear in their input size.
    let shared: serde_json::Map<String, Value> = p
        .as_object()
        .ok_or_else(|| bad(CMD, "params must be an object"))?
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "strokes" | "brush" | "regionalPlan" | "modelRevisions"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    for entry in list {
        let mut one = Value::Object(shared.clone());
        let Some(o) = one.as_object_mut() else { return Err(bad(CMD, "params must be an object")) };
        o.remove("strokes");
        let explicit = if p.get("method").is_some() { entry.get("brush").map(brush::decode).transpose()?.or_else(|| common.clone()) } else { None };
        match entry {
            // A stroke drawn with another brush than the rest (the waiting strokes of Remove after
            // each stroke off keep the size, hardness and spacing each was drawn with).
            Value::Object(e) => {
                let points = e.get("points").ok_or_else(|| bad(CMD, ENTRY))?;
                o.insert("points".into(), points.clone());
                for k in ["size", "hardness", "spacing"] {
                    if let Some(v) = e.get(k) {
                        o.insert(k.into(), v.clone());
                    }
                }
            }
            points => {
                o.insert("points".into(), points.clone());
            }
        }
        let (stroke, layer) = parse_brush_with(s, &one, CMD, explicit)?;
        reject_huge_stroke(CMD, &stroke)?;
        strokes.push(stroke);
        id = layer;
    }
    Ok((strokes, id))
}

fn plan(s: &Session, p: &Value) -> Result<Plan> {
    let (strokes, id) = parse_strokes(s, p)?;
    let stroke_work_bytes = work::estimate(&strokes)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let bounds = strokes.iter().map(stroke_bounds).reduce(|a, b| a.union(&b)).ok_or_else(|| bad(CMD, "`strokes` is empty"))?;
    let pixels = u64::from(bounds.width()) * u64::from(bounds.height());
    let regional = p.get("method").and_then(Value::as_str) == Some("auto") && s.prefs().generative_remove.workflow == "recommended";
    if !regional && pixels > MAX_PIXELS {
        return Err(bad(CMD, format!("the stroke is too large to remove ({pixels} pixels, at most {MAX_PIXELS})")));
    }
    let area = bounds.intersect(&canvas);
    if area.is_empty() {
        return Err(bad(CMD, "the stroke is outside the canvas"));
    }
    let window = area.inflate(sampling_margin(area).min(MAX_MARGIN)).intersect(&canvas);
    let close_loops = flag(p, "closeLoops", true);
    let all_layers = flag(p, "sampleAllLayers", false) && targets_pixels(p);
    // A mask or channel has no "above": the fill goes where the stroke went.
    let new_layer = flag(p, "newLayer", false) && targets_pixels(p);
    let regions = if regional { hybrid::regions(&strokes, p, canvas)? } else { Vec::new() };
    Ok(Plan { id, strokes, stroke_work_bytes, window, close_loops, all_layers, new_layer, regions })
}

pub(super) fn check_stroke_work(brush: &BrushSettings, points: &[[f64; 3]]) -> Result<()> {
    work::check(brush, points)
}

/// The hole over the plan's window: the strokes' footprints, their enclosed areas and a margin,
/// limited to the selection. Empty when nothing would change.
fn hole(plan: &Plan, sel: Option<&Surface>) -> Result<Vec<bool>> {
    hole_for(plan.strokes.iter(), plan.window, plan.close_loops, sel)
}

fn hole_for<'a>(strokes: impl Iterator<Item = &'a Stroke>, win: Rect, close_loops: bool, sel: Option<&Surface>) -> Result<Vec<bool>> {
    let (w, h) = (win.width() as usize, win.height() as usize);
    let mut m = vec![false; w * h];
    for stroke in strokes {
        let (bounds, cov) = super::coverage(CMD, stroke)?;
        let bw = bounds.width() as usize;
        for (i, c) in cov.iter().enumerate() {
            let (x, y) = (bounds.x0 + (i % bw.max(1)) as i32, bounds.y0 + (i / bw.max(1)) as i32);
            if *c > 0.0 && win.contains(x, y) {
                let j = (y - win.y0) as usize * w + (x - win.x0) as usize;
                if let Some(v) = m.get_mut(j) {
                    *v = true;
                }
            }
        }
    }
    if close_loops {
        m = photocraft_algo::remove::close_loops(w, h, &m);
    }
    m = photocraft_algo::remove::dilate(w, h, &m, GROW);
    if let Some(sel) = sel {
        for (v, c) in m.iter_mut().zip(coverage(sel, win)) {
            *v &= c > 0.0;
        }
    }
    Ok(m)
}

/// The filled window and the hole it fills, computed on `doc` (a snapshot). `None` when the
/// layer's own window is fully transparent: nothing to remove and nothing to fill from.
fn run_remove(doc: &mut Document, plan: &Plan, p: &Value, ctx: &crate::jobs::JobCtx, label: &str) -> Result<Option<(Region, Vec<bool>)>> {
    let hole = hole(plan, doc.selection.as_ref())?;
    if !hole.iter().any(|m| *m) {
        return Err(bad(CMD, "the stroke doesn't cover anything that can change (check the selection)"));
    }
    if hole.iter().all(|m| *m) {
        return Err(bad(CMD, "nothing is left around the stroke to fill it from"));
    }
    ctx.check()?;
    let win = plan.window;
    let (img, fmt) = {
        let (surf, _) = crate::channel_cmds::target_surface(doc, plan.id, p)?;
        let fmt = surf.format();
        let img = if plan.all_layers { composite_region(doc, None, SampleLayers::All, win, fmt) } else { Region::read(surf, win) };
        (img, fmt)
    };
    // A no-op without the fill or a history step, as the other retouch tools on an empty layer.
    if !plan.all_layers && alpha_index(&fmt).is_some_and(|a| img.data.chunks_exact(img.ch).all(|px| px.get(a).is_none_or(|v| *v <= 0.0))) {
        return Ok(None);
    }
    let (w, h) = (win.width() as usize, win.height() as usize);
    let seed = if p.get("method").is_some() { p.get("seed").and_then(Value::as_u64).unwrap_or(SEED) } else { SEED };
    let mut data = ctx
        .stage(0.0, 1.0, label, |ctl| photocraft_algo::remove::remove_with(w, h, img.ch, &img.data, &hole, seed, ctl))
        .map_err(|_| EngineError::Cancelled)?;
    // The fill's seam blending can step slightly past the storable range.
    clamp_samples(&fmt, &mut data);
    Ok(Some((Region { rect: win, ch: img.ch, data }, hole)))
}

/// The fill on a new layer above `id`: the hole's pixels (opaque where the source had no alpha),
/// transparent elsewhere. Returns the damage and the new layer.
fn fill_to_new_layer(doc: &mut Document, id: Option<LayerId>, out: &Region, hole: &[bool]) -> Result<(Rect, LayerId)> {
    let base = doc.pixel_format();
    let fmt = PixelFormat::new(base.mode, base.sample, true);
    let n = fmt.channels();
    let colours = n.saturating_sub(1);
    if out.ch != n && out.ch != colours {
        return Err(EngineError::Other("internal error: the fill's channels don't match the document".into()));
    }
    let mut data = vec![0.0f32; hole.len().saturating_mul(n)];
    for ((dst, src), m) in data.chunks_exact_mut(n).zip(out.data.chunks_exact(out.ch)).zip(hole) {
        if !*m {
            continue;
        }
        for (d, s) in dst.iter_mut().zip(src.iter().take(colours)) {
            *d = *s;
        }
        if let Some(a) = dst.last_mut() {
            *a = if out.ch == n { src.last().copied().unwrap_or(1.0) } else { 1.0 };
        }
    }
    let mut l = Layer::raster(doc.next_layer_name("Layer"), fmt);
    crate::pixels_mut(&mut l)?.write_region(out.rect, &data);
    // Above the sampled layer and in its clipping mask, like Layer via Copy.
    let nid = crate::layer_place::insert_new(doc, id, l, crate::layer_place::Place::Above);
    Ok((out.rect, nid))
}

pub(super) fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    let method = smart::decide(s, p)?;
    let mut params = p.clone();
    if matches!(method, smart::Method::Hybrid) {
        params["preview"] = json!(true);
    }
    let p = &params;
    let plan = plan(s, p)?;
    // Keep the exact rendered brushes for the journal/action step, independent of later tool state.
    let resolved_brushes = if p.get("method").is_some() { Some(brush::snapshots(&plan.strokes)?) } else { None };
    let label = "Remove";
    let service = s.removal_service();
    // Freeze the user's budget at submission; the worker acquires the shared permit before it
    // allocates any pixels, even if this preference changes while the request is waiting.
    let budget_bytes = u64::from(s.prefs().performance.memory_usage_mb).saturating_mul(1024 * 1024);
    let admission = smart::admission(&plan, &method, &service, budget_bytes)?;
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let preview_capture = (p.get("preview").and_then(Value::as_bool) == Some(true)).then(|| crate::removal_preview::Capture::new(s)).transpose()?;
    let preview_source = doc.clone();
    let used_working_profile = doc.icc_profile.is_none();
    let working_rgb = s.color.settings.working_rgb.clone();
    let p = p.clone();
    let id = plan.id;
    let new_layer = plan.new_layer;
    crate::jobs::run(
        s,
        label,
        true,
        {
            let p = p.clone();
            move |ctx| {
                ctx.progress(0.0, label);
                smart::work(&doc, &plan, &p, method, service, admission, ctx)
            }
        },
        move |s, outcome| {
            let mut result = match outcome.content {
                smart::Content::Classical(None) => json!({"changed": false}),
                smart::Content::Classical(Some((out, hole))) if new_layer => {
                    let (damage, nid) = s.edit(label, |doc, active| {
                        let r = fill_to_new_layer(doc, id, &out, &hole)?;
                        *active = Some(r.1);
                        Ok(r)
                    })?;
                    s.hint_damage(damage);
                    json!({"damage": damage_json(damage), "layer":nid.0})
                }
                smart::Content::Classical(Some((out, hole))) => {
                    let damage = s.edit(label, |doc, _| replace(doc, Target { id, all_layers: false }, &p, &out, &hole))?;
                    s.hint_damage(damage);
                    json!({"damage": damage_json(damage)})
                }
                smart::Content::Ai(content) => {
                    // Content owns the reservation until its output buffers are dropped.
                    if used_working_profile && s.color.settings.working_rgb != working_rgb {
                        return Err(bad(CMD, "the working colour profile changed while AI Remove was running"));
                    }
                    if let Some(capture) = preview_capture {
                        let mut candidate_doc = (*preview_source).clone();
                        let (damage, layer) = smart::apply_content(&mut candidate_doc, id, &p, &content, new_layer)?;
                        let mut result = json!({"damage":damage_json(damage)});
                        if let Some(id) = layer {
                            result["layer"] = json!(id.0);
                        }
                        if let (Some(target), Some(metadata)) = (result.as_object_mut(), outcome.metadata.and_then(|value| value.as_object().cloned())) {
                            target.extend(metadata);
                        }
                        if let Some(brushes) = resolved_brushes {
                            result["__removeBrushes"] = Value::Array(brushes);
                        }
                        let params = crate::jobs::canonical_remove_params(CMD, p.clone(), &mut result);
                        let active_layer = layer.or_else(|| s.active().and_then(|st| st.active_layer));
                        return crate::removal_preview::publish(
                            s,
                            capture,
                            crate::removal_preview::Candidate {
                                doc: std::sync::Arc::new(candidate_doc),
                                _content: Box::new(content),
                                active_layer,
                                crop: damage,
                                params,
                                result,
                            },
                        );
                    }
                    let (damage, layer) = s.edit(label, |doc, active| {
                        let result = smart::apply_content(doc, id, &p, &content, new_layer)?;
                        if let Some(id) = result.1 {
                            *active = Some(id);
                        }
                        Ok(result)
                    })?;
                    s.hint_damage(damage);
                    let mut value = json!({"damage":damage_json(damage)});
                    if let Some(id) = layer {
                        value["layer"] = json!(id.0);
                    }
                    value
                }
            };
            if let (Some(target), Some(metadata)) = (result.as_object_mut(), outcome.metadata.and_then(|v| v.as_object().cloned())) {
                target.extend(metadata);
            }
            if let (Some(target), Some(brushes)) = (result.as_object_mut(), resolved_brushes) {
                // Dispatch consumes this internal field before exposing the result to a caller.
                target.insert("__removeBrushes".into(), Value::Array(brushes));
            }
            Ok(result)
        },
    )
}

pub(super) fn brush_snapshot(brush: &BrushSettings) -> Result<Value> {
    brush::snapshot(brush)
}

pub(super) fn context_reason(s: &Session, p: &Value) -> Option<String> {
    smart::context_reason(s, p)
}
pub(super) fn validate_options(s: &Session, p: &Value) -> Result<()> {
    smart::validate(s, p)?;
    // An invalid or unaffordable AI request must not first commit a floating selection.
    // The normal path checks admission again after the floating selection is dropped.
    if s.active().is_some_and(|st| crate::float_cmds::floating(st).is_some()) && matches!(p.get("method").and_then(Value::as_str), Some("ai" | "auto")) {
        let routed = crate::channel_cmds::inject_target(s, CMD, p.clone());
        let method = smart::decide(s, &routed)?;
        if matches!(&method, smart::Method::Ai { .. } | smart::Method::Hybrid) {
            let plan = plan(s, &routed)?;
            let budget_bytes = u64::from(s.prefs().performance.memory_usage_mb).saturating_mul(1024 * 1024);
            smart::admission(&plan, &method, &s.removal_service(), budget_bytes)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod planning_tests {
    use super::*;

    #[test]
    fn review2_window_contains_the_rendered_soft_and_rotated_tip_bounds() {
        for sampled in [false, true] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width":7000,"height":7000,"background":"transparent"})).unwrap();
            let mut brush = BrushSettings { size: if sampled { 2500.0 } else { 2200.0 }, hardness: 0.0, angle: 45.0, ..Default::default() };
            if sampled {
                brush.tip = photocraft_paint::TipShape::Sampled(photocraft_paint::GrayTile::from_fn(8, 8, |_, _| 1.0));
            }
            let snapshot = brush_snapshot(&brush).unwrap();
            let plan = plan(&s, &json!({"points":[[3500,3500]],"brush":snapshot,"method":"classical"})).unwrap();
            let context = photocraft_paint::BrushContext::new(&brush);
            for dab in photocraft_paint::dabs(&plan.strokes[0]) {
                let rendered = context.dab_rect(&dab, false);
                assert_eq!(plan.window.intersect(&rendered), rendered, "planned window clips rendered tip: sampled={sampled}");
            }
        }
    }

    #[test]
    fn review2_large_soft_brush_is_refused_before_rendering_oversized_coverage() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":7000,"height":7000,"background":"transparent"})).unwrap();
        let result = plan(&s, &json!({"points":[[3500,3500]],"size":3000,"hardness":0,"method":"classical"}));
        assert!(result.is_err(), "nominal-radius admission accepted an oversized rendered footprint");
    }

    #[test]
    fn review3_compact_repeated_stroke_is_refused_before_dab_generation() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":256,"height":128,"background":"transparent"})).unwrap();
        let mut brush = BrushSettings { size: 1.0, ..Default::default() };
        brush.scattering.enabled = true;
        brush.scattering.count = 16;
        let snapshot = brush_snapshot(&brush).unwrap();
        let points: Vec<_> = (0..10_000).map(|i| [64 + (i % 2) * 100, 64]).collect();
        let result = plan(&s, &json!({"method":"classical","brush":snapshot,"points":points}));
        assert!(result.is_err(), "a small spatial bound admitted millions of buffered dabs");
    }

    #[test]
    fn review3_one_segment_accounts_for_primary_and_dual_dab_bursts() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":5000,"height":80,"background":"transparent"})).unwrap();
        for dual in [false, true] {
            let mut brush = BrushSettings { size: 1.0, spacing: 0.01, ..Default::default() };
            brush.scattering.enabled = true;
            brush.scattering.count = 16;
            brush.dual_brush.enabled = dual;
            brush.dual_brush.size = 1.0;
            brush.dual_brush.count = 16;
            let snapshot = brush_snapshot(&brush).unwrap();
            let result = plan(&s, &json!({"method":"classical","brush":snapshot,"points":[[100,40],[4100,40]]}));
            assert_eq!(result.is_err(), dual, "one generated segment must include both dab buffers: dual={dual}");
        }
    }

    #[test]
    fn review3_same_crop_admission_accounts_for_generated_work() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":128,"height":128,"background":"transparent"})).unwrap();
        let mut brush = BrushSettings { size: 1.0, ..Default::default() };
        brush.scattering.enabled = true;
        brush.scattering.count = 16;
        let snapshot = brush_snapshot(&brush).unwrap();
        let short = plan(&s, &json!({"method":"classical","brush":snapshot,"points":[[64,64],[65,64]]})).unwrap();
        let points: Vec<_> = (0..1024).map(|i| [64 + i % 2, 64]).collect();
        let repeated = plan(&s, &json!({"method":"classical","brush":snapshot,"points":points})).unwrap();
        assert_eq!(short.window, repeated.window);
        let service = photocraft_ml::RemovalService::new(None);
        let model = service.catalog().iter().find(|m| m.id == "migan-v1").unwrap();
        let method = smart::Method::Ai { id: model.id.into(), sha256: model.sha256.into(), automatic: false };
        let budget = (u64::from(model.memory_mib) + 136) * 1024 * 1024;
        assert!(smart::admission(&short, &method, &service, budget).is_ok(), "small dab buffer should fit");
        assert!(smart::admission(&repeated, &method, &service, budget).is_err(), "same spatial crop hid the larger generated-work allocation");
    }
}
