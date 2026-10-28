//! Bound generated brush work as well as its spatial footprint before coverage allocation.
use super::{BrushSettings, CMD, EngineError, Result, Stroke, StrokePoint, bad};

/// Both dab buffers together; bounded even for a compact path that repeatedly crosses itself.
const MAX_GENERATED_DABS: u64 = 1 << 18;

fn too_complex() -> EngineError {
    bad(CMD, format!("stroke needs too much brush work (at most {MAX_GENERATED_DABS} estimated dabs); use shorter strokes or less scattering"))
}

/// Worst-case dabs and simultaneously live generator buffers for one untimed Remove stroke.
fn bound(brush: &BrushSettings, len: usize, points: impl IntoIterator<Item = (f64, f64, f64)>) -> Result<(u64, u64)> {
    let samples = u64::try_from(len).ok().filter(|n| *n > 0).and_then(|n| n.checked_add(1)).ok_or_else(too_complex)?;
    // Shape dynamics can shrink spacing to the walker's minimum. Active count dynamics can
    // choose any of the builder's 1..16 dabs, including hostile response-curve values.
    let primary = if brush.scattering.enabled { if brush.scattering.count_jitter.is_active() { 16 } else { brush.scattering.count.clamp(1, 16) } } else { 1 };
    let dual = if brush.dual_brush.enabled { brush.dual_brush.count.clamp(1, 16) } else { 0 };
    let count = u64::from(primary + dual);
    let step_limit = MAX_GENERATED_DABS / count;
    let mut steps = samples.checked_mul(3).and_then(|n| n.checked_add(4)).filter(|n| *n <= step_limit).ok_or_else(too_complex)? as f64;
    let mut previous: Option<(f64, f64)> = None;
    for (x, y, time) in points {
        // Remove's [x,y,pressure] parser constructs timestamp-zero samples. Smoothing then emits
        // at most one position per input plus its final catch-up, with no timed airbrush burst.
        if !x.is_finite() || !y.is_finite() || time != 0.0 {
            return Err(bad(CMD, "Remove needs finite, untimed stroke samples"));
        }
        if let Some((px, py)) = previous {
            // Smoothing moves towards input positions, so its total travel (including catch-up)
            // cannot exceed the input polyline. Four steps per pixel cover the 0.25-pixel lower
            // spacing bound; sample overhead also covers initial dabs and pixel-grid rounding.
            steps += 4.0 * (x - px).hypot(y - py);
            if !steps.is_finite() || steps > step_limit as f64 {
                return Err(too_complex());
            }
        }
        previous = Some((x, y));
    }
    let steps = steps.ceil() as u64; // finite and at most MAX_GENERATED_DABS
    let dabs = steps.checked_mul(count).filter(|n| *n <= MAX_GENERATED_DABS).ok_or_else(too_complex)?;
    // Vec growth can retain spare capacity. Include both primary/dual dabs, the largest walker
    // step buffer, and the smoothing buffer while the renderer rasterizes the generated dabs.
    let dab_bytes = dabs.checked_mul(std::mem::size_of::<photocraft_paint::Dab>() as u64).and_then(|n| n.checked_mul(2));
    let walker_capacity = if dual > 0 { 4 } else { 2 }; // primary steps remain live while dual steps are generated
    let step_bytes = steps.checked_mul(std::mem::size_of::<photocraft_paint::dynamics::StepInput>() as u64).and_then(|n| n.checked_mul(walker_capacity));
    // The retained push scratch and the short finish/catch-up scratch can coexist.
    let point_bytes = samples.checked_mul(2).and_then(|n| n.checked_add(4)).and_then(|n| n.checked_mul(std::mem::size_of::<StrokePoint>() as u64));
    let bytes =
        dab_bytes.and_then(|n| step_bytes.and_then(|s| n.checked_add(s))).and_then(|n| point_bytes.and_then(|p| n.checked_add(p))).ok_or_else(too_complex)?;
    Ok((dabs, bytes))
}

/// Also bound the combined request's work; strokes render sequentially, so only its peak buffer
/// estimate contributes to memory admission.
pub(super) fn estimate(strokes: &[Stroke]) -> Result<u64> {
    let (mut total, mut peak) = (0u64, 0u64);
    for stroke in strokes {
        let (dabs, bytes) = bound(&stroke.brush, stroke.points.len(), stroke.points.iter().map(|p| (p.x, p.y, p.time)))?;
        total = total.checked_add(dabs).filter(|n| *n <= MAX_GENERATED_DABS).ok_or_else(too_complex)?;
        peak = peak.max(bytes);
    }
    Ok(peak)
}

/// Live and queued UI tint must pass the same bound before invoking the renderer.
pub(super) fn check(brush: &BrushSettings, points: &[[f64; 3]]) -> Result<()> {
    bound(brush, points.len(), points.iter().map(|p| (p[0], p[1], 0.0))).map(|_| ())
}
