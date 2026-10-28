//! Validated, bounded snapshots of the effective Remove brush.
use super::{CMD, EngineError, Result, Stroke, Value, bad};
use photocraft_paint::{BrushSettings, GrayTile, Pattern, StoredTile, TipShape};

const MAX_BITMAP_SIDE: u64 = 4096;
const MAX_BITMAP_PIXELS: u64 = 4_194_304;
const MAX_BRUSH_BYTES: usize = 16 * 1024 * 1024;

fn invalid(message: &str) -> EngineError {
    bad(CMD, format!("invalid Remove brush: {message}"))
}

fn charge(remaining: &mut usize, amount: usize) -> Result<()> {
    *remaining = remaining.checked_sub(amount).ok_or_else(|| invalid("brush snapshots exceed 16 MiB"))?;
    Ok(())
}

fn shape() -> Result<Value> {
    let mut schema = serde_json::to_value(BrushSettings::default()).map_err(|_| invalid("cannot build brush schema"))?;
    if let Some(obj) = schema.as_object_mut() {
        // This optional boolean is omitted from a default brush, but is emitted when true.
        obj.insert("square".into(), Value::Bool(false));
    }
    Ok(schema)
}

fn dimensions(value: &serde_json::Map<String, Value>) -> Result<usize> {
    let w = value.get("width").and_then(Value::as_u64).filter(|x| (1..=MAX_BITMAP_SIDE).contains(x)).ok_or_else(|| invalid("invalid bitmap width"))?;
    let h = value.get("height").and_then(Value::as_u64).filter(|x| (1..=MAX_BITMAP_SIDE).contains(x)).ok_or_else(|| invalid("invalid bitmap height"))?;
    let n = w.checked_mul(h).filter(|x| *x <= MAX_BITMAP_PIXELS).ok_or_else(|| invalid("bitmap is too large"))?;
    usize::try_from(n).map_err(|_| invalid("bitmap is too large"))
}

fn object<'a>(value: &'a Value, keys: &[&str]) -> Result<&'a serde_json::Map<String, Value>> {
    let map = value.as_object().ok_or_else(|| invalid("expected object"))?;
    if map.keys().any(|key| !keys.contains(&key.as_str())) {
        return Err(invalid("unknown bitmap or pattern field"));
    }
    Ok(map)
}

fn tile(value: &Value, remaining: &mut usize) -> Result<()> {
    let map = object(value, &["width", "height", "data"])?;
    let n = dimensions(map)?;
    match map.get("data") {
        Some(Value::String(data)) => {
            let expected = n.checked_mul(2).and_then(|bytes| bytes.checked_add(2)).map(|bytes| bytes / 3 * 4).ok_or_else(|| invalid("bitmap size overflow"))?;
            if data.len() != expected {
                return Err(invalid("bitmap data length does not match dimensions"));
            }
            charge(remaining, data.len())?;
        }
        Some(Value::Array(data)) => {
            if data.len() != n || data.iter().any(|v| v.as_f64().is_none_or(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))) {
                return Err(invalid("bitmap samples do not match dimensions or range"));
            }
            charge(remaining, n.checked_mul(8).ok_or_else(|| invalid("bitmap size overflow"))?)?;
        }
        _ => return Err(invalid("bitmap data must be base64 or a sample array")),
    }
    Ok(())
}

fn stored(value: &Value, remaining: &mut usize) -> Result<()> {
    let map = object(value, &["key", "width", "height", "preview"])?;
    dimensions(map)?;
    let key = map.get("key").and_then(Value::as_str).filter(|key| key.len() <= 128).ok_or_else(|| invalid("invalid stored bitmap key"))?;
    charge(remaining, key.len())?;
    tile(map.get("preview").ok_or_else(|| invalid("missing stored bitmap preview"))?, remaining)
}

fn variant(value: &Value) -> Result<(&str, &Value)> {
    let obj = value.as_object().filter(|o| o.len() == 1).ok_or_else(|| invalid("invalid brush variant"))?;
    obj.iter().next().map(|(key, value)| (key.as_str(), value)).ok_or_else(|| invalid("invalid brush variant"))
}

fn tip(value: &Value, remaining: &mut usize) -> Result<()> {
    if value.as_str() == Some("round") {
        return Ok(());
    }
    match variant(value)? {
        ("sampled", value) => tile(value, remaining),
        ("stored", value) => stored(value, remaining),
        _ => Err(invalid("unknown brush tip")),
    }
}

fn pattern(value: &Value, remaining: &mut usize) -> Result<()> {
    match variant(value)? {
        ("tile", value) => tile(value, remaining),
        ("stored", value) => stored(value, remaining),
        ("procedural", value) => {
            let obj = object(value, &["style", "size", "seed"])?;
            let style = obj.get("style").and_then(Value::as_str).filter(|v| matches!(*v, "noise" | "canvas" | "paper" | "dots"));
            let size = obj.get("size").and_then(Value::as_u64).filter(|v| (1..=1024).contains(v));
            // The renderer multiplies this seed by a 40-bit salt in u64.
            let seed = obj.get("seed").and_then(Value::as_u64).filter(|v| *v <= 0x00ff_ffff);
            if style.is_none() || size.is_none() || seed.is_none() {
                return Err(invalid("invalid procedural pattern"));
            }
            charge(remaining, 64)
        }
        _ => Err(invalid("unknown texture pattern")),
    }
}

fn validate(value: &Value, schema: &Value, path: &str, remaining: &mut usize) -> Result<()> {
    match path {
        "tip" | "dualBrush.tip" => return tip(value, remaining),
        "texture.pattern" => return pattern(value, remaining),
        _ => {}
    }
    match (value, schema) {
        (Value::Object(map), Value::Object(expected)) => {
            charge(remaining, map.len().saturating_mul(16))?;
            for (key, child) in map {
                let target = expected.get(key).ok_or_else(|| invalid("unknown brush field"))?;
                charge(remaining, key.len())?;
                let next = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                validate(child, target, &next, remaining)?;
            }
            Ok(())
        }
        (Value::Array(array), Value::Array(expected)) => {
            if array.len() > 64 {
                return Err(invalid("brush array is too large"));
            }
            charge(remaining, array.len().saturating_mul(8))?;
            let sample = expected.first().ok_or_else(|| invalid("unexpected brush array"))?;
            for child in array {
                validate(child, sample, path, remaining)?;
            }
            Ok(())
        }
        (Value::String(s), Value::String(_)) if s.len() <= 256 => charge(remaining, s.len()),
        (Value::Number(n), Value::Number(_)) if n.as_f64().is_some_and(|v| v.is_finite() && v.abs() <= f64::from(f32::MAX)) => charge(remaining, 8),
        (Value::Bool(_), Value::Bool(_)) | (Value::Null, Value::Null) => charge(remaining, 1),
        _ => Err(invalid("brush field has the wrong type or range")),
    }
}

/// Check every supplied brush, including tip buffers, before cloning or deserializing it.
pub(super) fn validate_request(p: &Value) -> Result<()> {
    let has_brush = p.get("brush").is_some()
        || p.get("strokes").and_then(Value::as_array).is_some_and(|strokes| strokes.iter().any(|stroke| stroke.get("brush").is_some()));
    if !has_brush {
        return Ok(());
    }
    let schema = shape()?;
    let mut remaining = MAX_BRUSH_BYTES;
    if let Some(brush) = p.get("brush") {
        validate(brush, &schema, "", &mut remaining)?;
    }
    if let Some(strokes) = p.get("strokes").and_then(Value::as_array) {
        for stroke in strokes {
            if let Some(brush) = stroke.get("brush") {
                validate(brush, &schema, "", &mut remaining)?;
            }
        }
    }
    Ok(())
}

pub(super) fn decode(value: &Value) -> Result<BrushSettings> {
    let brush: BrushSettings = serde_json::from_value(value.clone()).map_err(|e| bad(CMD, format!("invalid Remove brush: {e}")))?;
    crate::brush_cmds::validate_brush(&brush, CMD)?;
    Ok(brush)
}

fn native_tile(tile: &GrayTile, total: &mut usize) -> Result<()> {
    let pixels = u64::from(tile.width).checked_mul(u64::from(tile.height)).filter(|v| *v <= MAX_BITMAP_PIXELS).ok_or_else(|| invalid("bitmap is too large"))?;
    if tile.width == 0 || tile.height == 0 || u64::from(tile.width) > MAX_BITMAP_SIDE || u64::from(tile.height) > MAX_BITMAP_SIDE || !tile.is_valid() {
        return Err(invalid("invalid bitmap dimensions"));
    }
    let encoded = usize::try_from(pixels)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(2))
        .map(|n| n / 3 * 4)
        .ok_or_else(|| invalid("bitmap size overflow"))?;
    *total = total.checked_add(encoded).ok_or_else(|| invalid("brush snapshot size overflow"))?;
    if *total > MAX_BRUSH_BYTES {
        return Err(invalid("brush snapshots exceed 16 MiB"));
    }
    Ok(())
}

fn native_stored(tile: &StoredTile, total: &mut usize) -> Result<()> {
    let w = u64::from(tile.width);
    let h = u64::from(tile.height);
    if w == 0 || h == 0 || w > MAX_BITMAP_SIDE || h > MAX_BITMAP_SIDE || w.saturating_mul(h) > MAX_BITMAP_PIXELS || tile.key.len() > 128 {
        return Err(invalid("invalid stored bitmap"));
    }
    native_tile(tile.bitmap(), total)
}

fn native_tip(tip: &TipShape, total: &mut usize) -> Result<()> {
    match tip {
        TipShape::Round => Ok(()),
        TipShape::Sampled(tile) => native_tile(tile, total),
        TipShape::Stored(tile) => native_stored(tile, total),
    }
}

/// Capture one effective brush after validating its bitmap sizes before serialization.
pub(super) fn snapshot(brush: &BrushSettings) -> Result<Value> {
    let stroke = Stroke { brush: brush.clone(), points: Vec::new() };
    snapshots(std::slice::from_ref(&stroke))?.pop().ok_or_else(|| invalid("missing brush snapshot"))
}

/// Serialize the actual stroke brushes for action recording without depending on later tool state.
pub(super) fn snapshots(strokes: &[Stroke]) -> Result<Vec<Value>> {
    let mut total = 0usize;
    for stroke in strokes {
        total = total.checked_add(4096).ok_or_else(|| invalid("brush snapshot size overflow"))?;
        native_tip(&stroke.brush.tip, &mut total)?;
        native_tip(&stroke.brush.dual_brush.tip, &mut total)?;
        match &stroke.brush.texture.pattern {
            Pattern::Tile(tile) => native_tile(tile, &mut total)?,
            Pattern::Stored(tile) => native_stored(tile, &mut total)?,
            Pattern::Procedural { .. } => {}
        }
        if total > MAX_BRUSH_BYTES {
            return Err(invalid("brush snapshots exceed 16 MiB"));
        }
    }
    let schema = shape()?;
    let mut remaining = MAX_BRUSH_BYTES;
    let mut values = Vec::with_capacity(strokes.len());
    for stroke in strokes {
        let value = serde_json::to_value(&stroke.brush).map_err(|e| bad(CMD, format!("cannot serialize Remove brush: {e}")))?;
        validate(&value, &schema, "", &mut remaining)?;
        values.push(value);
    }
    Ok(values)
}
