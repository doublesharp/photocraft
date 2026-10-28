use super::lifecycle::{apply, cancel, close, fill_edit, redo, report, reset, set_sampling, undo};
use super::preview::{rebase, wait_preview};
use super::{CafTool, CafWorkspace, LASSO_OPS, PhotocraftApp, ToolEvent, Value, json};

/// Control channel: `edit.contentAwareFill {"ui": {...}}` while the workspace is open.
pub fn control(app: &mut PhotocraftApp, ui: &Value) -> Result<Value, String> {
    let flag = |k: &str| ui.get(k).and_then(Value::as_bool) == Some(true);
    if flag("cancel") {
        cancel(app);
        return Ok(json!({"cancelled": true}));
    }
    super::ai::validate_controls(app, ui)?;
    let before = app.distort.caf.as_ref().ok_or("Content-Aware Fill is not open")?.opts.clone();
    {
        let w = app.distort.caf.as_mut().ok_or("Content-Aware Fill is not open")?;
        let mut o = w.opts.clone();
        o.apply(ui)?;
        // The sampling option goes through `set_sampling` (it resets the strokes).
        let sampling = std::mem::replace(&mut o.sampling, w.opts.sampling.clone());
        w.opts = o;
        if let Some(t) = ui.get("tool").and_then(Value::as_str) {
            w.tool = CafTool::from_name(t).ok_or(format!("bad tool `{t}` (brush|lasso|hand|zoom)"))?;
        }
        if let Some(n) = ui.get("brushSize").and_then(Value::as_f64).filter(|n| n.is_finite()) {
            w.brush_size = n.clamp(1.0, 5000.0);
        }
        if let Some(m) = ui.get("brushMode").and_then(Value::as_str) {
            w.brush_add = m == "add";
        }
        if let Some(m) = ui.get("lassoMode").and_then(Value::as_str) {
            w.lasso_op = LASSO_OPS.iter().position(|o| *o == m).ok_or(format!("bad lassoMode `{m}`"))?;
        }
        if let Some(n) = ui.get("expandBy").and_then(Value::as_f64).filter(|n| n.is_finite()) {
            w.expand_by = (n as f32).clamp(1.0, 100.0);
        }
        if sampling != w.opts.sampling {
            set_sampling(app, &sampling);
        }
    }
    let after = app.distort.caf.as_ref().map(|w| w.opts.clone());
    if after.as_ref().is_some_and(|a| a.sample_all != before.sample_all || a.method != before.method) {
        if let Some(w) = app.distort.caf.as_mut()
            && w.opts.method == "ai"
            && w.tool == CafTool::Brush
        {
            w.tool = CafTool::Lasso;
        }
        rebase(app);
    } else if after.as_ref().is_some_and(|a| *a != before)
        && let Some(w) = app.distort.caf.as_mut()
    {
        w.stale();
        w.restate_all();
    }
    if ui.get("models").and_then(Value::as_bool) == Some(true) {
        super::ai::open_models(app);
    }
    if let Some(s) = ui.get("stroke") {
        let pts: Vec<[f64; 2]> = serde_json::from_value(s.get("points").cloned().unwrap_or(Value::Null)).map_err(|e| format!("bad stroke points: {e}"))?;
        let w = app.distort.caf.as_mut().ok_or("Content-Aware Fill is not open")?;
        if let Some(n) = s.get("size").and_then(Value::as_f64) {
            w.brush_size = n.clamp(1.0, 5000.0);
        }
        let alt = s.get("alt").and_then(Value::as_bool).unwrap_or(false);
        if let Some(first) = pts.first() {
            w.begin_stroke(*first, alt);
            for p in &pts[1..] {
                w.extend_stroke(*p);
            }
            w.end_stroke();
        }
    }
    if let Some(l) = ui.get("lasso") {
        let pts: Vec<[f64; 2]> = serde_json::from_value(l.get("points").cloned().unwrap_or(Value::Null)).map_err(|e| format!("bad lasso points: {e}"))?;
        let op = match l.get("op").and_then(Value::as_str) {
            Some(o) if LASSO_OPS.contains(&o) => o.to_string(),
            Some(o) => return Err(format!("bad lasso op `{o}`")),
            None => {
                let modifiers = egui::Modifiers {
                    shift: l.get("shift").and_then(Value::as_bool).unwrap_or(false),
                    alt: l.get("alt").and_then(Value::as_bool).unwrap_or(false),
                    ..Default::default()
                };
                let modifiers = crate::workspace_ui::sticky_mods(app, modifiers);
                let operation = app.distort.caf.as_ref().map_or(0, |w| lasso_operation(w, modifiers));
                LASSO_OPS.get(operation).copied().unwrap_or("new").to_string()
            }
        };
        fill_edit(app, json!({"op": op, "points": pts}));
    }
    for k in ["expand", "contract"] {
        if let Some(n) = ui.get(k).and_then(Value::as_f64) {
            fill_edit(app, json!({"op": k, "by": n.clamp(1.0, 100.0)}));
        }
    }
    if let Some(r) = ui.get("reset").and_then(Value::as_str) {
        reset(app, r);
    }
    if flag("undo") {
        undo(app);
    }
    if flag("redo") {
        redo(app);
    }
    if flag("apply") {
        apply(app)?;
    }
    if flag("waitPreview") {
        wait_preview(app)?;
    }
    // OK last, with the settings given alongside. A failed fill keeps the workspace open, as
    // the OK button does.
    if flag("commit") {
        if super::ai::already_applied(app) {
            super::lifecycle::commit(app);
            return Ok(json!({"committed":true,"alreadyApplied":true}));
        }
        let r = apply(app);
        match &r {
            Ok(_) => close(app),
            Err(e) => report(app, e.clone()),
        }
        return r.map(|v| json!({"committed": true, "result": v}));
    }
    Ok(app.distort.caf.as_ref().map(CafWorkspace::describe).unwrap_or(Value::Null))
}

/// The toolbar's lasso mode overridden by the same keys as the canvas selection tools.
/// Capturing it at pointer-down lets the keys be released before the outline closes.
fn lasso_operation(w: &CafWorkspace, mods: egui::Modifiers) -> usize {
    match crate::tool_feedback::selection_mode(crate::state::Tool::Lasso, w.lasso_op.min(3) as u8, mods) {
        "add" => 1,
        "subtract" => 2,
        "intersect" => 3,
        _ => 0,
    }
}

/// Pointer in document coordinates (from the view or the control channel).
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) {
    let mods = crate::workspace_ui::sticky_mods(app, mods);
    let Some(w) = app.distort.caf.as_mut() else { return };
    match (w.tool, ev) {
        (CafTool::Brush, ToolEvent::Down { x, y, .. }) => w.begin_stroke([x, y], mods.alt),
        (CafTool::Brush, ToolEvent::Move { x, y, .. }) => {
            w.hover = Some([x, y]);
            w.extend_stroke([x, y]);
        }
        (CafTool::Brush, ToolEvent::Up { x, y }) => {
            w.extend_stroke([x, y]);
            w.end_stroke();
        }
        (CafTool::Lasso, ToolEvent::Down { x, y, .. }) => {
            // Shift adds, Alt subtracts, both intersect (as Photoshop's selection tools).
            let op = lasso_operation(w, mods);
            w.lasso = Some((op, vec![[x, y]]));
        }
        (CafTool::Lasso, ToolEvent::Move { x, y, .. }) => {
            if let Some((_, pts)) = &mut w.lasso
                && pts.last().is_none_or(|l| l[0] != x || l[1] != y)
            {
                pts.push([x, y]);
            }
        }
        (CafTool::Lasso, ToolEvent::Up { .. }) => {
            if let Some((op, pts)) = w.lasso.take()
                && pts.len() >= 3
            {
                fill_edit(app, json!({"op": LASSO_OPS[op], "points": pts}));
            }
        }
        (CafTool::Zoom, ToolEvent::Down { x, y, .. }) => {
            let f = if mods.alt { 0.5 } else { 2.0 };
            w.zoom = (w.zoom * f).clamp(0.01, 32.0);
            w.center = [x, y];
        }
        _ => {}
    }
}

pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let cmd_shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
    if ctx.input_mut(|i| i.consume_key(cmd_shift, egui::Key::Z)) {
        redo(app);
        return;
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
        undo(app);
        return;
    }
    let Some(w) = app.distort.caf.as_mut() else { return };
    // Ctrl+H hides the overlay (Hide Extras).
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::H)) {
        w.opts.show_overlay = !w.opts.show_overlay;
        w.overlay_dirty = true;
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::OpenBracket)) {
        w.brush_size = (w.brush_size * 0.9).max(1.0).round();
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::CloseBracket)) {
        w.brush_size = (w.brush_size * 1.1).min(5000.0).round().max(w.brush_size + 1.0);
    }
    for (k, t) in [(egui::Key::B, CafTool::Brush), (egui::Key::L, CafTool::Lasso), (egui::Key::H, CafTool::Hand), (egui::Key::Z, CafTool::Zoom)] {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, k)) {
            w.tool = t;
        }
    }
}

pub(super) fn tool_icon(t: CafTool) -> &'static str {
    match t {
        CafTool::Brush => "brush",
        CafTool::Lasso => "lasso",
        CafTool::Hand => "hand",
        CafTool::Zoom => "zoom-in",
    }
}

pub(super) fn tool_tip(t: CafTool) -> String {
    match t {
        CafTool::Brush => format!("{} (B)", tl!("Sampling Brush Tool")),
        CafTool::Lasso => format!("{} (L)", tl!("Lasso Tool")),
        CafTool::Hand => format!("{} (H)", tl!("Hand Tool")),
        CafTool::Zoom => format!("{} (Z)", tl!("Zoom Tool")),
    }
}

#[cfg(test)]
mod selection_modifier_tests {
    use super::*;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width":48,"height":48})).unwrap();
        app.run("select.rect", json!({"x":12,"y":12,"width":16,"height":16})).unwrap();
        super::super::open(&mut app).unwrap();
        app.distort.caf.as_mut().unwrap().tool = CafTool::Lasso;
        app
    }

    #[test]
    fn selection_modifiers_caf_lasso_uses_sticky_intent_and_latches_it_at_press() {
        for (shift, alt, expected) in [(true, false, "add"), (false, true, "subtract"), (true, true, "intersect")] {
            let mut app = app();
            app.ui.shell.sticky_shift = shift;
            app.ui.shell.sticky_alt = alt;
            pointer(&mut app, ToolEvent::Down { x: 8.0, y: 8.0, pressure: 1.0 }, egui::Modifiers::NONE);
            app.ui.shell.sticky_shift = false;
            app.ui.shell.sticky_alt = false;
            for (x, y) in [(30.0, 8.0), (30.0, 30.0), (8.0, 30.0)] {
                pointer(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, egui::Modifiers::NONE);
            }
            pointer(&mut app, ToolEvent::Up { x: 8.0, y: 8.0 }, egui::Modifiers::NONE);
            assert_eq!(app.distort.caf.as_ref().unwrap().fill_edits.last().unwrap()["op"], expected);
        }
    }

    #[test]
    fn selection_modifiers_caf_control_lasso_accepts_the_same_modifier_overrides() {
        let mut app = app();
        control(&mut app, &json!({"lasso":{"points":[[8,8],[30,8],[30,30],[8,30]],"shift":true,"alt":true}})).unwrap();
        assert_eq!(app.distort.caf.as_ref().unwrap().fill_edits.last().unwrap()["op"], "intersect");
    }
}
