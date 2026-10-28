use super::*;
use crate::canvas::{ToolEvent, tool_event};

static REMOVE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
    app.run("tools.setBrush", json!({"brush": {"size": 16, "hardness": 1.0}})).unwrap();
    app
}

fn drag(app: &mut PhotocraftApp, tool: Tool) {
    app.ui.tool = tool;
    let m = egui::Modifiers::NONE;
    tool_event(app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
    assert!(!app.ui.status_error, "{tool:?}: {}", app.ui.status);
}

fn active(app: &PhotocraftApp) -> &photocraft_doc::Layer {
    let st = app.session.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).unwrap()
}

fn stripes(app: &mut PhotocraftApp, step: usize, target: &str) {
    for x in (0..100).step_by(step) {
        app.run("paint.pencil", json!({"points": [[x, 0], [x, 60]], "size": 2, "color": "#606060", "target": target})).unwrap();
    }
}

#[test]
fn clone_stamp_in_pattern_preview_wraps_a_dab_across_the_right_edge() {
    for pattern_preview in [false, true] {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[10, 10]], "size": 24, "color": "#ff0000"})).unwrap();
        app.ui.tool = Tool::CloneStamp;
        app.ui.clone_source = Some([10.0, 10.0]);
        app.ui.view.pattern_preview = pattern_preview;
        let modifiers = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 98.0, y: 30.0, pressure: 1.0 }, modifiers);
        tool_event(&mut app, ToolEvent::Up { x: 98.0, y: 30.0 }, modifiers);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let surface = active(&app).surface().unwrap();
        let center = surface.rgba(98, 30);
        assert!(center[0] > 0.9 && center[1] < 0.1, "clone center: {center:?}");
        let opposite = surface.rgba(1, 30);
        if pattern_preview {
            assert!(opposite[0] > 0.9 && opposite[1] < 0.1, "Pattern Preview must wrap the clone footprint: {opposite:?}");
        } else {
            assert!(opposite[0] > 0.9 && opposite[1] > 0.9, "ordinary cloning must not wrap: {opposite:?}");
        }
    }
}

#[test]
fn alt_flips_dodge_burn_and_blur_sharpen() {
    for (tool, flipped) in [(Tool::Dodge, Tool::Burn), (Tool::Burn, Tool::Dodge), (Tool::Blur, Tool::Sharpen), (Tool::Sharpen, Tool::Blur)] {
        assert_eq!(alt_flipped(tool, true), flipped);
        assert_eq!(alt_flipped(tool, false), tool);
    }
    for tool in [Tool::Sponge, Tool::Smudge, Tool::Brush] {
        assert_eq!(alt_flipped(tool, true), tool, "{tool:?} has no ⌥ counterpart");
    }
}

#[test]
fn alt_stroke_runs_the_opposite_tool() {
    // ⌥ held as the stroke starts: Dodge burns, Burn dodges, Blur sharpens, Sharpen blurs.
    for (tool, cmd) in [(Tool::Dodge, "paint.burn"), (Tool::Burn, "paint.dodge"), (Tool::Blur, "paint.sharpen"), (Tool::Sharpen, "paint.blur")] {
        let mut app = app();
        app.ui.tool = tool;
        let m = egui::Modifiers::ALT;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some(cmd), "{tool:?} with ⌥");
        assert_eq!(app.ui.tool, tool, "the selected tool stays {tool:?}");
    }
}

#[test]
fn clone_stamp_brush_opacity_and_flow_reach_the_painted_pixels() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[10, 10]], "size": 16, "color": "#ff0000"})).unwrap();
    app.run("layer.new.layer", json!({})).unwrap();
    app.ui.tool = Tool::CloneStamp;
    app.ui.clone_source = Some([10.0, 10.0]);
    app.ui.tool_options.clone_sample = "all".into();
    app.run("tools.setBrush", json!({"brush": {"opacity": 0.5, "flow": 0.5}})).unwrap();
    assert!(finish_stroke(&mut app, Tool::CloneStamp, &[[50.0, 30.0, 1.0]], egui::Modifiers::NONE));
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let pixel = active(&app).surface().unwrap().rgba(50, 30);
    assert!(pixel[0] > 0.9 && pixel[3] > 0.0 && pixel[3] < 0.5, "partial-opacity, partial-flow red stamp: {pixel:?}");
    let (id, params) = app.session.journal.last().unwrap();
    assert_eq!(id, "paint.cloneStamp");
    assert_eq!(params["opacity"], 50.0);
    assert_eq!(params["flow"], 50.0);
}

/// A Healing Brush whose brush holds a mode outside its menu (Overlay, restored with the tool
/// active, so no tool switch sees it) heals in Normal; the Clone Stamp keeps Overlay.
#[test]
fn healing_brush_heals_in_normal_with_a_mode_outside_its_menu() {
    for (tool, want) in [(Tool::Healing, "Normal"), (Tool::CloneStamp, "Overlay")] {
        let mut app = app();
        app.ui.tool = tool;
        crate::paint_mouse::sync_tool_brush(&mut app);
        app.session.tools.brush.mode = photocraft_color::BlendMode::Overlay;
        app.ui.clone_source = Some([10.0, 10.0]);
        drag(&mut app, tool);
        let (_, params) = app.session.journal.last().unwrap();
        assert_eq!(params["mode"], want, "{tool:?}");
    }
}

/// The options bar's Mode (each tool's own brush's) reaches the stamps', the History Brush's
/// and the Healing Brush's strokes, live and committed.
#[test]
fn retouch_mode_reaches_the_command() {
    for (tool, cmd, mode) in [
        (Tool::CloneStamp, "paint.cloneStamp", "Multiply"),
        (Tool::Healing, "paint.healingBrush", "Luminosity"),
        (Tool::HistoryBrush, "paint.historyBrush", "Screen"),
        (Tool::PatternStamp, "paint.patternStamp", "Overlay"),
    ] {
        let mut app = app();
        app.ui.tool = tool;
        crate::paint_mouse::sync_tool_brush(&mut app);
        app.run("tools.setBrush", json!({"brush": {"mode": mode}})).unwrap();
        app.ui.clone_source = Some([10.0, 10.0]);
        drag(&mut app, tool);
        let (id, params) = app.session.journal.last().unwrap();
        assert_eq!((id.as_str(), params["mode"].as_str()), (cmd, Some(mode)), "{tool:?}");
    }
}

/// The Paint Bucket's Mode reaches the fill: Multiply darkens, Clear erases.
#[test]
fn paint_bucket_mode_reaches_the_fill() {
    let mut app = app();
    app.run("layer.new.layer", json!({})).unwrap();
    app.ui.tool = Tool::PaintBucket;
    app.session.tools.foreground = [0.5, 0.5, 0.5, 1.0];
    let click = |app: &mut PhotocraftApp| {
        let m = egui::Modifiers::NONE;
        tool_event(app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: 20.0, y: 20.0 }, m);
    };
    click(&mut app);
    app.session.tools.foreground = [1.0, 0.0, 0.0, 1.0];
    app.ui.tool_options.bucket_blend_mode = photocraft_color::BlendMode::Multiply;
    click(&mut app);
    let px = active(&app).surface().unwrap().rgba(20, 20);
    assert!((px[0] - 0.5).abs() < 0.02 && px[1] < 0.02, "multiplied: {px:?}");
    app.ui.tool_options.bucket_clear = true;
    click(&mut app);
    assert_eq!(app.session.journal.last().unwrap().1["mode"], "clear");
    assert!(active(&app).surface().unwrap().rgba(20, 20)[3] < 0.01, "cleared");
}

#[test]
fn clone_without_a_source_names_the_platform_modifier() {
    // Option-click on the Mac, Alt-click elsewhere (#251).
    for tool in [Tool::CloneStamp, Tool::Healing] {
        let mut app = app();
        app.ui.tool = tool;
        assert!(finish_stroke(&mut app, tool, &[[10.0, 30.0, 1.0], [50.0, 30.0, 1.0]], egui::Modifiers::NONE));
        assert!(app.ui.status_error, "{tool:?}");
        let want = if cfg!(target_os = "macos") { "Option-click" } else { "Alt-click" };
        assert!(app.ui.status.starts_with(want), "{tool:?}: {}", app.ui.status);
    }
}

#[test]
fn retouch_strokes_paint_the_targeted_mask() {
    // #207: with the mask targeted, retouching changes the mask, never the pixels.
    for tool in [Tool::Blur, Tool::Sharpen, Tool::Smudge, Tool::Dodge, Tool::Burn] {
        let mut app = app();
        app.run("layer.new.layer", json!({})).unwrap();
        stripes(&mut app, 4, "pixels");
        app.run("layer.layerMask.revealAll", json!({})).unwrap();
        stripes(&mut app, 6, "mask");
        app.ui.mask_target = true;
        app.ui.tool_options.protect_detail = false;
        let b = app.session.active().unwrap().doc.bounds();
        let px0 = active(&app).surface().unwrap().read_region(b);
        let mask0 = active(&app).mask.as_ref().unwrap().surface.read_region(b);
        drag(&mut app, tool);
        assert_eq!(active(&app).surface().unwrap().read_region(b), px0, "{tool:?}: pixels untouched");
        assert_ne!(active(&app).mask.as_ref().unwrap().surface.read_region(b), mask0, "{tool:?}: mask changed");
    }
}

#[test]
fn sample_all_layers_reaches_the_command() {
    // #207, #731: the options-bar checkbox reaches the command.
    for tool in [Tool::SpotHealing, Tool::Remove, Tool::Blur, Tool::Sharpen, Tool::Smudge] {
        for all in [false, true] {
            let mut app = app();
            stripes(&mut app, 4, "pixels");
            app.run("layer.new.layer", json!({})).unwrap();
            *app.ui.tool_options.sample_all.slot(tool).unwrap() = all;
            drag(&mut app, tool);
            let a = active(&app).surface().unwrap().rgba(30, 30)[3];
            assert_eq!(a > 0.0, all, "{tool:?} sampleAllLayers={all}: alpha {a}");
        }
    }
}

#[test]
fn sample_all_layers_is_each_tools_own() {
    // #5524: Sample All Layers turned on for another tool (the Magic Wand, say) left Blur and
    // Smudge copying the composite onto an annotation layer, freezing a sharp patch above the
    // picture. Photoshop keeps the option per tool, so those strokes leave the empty layer empty.
    for tool in [Tool::Blur, Tool::Sharpen, Tool::Smudge] {
        let mut app = app();
        stripes(&mut app, 4, "pixels");
        app.run("layer.new.layer", json!({})).unwrap();
        let mut others = crate::state::SampleAllLayers::default();
        for t in [
            Tool::MagicWand,
            Tool::QuickSelection,
            Tool::ObjectSelection,
            Tool::PaintBucket,
            Tool::MagicEraser,
            Tool::SpotHealing,
            Tool::Patch,
            Tool::ContentAwareMove,
            Tool::Remove,
            Tool::Blur,
            Tool::Sharpen,
            Tool::Smudge,
        ] {
            if let Some(v) = others.slot(t).filter(|_| t != tool) {
                *v = true;
            }
        }
        app.ui.tool_options.sample_all = others;
        assert!(!app.ui.tool_options.sample_all.get(tool));
        drag(&mut app, tool);
        let a = active(&app).surface().unwrap().rgba(30, 30)[3];
        assert_eq!(a, 0.0, "{tool:?} sampled all layers through another tool's option");
    }
}

#[test]
fn object_selection_click_falls_back_to_quick_selection() {
    // A click cannot be a rectangle: it selects at the cursor instead of looking dead.
    let mut app = app();
    app.ui.tool = Tool::ObjectSelection;
    app.ui.tool_options.object_finder = false;
    let m = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.quick"));
}

#[test]
fn object_selection_rect_runs_object_selection_and_reports() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[20, 10], [20, 50]], "size": 8, "color": "#ff0000"})).unwrap();
    app.ui.tool = Tool::ObjectSelection;
    let m = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 60.0, y: 55.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 55.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.object"));
}

#[test]
fn empty_smart_result_is_reported_without_error() {
    // An empty smart selection (Select Subject missing on a busy photo) is an info, not
    // an error and not silence: the button must visibly answer.
    let mut app = app();
    report_smart_result(&mut app, Ok(json!({"selected": false, "changed": false})), "nothing there");
    assert_eq!(app.ui.status.as_str(), "nothing there");
    assert!(!app.ui.status_error);
    app.ui.status.clear();
    report_smart_result(&mut app, Ok(json!({"selected": true, "changed": true})), "nothing there");
    assert!(app.ui.status.is_empty());
}

#[test]
fn mixer_brush_is_selectable_and_paints_only_inside_the_selection_via_control() {
    use crate::control::{ControlRequest, Outcome, handle};

    let mut app = app();
    app.run("paint.pencil", json!({"points": [[50, 30]], "size": 200, "color": "#204080"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#f02010"})).unwrap();
    app.run(
        "tools.setBrush",
        json!({
            "pressureSize": false,
            "size": 12,
            "mixer": {"wet": 0.0, "load": 1.0, "mix": 0.0, "flow": 1.0}
        }),
    )
    .unwrap();
    app.run("select.rect", json!({"x": 35, "y": 20, "width": 30, "height": 20})).unwrap();
    let before = {
        let surface = active(&app).surface().unwrap();
        [surface.rgba(20, 30), surface.rgba(50, 30), surface.rgba(80, 30)]
    };

    let ctx = egui::Context::default();
    let (req, _rx) = ControlRequest::new(
        "ui.pointer",
        json!({
            "tool": "mixerBrush",
            "events": [
                {"kind": "down", "x": 8, "y": 30},
                {"kind": "move", "x": 92, "y": 30},
                {"kind": "up", "x": 92, "y": 30}
            ]
        }),
    );
    assert!(matches!(handle(&mut app, &ctx, &req), Outcome::Done(_)));
    assert_eq!(app.ui.tool, Tool::MixerBrush);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.mixerBrush"));

    let surface = active(&app).surface().unwrap();
    assert_ne!(surface.rgba(50, 30), before[1], "the selected pixels are mixed");
    assert_eq!(surface.rgba(20, 30), before[0], "outside the selection is unchanged");
    assert_eq!(surface.rgba(80, 30), before[2], "the far side outside the selection is unchanged");
}

#[test]
fn patch_tool_lassoes_then_drags_the_patch() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[68, 30], [74, 30]], "size": 6, "color": "#ff0000"})).unwrap();
    let red = |app: &PhotocraftApp| active(app).surface().unwrap().rgba(71, 30)[1] < 0.5;
    assert!(red(&app));
    app.ui.tool = Tool::Patch;
    let m = egui::Modifiers::NONE;
    // Outside any selection the drag is a lasso.
    tool_event(&mut app, ToolEvent::Down { x: 60.0, y: 20.0, pressure: 1.0 }, m);
    for [x, y] in [[84.0, 20.0], [84.0, 40.0], [60.0, 40.0]] {
        tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
    }
    tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 40.0 }, m);
    assert!(app.session.active().unwrap().doc.selection.is_some(), "lasso made a selection");
    assert!(red(&app), "the lasso alone changes no pixels");
    // ⇧-drag inside the selection adds to it rather than patching.
    assert!(!patch_drags_selection(&app, [70.0, 30.0], egui::Modifiers::SHIFT));
    // Dragging inside it patches from where it is dropped; the offset is limited to the canvas.
    assert!(patch_drags_selection(&app, [70.0, 30.0], m));
    assert_eq!(patch_offset(&mut app, [70.0, 30.0], [-200.0, 30.0]), [-60, 0]);
    tool_event(&mut app, ToolEvent::Down { x: 70.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let px = active(&app).surface().unwrap().rgba(71, 30);
    assert!(px[0] > 0.95 && px[1] > 0.95 && px[2] > 0.95, "blemish patched with the white background: {px:?}");
}

#[test]
fn patch_tool_content_aware_fills_from_where_it_is_dropped() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[68, 30], [74, 30]], "size": 6, "color": "#ff0000"})).unwrap();
    app.run("select.rect", json!({"x": 60, "y": 20, "width": 24, "height": 20})).unwrap();
    app.ui.tool = Tool::Patch;
    app.ui.tool_options.patch_content_aware = true;
    app.ui.tool_options.patch_structure = 5.0;
    app.ui.tool_options.patch_color = 3.0;
    let m = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 70.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let (id, p) = app.session.journal.last().cloned().unwrap();
    assert_eq!(id, "paint.patch");
    assert_eq!((p["contentAware"].as_bool(), p["structure"].as_f64(), p["color"].as_f64()), (Some(true), Some(5.0), Some(3.0)));
    let px = active(&app).surface().unwrap().rgba(71, 30);
    assert!(px[0] > 0.95 && px[1] > 0.95 && px[2] > 0.95, "blemish filled from the white background: {px:?}");
}

#[test]
fn content_aware_move_tool_lassoes_then_moves_or_extends() {
    for (mode, keeps) in [("move", false), ("extend", true)] {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[20, 30], [24, 30]], "size": 6, "color": "#0000ff"})).unwrap();
        let blue = |app: &PhotocraftApp, x: i32| active(app).surface().unwrap().rgba(x, 30)[0] < 0.5;
        app.ui.tool = Tool::ContentAwareMove;
        app.ui.tool_options.cam_mode = mode.into();
        app.ui.tool_options.cam_structure = 7.0;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 12.0, y: 20.0, pressure: 1.0 }, m);
        for [x, y] in [[32.0, 20.0], [32.0, 40.0], [12.0, 40.0]] {
            tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
        }
        tool_event(&mut app, ToolEvent::Up { x: 12.0, y: 40.0 }, m);
        assert!(app.session.active().unwrap().doc.selection.is_some(), "lasso made a selection");
        assert!(blue(&app, 22), "the lasso alone changes no pixels");
        tool_event(&mut app, ToolEvent::Down { x: 22.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 72.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{mode}: {}", app.ui.status);
        assert!(blue(&app, 72), "{mode}: the content lands where it was dropped");
        assert_eq!(blue(&app, 22), keeps, "{mode}: the original place");
        assert_eq!(app.session.journal.last().map(|(id, p)| (id.as_str(), p["mode"].as_str())), Some(("paint.contentAwareMove", Some(mode))));
        let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap();
        assert!(sel.sample_channel(72, 30, 0) > 0.0 && sel.sample_channel(22, 30, 0) == 0.0, "{mode}: the selection follows");
    }
}

#[test]
fn content_aware_move_transform_on_drop_turns_the_patch_on_enter() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[16, 30], [28, 30]], "size": 4, "color": "#0000ff"})).unwrap();
    let blue = |app: &PhotocraftApp, x: i32, y: i32| active(app).surface().unwrap().rgba(x, y)[0] < 0.5;
    app.ui.tool = Tool::ContentAwareMove;
    app.ui.tool_options.cam_transform_on_drop = true;
    app.ui.tool_options.cam_structure = 7.0;
    let m = egui::Modifiers::NONE;
    let drag = |app: &mut PhotocraftApp, path: &[[f64; 2]]| {
        let ([x, y], rest) = (path[0], &path[1..]);
        tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, m);
        for &[x, y] in rest {
            tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
        }
        let [x, y] = path[path.len() - 1];
        tool_event(app, ToolEvent::Up { x, y }, m);
    };
    drag(&mut app, &[[6.0, 14.0], [38.0, 14.0], [38.0, 46.0], [6.0, 46.0]]);
    let steps = app.session.active().unwrap().history.past_len();
    // The drop puts a box on the patch; nothing is computed yet.
    drag(&mut app, &[[22.0, 30.0], [50.0, 30.0], [72.0, 30.0]]);
    assert!(crate::cam_transform::active(&app), "{}", app.ui.status);
    assert!(blue(&app, 22, 30) && !blue(&app, 72, 30), "the document waits for the commit");
    assert_eq!(app.session.active().unwrap().history.past_len(), steps);
    // Dragging outside the box turns it 90° about its centre (72, 30).
    drag(&mut app, &[[98.0, 55.0], [60.0, 58.0], [47.0, 56.0]]);
    let place = crate::cam_transform::placement(app.ui.cam_transform.as_ref().unwrap()).unwrap();
    assert_eq!((place.offset, place.scale), ([50, 0], [100.0, 100.0]));
    assert!((place.rotate - 90.0).abs() < 1e-3, "{place:?}");
    crate::cam_transform::commit(&mut app);
    assert!(!crate::cam_transform::active(&app) && !app.ui.status_error, "{}", app.ui.status);
    let (id, p) = app.session.journal.last().unwrap();
    assert_eq!((id.as_str(), &p["offset"]), ("paint.contentAwareMove", &json!([50, 0])));
    assert!((p["rotate"].as_f64().unwrap() - 90.0).abs() < 1e-3, "{p}");
    // The horizontal line lands upright, in one History step.
    assert!(blue(&app, 72, 26) && blue(&app, 72, 34), "the turned line");
    assert!(!blue(&app, 66, 30) && !blue(&app, 78, 30), "no unturned line");
    assert!(!blue(&app, 22, 30), "the old place is filled");
    assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1);
}

#[test]
fn red_eye_click_corrects_a_pupil_and_errors_on_a_miss() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[50, 30]], "size": 18, "hardness": 100, "color": "#f21818"})).unwrap();
    let before = active(&app).surface().unwrap().rgba(50, 30);
    assert!(before[0] > before[1] + 0.4, "{before:?}");
    app.ui.tool = Tool::RedEye;
    // Pupil Size 15 → search radius 22 px, so a click at (8,8) cannot reach the blob at (50,30).
    app.ui.tool_options.red_eye_pupil_size = 15.0;
    tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, egui::Modifiers::NONE);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.redEye"));
    let after = active(&app).surface().unwrap().rgba(50, 30);
    assert!(after[0] < before[0] - 0.15, "before {before:?} after {after:?}");
    tool_event(&mut app, ToolEvent::Down { x: 8.0, y: 8.0, pressure: 1.0 }, egui::Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 8.0, y: 8.0 }, egui::Modifiers::NONE);
    assert!(app.ui.status_error);
    assert!(app.ui.status.contains("no red-eye pixels"), "{}", app.ui.status);
}

#[test]
fn red_eye_options_bar_shows_pupil_size_and_darken() {
    use crate::theme::ThemeKind;
    use egui::vec2;
    use egui_kittest::kittest::Queryable;
    let mut app = app();
    app.ui.tool = Tool::RedEye;
    let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 60.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::panels::options_bar(app, ui);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, ThemeKind::Studio);
    h.run_steps(4);
    h.get_by_label("Pupil Size");
    h.get_by_label("Darken Amount");
    assert_eq!(h.state().ui.tool_options.red_eye_pupil_size, 50.0);
    assert_eq!(h.state().ui.tool_options.red_eye_darken, 50.0);
}

#[test]
fn patch_tool_says_to_drag_the_selection_after_the_lasso_and_after_a_click() {
    let hint = tl!("Now drag the selection onto another area");
    let m = egui::Modifiers::NONE;
    let mut app = app();
    app.ui.tool = Tool::Patch;
    app.ui.status.clear();
    // No selection yet: the drag is a lasso, which outlines the patch and says what comes next.
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 10.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 10.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 30.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 30.0 }, m);
    assert!(app.session.active().is_some_and(|st| st.doc.selection.is_some()), "the lasso made a selection");
    assert_eq!((app.ui.status.as_str(), app.ui.status_error), (hint, false));
    // A press and release inside the selection moves nothing: it says so instead of staying silent.
    app.ui.status.clear();
    tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 20.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 20.0 }, m);
    assert_eq!((app.ui.status.as_str(), app.ui.status_error), (hint, false));
    assert!(app.session.active().is_some_and(|st| st.doc.selection.is_some()), "the selection stays");
}

#[test]
fn remove_tool_stroke_removes_what_it_covers() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
    app.run("paint.pencil", json!({"points": [[30, 20], [30, 40]], "size": 4, "color": "#000000"})).unwrap();
    let bg = active(&app).surface().unwrap().rgba(70, 30);
    app.ui.tool = Tool::Remove;
    let m = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 18.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 42.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 42.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.remove"));
    let px = active(&app).surface().unwrap().rgba(30, 30);
    assert!(px.iter().zip(bg).all(|(a, b)| (a - b).abs() < 0.02), "{px:?} vs {bg:?}");
}

/// A waiting Remove stroke keeps the brush it was drawn with (#8923 review): changing the brush
/// before ↩ neither changes its tint nor what its removal takes.
#[test]
fn waiting_remove_strokes_keep_their_own_brush() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    let m = egui::Modifiers::NONE;
    let stroke = |app: &mut PhotocraftApp, x: f64| {
        tool_event(app, ToolEvent::Down { x, y: 20.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x, y: 40.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x, y: 40.0 }, m);
    };
    app.run("tools.setBrush", json!({"brush": {"size": 20, "hardness": 0.0}})).unwrap();
    stroke(&mut app, 25.0);
    let tint = |app: &PhotocraftApp| app.remove_trail.as_ref().map(|t| (0..100).filter(|&x| t.coverage(x, 30) > 0).count());
    let first = tint(&app);
    app.run("tools.setBrush", json!({"brush": {"size": 4, "hardness": 1.0}})).unwrap();
    stroke(&mut app, 75.0);
    assert!(tint(&app).unwrap() > first.unwrap(), "the second stroke adds its tint");
    assert!((0..100).filter(|&x| app.remove_trail.as_ref().unwrap().coverage(x, 30) > 0).any(|x| x < 25 - 10), "the first keeps its soft 20 px reach");
    assert!(remove_pending_commit(&mut app));
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let (_, p) = app.session.journal.last().cloned().expect("a journaled command");
    let sizes: Vec<_> = p["strokes"].as_array().unwrap().iter().map(|e| (e["brush"]["size"].as_f64(), e["brush"]["hardness"].as_f64())).collect();
    assert_eq!(sizes, [(Some(20.0), Some(0.0)), (Some(4.0), Some(1.0))], "each stroke is removed with its own complete brush");
}

#[test]
fn remove_after_each_stroke_off_collects_strokes_until_enter_or_escape() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
    app.run("paint.pencil", json!({"points": [[30, 20], [30, 40]], "size": 4, "color": "#000000"})).unwrap();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    let m = egui::Modifiers::NONE;
    let journal = app.session.journal.len();
    for x in [28.0, 32.0] {
        tool_event(&mut app, ToolEvent::Down { x, y: 18.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x, y: 42.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x, y: 42.0 }, m);
    }
    // Nothing ran yet: the strokes wait, tinted, and the status says so.
    assert_eq!(app.session.journal.len(), journal, "no command before Enter");
    assert_eq!(app.remove_pending.len(), 2);
    assert!(app.remove_trail.is_some());
    assert!(app.ui.status.contains("2") && !app.ui.status_error, "{}", app.ui.status);
    // Esc drops them.
    assert!(remove_pending_cancel(&mut app));
    assert!(app.remove_pending.is_empty() && app.remove_trail.is_none());
    assert!(!remove_pending_cancel(&mut app));
    // Collected again, Enter applies them as one `paint.remove` with both strokes.
    for x in [28.0, 32.0] {
        tool_event(&mut app, ToolEvent::Down { x, y: 18.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x, y: 42.0 }, m);
    }
    let bg = active(&app).surface().unwrap().rgba(70, 30);
    assert!(remove_pending_commit(&mut app));
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let (id, p) = app.session.journal.last().cloned().expect("a journaled command");
    assert_eq!(id, "paint.remove");
    assert_eq!(p["strokes"].as_array().map(Vec::len), Some(2));
    assert!(p.get("points").is_none());
    assert!(app.remove_pending.is_empty());
    let px = active(&app).surface().unwrap().rgba(30, 30);
    assert!(px.iter().zip(bg).all(|(a, b)| (a - b).abs() < 0.02), "{px:?} vs {bg:?}");
    assert!(!remove_pending_commit(&mut app), "nothing left to apply");
}

#[test]
fn every_waiting_remove_stroke_is_tinted_in_full() {
    // #6555: each stroke used to be stamped only past the previous stroke's point count.
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    let strokes: [&[[f64; 3]]; 3] = [
        &[[10.0, 10.0, 1.0], [40.0, 10.0, 1.0], [70.0, 10.0, 1.0]],
        &[[10.0, 30.0, 1.0], [40.0, 30.0, 1.0]],
        &[[10.0, 50.0, 1.0], [20.0, 50.0, 1.0], [30.0, 50.0, 1.0], [50.0, 50.0, 1.0], [90.0, 50.0, 1.0]],
    ];
    for s in strokes {
        remove_pending_add(&mut app, s);
    }
    assert_eq!(app.remove_pending.len(), 3);
    let t = app.remove_trail.as_ref().expect("the waiting strokes are tinted");
    let k = f64::from(t.scale());
    for s in strokes {
        for w in s.windows(2) {
            for i in 0..=8 {
                let f = f64::from(i) / 8.0;
                let (x, y) = (w[0][0] + (w[1][0] - w[0][0]) * f, w[0][1] + (w[1][1] - w[0][1]) * f);
                assert_eq!(t.coverage((x / k) as usize, (y / k) as usize), 255, "({x}, {y}) on a waiting stroke is tinted");
            }
        }
    }
    // Enter, Esc, another tool and another document all drop the tint with the strokes.
    assert!(remove_pending_commit(&mut app));
    assert!(app.remove_trail.is_none());
    remove_pending_add(&mut app, strokes[0]);
    assert!(remove_pending_cancel(&mut app) && app.remove_trail.is_none());
    remove_pending_add(&mut app, strokes[1]);
    app.ui.tool = Tool::SpotHealing;
    assert!(!remove_pending_live(&mut app) && app.remove_trail.is_none());
    app.ui.tool = Tool::Remove;
    remove_pending_add(&mut app, strokes[2]);
    app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
    assert!(!remove_pending_live(&mut app) && app.remove_trail.is_none());
}

#[test]
fn queued_remove_cannot_commit_after_layer_or_channel_target_changes() {
    for change in ["layer", "color", "mask"] {
        let mut app = app();
        app.run("layer.new.layer", json!({"name": "Original"})).unwrap();
        if change == "color" {
            app.run("channel.target", json!({"channel": "red"})).unwrap();
        }
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
        assert_eq!(app.remove_pending.len(), 1);
        assert!(remove_pending_live(&mut app));

        match change {
            "layer" => {
                app.run("layer.new.layer", json!({"name": "Other"})).unwrap();
            }
            "color" => {
                app.run("channel.target", json!({"channel": "blue"})).unwrap();
            }
            _ => {
                app.run("layer.layerMask.revealAll", json!({})).unwrap();
                app.ui.mask_target = true;
            }
        }

        let journal = app.session.journal.len();
        assert!(!remove_pending_commit(&mut app), "{change}: Enter must not retarget a queued stroke");
        assert_eq!(app.session.journal.len(), journal, "{change}: no Remove command ran");
        assert!(app.remove_pending.is_empty() && app.remove_trail.is_none(), "{change}: queue cleared");
        assert!(app.remove_pending_target.is_none());
    }
}

#[test]
fn queued_remove_starts_a_new_queue_after_target_switch() {
    let mut app = app();
    app.run("layer.new.layer", json!({"name": "Original"})).unwrap();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
    app.run("layer.new.layer", json!({"name": "Other"})).unwrap();
    remove_pending_add(&mut app, &[[40.0, 10.0, 1.0], [44.0, 12.0, 1.0]]);
    assert_eq!(app.remove_pending.len(), 1, "strokes from different layers must not be combined");
    assert_eq!(app.remove_pending[0]["points"][0][0], 40.0);
    assert!(remove_pending_live(&mut app));
}

#[test]
fn remove_strokes_waiting_are_dropped_when_the_tool_changes() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[10.0, 10.0, 1.0], [20.0, 10.0, 1.0]]);
    assert_eq!(app.remove_pending.len(), 1);
    assert!(remove_pending_live(&mut app));
    app.ui.tool = Tool::SpotHealing;
    assert!(!remove_pending_live(&mut app));
    assert!(app.remove_pending.is_empty(), "the strokes belong to the Remove tool");
}

#[test]
fn create_a_new_layer_sends_new_layer_to_the_command() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_new_layer = true;
    let layers = app.session.active().unwrap().doc.walk().len();
    let m = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 18.0, pressure: 1.0 }, m);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 42.0 }, m);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    let (id, p) = app.session.journal.last().cloned().unwrap();
    assert_eq!((id.as_str(), p["newLayer"].as_bool()), ("paint.remove", Some(true)));
    assert_eq!(app.session.active().unwrap().doc.walk().len(), layers + 1, "the fill went to a new layer");
}

#[test]
fn red_eye_is_in_the_j_flyout() {
    let j = [Tool::Remove, Tool::SpotHealing, Tool::Healing, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye];
    assert!(j.contains(&Tool::RedEye));
    assert!(j.iter().all(|t| t.key() == 'J'));
}

#[test]
fn j_cycles_from_remove() {
    // J from another group picks the first J tool, and repeated J goes on in this order.
    let group: Vec<Tool> = Tool::ALL.iter().copied().filter(|t| t.key() == 'J').collect();
    assert_eq!(group, [Tool::Remove, Tool::SpotHealing, Tool::Healing, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye]);
}

#[test]
fn s_group_is_clone_stamp_then_pattern_stamp() {
    assert_eq!(Tool::CloneStamp.key(), 'S');
    assert_eq!(Tool::PatternStamp.key(), 'S');
    let group: Vec<Tool> = Tool::ALL.iter().copied().filter(|t| t.key() == 'S').collect();
    assert_eq!(group, vec![Tool::CloneStamp, Tool::PatternStamp]);
    assert_eq!(Tool::from_name("patternStamp"), Some(Tool::PatternStamp));
}

#[test]
fn pattern_stamp_stroke_paints_pixels_and_keeps_phase_when_aligned() {
    let mut app = app();
    assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[20.0, 30.0, 1.0], [50.0, 30.0, 1.0]], egui::Modifiers::NONE));
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.patternStamp"));
    assert!(app.ui.pattern_stamp_phase.is_some());
    let px = active(&app).surface().unwrap().rgba(30, 30);
    assert!(px[3] > 0.5, "the stamp paints through the brush: {px:?}");
    app.ui.tool_options.pattern_stamp_aligned = false;
    app.ui.pattern_stamp_phase = None;
    assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[70.0, 30.0, 1.0]], egui::Modifiers::NONE));
    assert!(app.ui.pattern_stamp_phase.is_none(), "unaligned strokes do not keep phase");
}

#[test]
fn pattern_stamp_does_not_require_a_clone_source() {
    let mut app = app();
    assert!(app.ui.clone_source.is_none());
    assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[12.0, 30.0, 1.0]], egui::Modifiers::NONE));
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert!(app.ui.clone_source.is_none());
}

#[test]
fn art_history_is_in_the_y_flyout() {
    let y = [Tool::HistoryBrush, Tool::ArtHistoryBrush];
    assert_eq!(y[0], Tool::HistoryBrush);
    assert_eq!(y[1], Tool::ArtHistoryBrush);
    assert!(y.iter().all(|t| t.key() == 'Y'));
    assert!(Tool::ArtHistoryBrush.is_brushlike());
}

#[test]
fn art_history_options_bar_defaults() {
    use crate::theme::ThemeKind;
    use egui::vec2;
    use egui_kittest::kittest::Queryable;
    let mut app = app();
    app.ui.tool = Tool::ArtHistoryBrush;
    let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 60.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::panels::options_bar(app, ui);
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, ThemeKind::Studio);
    h.run_steps(4);
    h.get_by_label("Style");
    h.get_by_label("Area");
    h.get_by_label("Tolerance");
    assert_eq!(h.state().ui.tool_options.art_history_style, "tightMedium");
    assert_eq!(h.state().ui.tool_options.art_history_area, 50.0);
    assert_eq!(h.state().ui.tool_options.art_history_tolerance, 0.0);
}

#[test]
fn art_history_stroke_runs_the_command() {
    let mut app = app();
    app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 20, "color": "#3060c0"})).unwrap();
    app.ui.tool = Tool::ArtHistoryBrush;
    app.ui.tool_options.art_history_style = "tightMedium".into();
    app.ui.tool_options.art_history_tolerance = 0.0;
    drag(&mut app, Tool::ArtHistoryBrush);
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.artHistoryBrush"));
}

/// Waiting masks cannot move onto a different selection or newer source pixels.
#[test]
fn queued_remove_cancels_after_source_revision_or_options_change() {
    for change in ["selection", "source", "output"] {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
        match change {
            "selection" => {
                app.run("select.rect", json!({"x": 1, "y": 1, "width": 5, "height": 5})).unwrap();
            }
            "source" => {
                app.run("paint.pencil", json!({"points": [[80, 40]], "size": 2, "color": "#ff0000"})).unwrap();
            }
            _ => app.ui.tool_options.remove_new_layer = true,
        }
        let journal = app.session.journal.len();
        assert!(!remove_pending_commit(&mut app), "{change}: a queued mask must keep its original source and output settings");
        assert_eq!(app.session.journal.len(), journal, "{change}: Enter cannot submit the stale queue");
        assert!(app.remove_pending.is_empty() && app.remove_trail.is_none());
    }
}

/// A press during processing is refused before capturing a stroke that cannot be committed.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn remove_busy_document_refuses_new_gesture_before_capture() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    let (done, wait) = std::sync::mpsc::channel();
    let started = app
        .session
        .start_job(
            "paint.remove",
            json!({}),
            "Remove",
            true,
            move |_| {
                wait.recv().map_err(|e| photocraft_engine::EngineError::Other(e.to_string()))?;
                Ok(())
            },
            |_, ()| Ok(json!({})),
        )
        .unwrap();
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, egui::Modifiers::NONE);
    let captured = app.drag.is_some();
    done.send(()).unwrap();
    if let photocraft_engine::jobs::Started::Job(id) = started {
        app.session.wait_job(id).unwrap();
    }
    assert!(!captured, "processing must refuse a Remove press before capture");
    assert!(app.remove_pending.is_empty());
}

#[test]
fn queued_remove_freezes_method_sampling_and_model_preferences() {
    for change in ["method", "sampling", "after", "selected model", "enabled models", "processing"] {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
        match change {
            "method" => app.ui.tool_options.remove_method = "classical".into(),
            "sampling" => app.ui.tool_options.sample_all.remove = true,
            "after" => app.ui.tool_options.remove_after_each_stroke = true,
            "selected model" => {
                app.run("prefs.set", json!({"path":"generativeRemove.selectedModel", "value":"migan-v1"})).unwrap();
            }
            "enabled models" => {
                app.run("prefs.set", json!({"path":"generativeRemove.enabledModels", "value":["migan-v1"]})).unwrap();
            }
            _ => {
                app.run("prefs.set", json!({"path":"generativeRemove.enabled", "value":true})).unwrap();
            }
        }
        let journal = app.session.journal.len();
        assert!(!remove_pending_commit(&mut app), "{change}: stale queue must not run");
        assert_eq!(app.session.journal.len(), journal, "{change}: no Remove command");
        assert!(app.remove_pending.is_empty() && app.remove_pending_target.is_none());
    }
}

#[test]
fn remove_gesture_does_not_retarget_when_source_or_settings_change_before_release() {
    for change in ["selection", "source", "output", "method", "processing", "tool"] {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.drag.is_some(), "{change}: press starts a gesture");
        match change {
            "selection" => {
                app.run("select.rect", json!({"x":1,"y":1,"width":5,"height":5})).unwrap();
            }
            "source" => {
                app.run("paint.pencil", json!({"points":[[80,40]],"size":2,"color":"#ff0000"})).unwrap();
            }
            "output" => app.ui.tool_options.remove_new_layer = true,
            "method" => app.ui.tool_options.remove_method = "classical".into(),
            "processing" => {
                app.run("prefs.set", json!({"path":"generativeRemove.enabled", "value":true})).unwrap();
            }
            _ => app.ui.tool = Tool::Pencil,
        }
        let journal = app.session.journal.len();
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 20.0 }, egui::Modifiers::NONE);
        assert_eq!(app.session.journal.len(), journal, "{change}: release must not submit Remove");
        assert!(app.remove_pending.is_empty(), "{change}: no queued stroke");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn queued_remove_enter_while_busy_keeps_the_mask_and_escape_discards_it() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
    let (done, wait) = std::sync::mpsc::channel();
    let started = app
        .session
        .start_job(
            "paint.remove",
            json!({}),
            "Remove",
            true,
            move |_| {
                wait.recv().map_err(|e| photocraft_engine::EngineError::Other(e.to_string()))?;
                Ok(())
            },
            |_, ()| Ok(json!({})),
        )
        .unwrap();
    let journal = app.session.journal.len();
    assert!(!remove_pending_commit(&mut app));
    assert_eq!(app.session.journal.len(), journal);
    assert_eq!(app.remove_pending.len(), 1, "a refused Enter does not lose the waiting mask");
    assert!(remove_pending_cancel(&mut app));
    done.send(()).unwrap();
    if let photocraft_engine::jobs::Started::Job(id) = started {
        app.session.wait_job(id).unwrap();
    }
    assert!(!remove_pending_commit(&mut app));
    assert_eq!(app.session.journal.len(), journal);
}

#[test]
fn remove_enter_during_a_gesture_waits_until_the_gesture_is_finished() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
    tool_event(&mut app, ToolEvent::Down { x: 25.0, y: 25.0, pressure: 1.0 }, egui::Modifiers::NONE);
    assert!(!remove_pending_commit(&mut app));
    assert_eq!(app.remove_pending.len(), 1);
    tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 25.0 }, egui::Modifiers::NONE);
    assert_eq!(app.remove_pending.len(), 2);
    assert!(remove_pending_commit(&mut app));
    let journal = app.session.journal.len();
    assert!(!remove_pending_commit(&mut app));
    assert_eq!(app.session.journal.len(), journal, "a second Enter cannot submit the same strokes again");
}

#[test]
fn remove_keyboard_escape_discards_and_enter_applies_once() {
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        app()
    });
    h.run_steps(4);
    h.state_mut().ui.tool = Tool::Remove;
    h.state_mut().ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(h.state_mut(), &[[12.0, 10.0, 1.0], [16.0, 12.0, 1.0]]);
    let journal = h.state().session.journal.len();
    h.key_press(egui::Key::Escape);
    h.run_steps(4);
    assert!(h.state().remove_pending.is_empty());
    assert_eq!(h.state().session.journal.len(), journal, "Escape discards the queued mask");
    remove_pending_add(h.state_mut(), &[[22.0, 10.0, 1.0], [26.0, 12.0, 1.0]]);
    h.key_press(egui::Key::Enter);
    h.run_steps(4);
    assert!(h.state().remove_pending.is_empty());
    assert_eq!(h.state().session.journal.len(), journal + 1);
    h.key_press(egui::Key::Enter);
    h.run_steps(4);
    assert_eq!(h.state().session.journal.len(), journal + 1, "Enter cannot apply twice");
}

#[test]
fn remove_dispatches_the_chosen_method_and_never_installs_a_model_on_a_stroke() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_method = "auto".into();
    drag(&mut app, Tool::Remove);
    let (id, params) = app.session.journal.last().unwrap();
    assert_eq!(id, "paint.remove");
    assert_eq!(params["method"], "classical", "the journal records the effective method");
    assert!(app.ui.status.contains("local AI processing is disabled"), "Auto's fallback must be visible: {}", app.ui.status);
    let journal = app.session.journal.len();
    app.ui.tool_options.remove_method = "ai".into();
    tool_event(&mut app, ToolEvent::Down { x: 15.0, y: 20.0, pressure: 1.0 }, egui::Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, egui::Modifiers::NONE);
    assert!(app.ui.status_error, "explicit AI must report its unavailable processing");
    assert_eq!(app.session.journal.len(), journal, "a failed AI stroke cannot become a classical edit");
    assert!(!app.session.jobs_with_recent().iter().any(|job| job.command == "ai.models.install"));
}

#[test]
fn remove_options_models_button_opens_the_manager() {
    use egui_kittest::kittest::Queryable;
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 720.0)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        app()
    });
    h.run_steps(4);
    h.state_mut().ui.tool = Tool::Remove;
    h.run_steps(4);
    h.get_by_label("Models…").click();
    h.run_steps(4);
    assert!(h.state().ui.dialogs.iter().any(|dialog| dialog.fields.get("__prefsui") == Some(&json!("removeModels"))));
    assert!(!h.state().session.has_jobs());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn background_remove_reports_automatic_fallback_and_effective_method() {
    let mut app = app();
    let started = app
        .session
        .start_job(
            "paint.remove",
            json!({}),
            "Remove",
            true,
            |_| Ok(()),
            |_, ()| Ok(json!({"effectiveMethod":"classical", "fallbackReason":"local AI processing is disabled"})),
        )
        .unwrap();
    let photocraft_engine::jobs::Started::Job(id) = started else { panic!("native job should run in the background") };
    app.session.wait_job(id).unwrap();
    let ctx = egui::Context::default();
    crate::jobs_ui::tick(&mut app, &ctx);
    assert!(app.ui.status.contains("local AI processing is disabled"), "{}", app.ui.status);
    let seen = crate::control::inspect(&app, &ctx);
    assert_eq!(seen["remove"]["lastResult"]["effectiveMethod"], "classical");
    assert_eq!(seen["remove"]["lastResult"]["fallbackReason"], "local AI processing is disabled");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn remove_press_started_while_busy_cannot_be_replayed_after_job_finishes() {
    for dragged in [false, true] {
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app()
        });
        h.run_steps(4);
        h.state_mut().ui.tool = Tool::Remove;
        h.run_steps(2);
        let pos = h.state().last_canvas_rect.center();
        let (done, wait) = std::sync::mpsc::channel();
        let started = h
            .state_mut()
            .session
            .start_job(
                "paint.remove",
                json!({}),
                "Remove",
                true,
                move |_| {
                    wait.recv().map_err(|error| photocraft_engine::EngineError::Other(error.to_string()))?;
                    Ok(())
                },
                |_, ()| Ok(json!({})),
            )
            .unwrap();
        let photocraft_engine::jobs::Started::Job(id) = started else { panic!("native job should run in the background") };
        let journal = h.state().session.journal.len();
        h.event(egui::Event::PointerMoved(pos));
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.step();
        assert!(h.state().drag.is_none());
        done.send(()).unwrap();
        h.state_mut().session.wait_job(id).unwrap();
        let end = if dragged { pos + egui::vec2(20.0, 0.0) } else { pos };
        if dragged {
            h.event(egui::Event::PointerMoved(end));
            h.step();
            assert!(h.state().drag.is_none(), "a refused press cannot become a drag");
        }
        h.event(egui::Event::PointerButton { pos: end, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(4);
        assert_eq!(h.state().session.journal.len(), journal, "a refused press cannot become a new stroke when the job finishes");
        assert!(h.state().drag.is_none() && h.state().remove_pending.is_empty());
        // The refusal ends on release: a later, independent click is accepted.
        h.event(egui::Event::PointerMoved(pos));
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.step();
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(4);
        assert_eq!(h.state().session.journal.len(), journal + 1, "the next press is not suppressed");
    }
}

#[test]
fn review_queue_is_discarded_when_method_changes_back_without_a_frame() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[40.0, 30.0, 1.0]]);
    let ctx = egui::Context::default();
    for method in ["ai", "classical"] {
        let (request, _reply) = crate::control::ControlRequest::new("ui.set", json!({"removeMethod":method}));
        let _ = crate::control::handle(&mut app, &ctx, &request);
    }
    assert!(app.remove_pending.is_empty(), "a reverted setting transition retained old strokes");
}

#[test]
fn review_queue_does_not_survive_a_document_round_trip() {
    let mut app = app();
    app.run("file.new", json!({"width":100,"height":60})).unwrap();
    app.session.set_active(0);
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    remove_pending_add(&mut app, &[[40.0, 30.0, 1.0]]);
    app.session.set_active(1);
    app.session.set_active(0);
    assert!(!remove_pending_live(&mut app), "a document round trip revived old strokes");
}

#[test]
fn review_queued_brush_geometry_survives_tip_and_scatter_changes() {
    let _guard = REMOVE_TEST_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    struct Black;
    impl photocraft_ml::RemovalBackend for Black {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(
            &self,
            _: &str,
            input: &photocraft_ml::RemovalInput,
            _: &dyn photocraft_ml::RemovalControl,
        ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
            Ok(photocraft_ml::RemovalOutput { rgb: vec![0.0; input.rgb.len()], edit_alpha: None, inference_scale: 1.0, load_ms: 0.0, forward_ms: 0.0 })
        }
    }
    let run = |change: bool| {
        let mut app = app();
        app.run("file.new", json!({"width":100,"height":60,"background":"white"})).unwrap();
        app.background_jobs = false;
        app.session.set_removal_service(std::sync::Arc::new(photocraft_ml::RemovalService::new(None).with_backend(std::sync::Arc::new(Black))));
        app.session.edit_prefs(|prefs| prefs.generative_remove.enabled = true);
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_method = "ai".into();
        app.ui.tool_options.remove_after_each_stroke = false;
        app.session.tools.brush.roundness = 0.2;
        app.session.tools.brush.angle = 30.0;
        remove_pending_add(&mut app, &[[50.0, 30.0, 1.0]]);
        if change {
            app.session.tools.brush.roundness = 1.0;
            app.session.tools.brush.angle = 0.0;
            app.session.tools.brush.scattering.enabled = true;
            app.session.tools.brush.scattering.scatter.jitter = 1.0;
        }
        assert!(remove_pending_commit(&mut app));
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let st = app.session.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().read_region(st.doc.bounds())
    };
    let unchanged = run(false);
    let changed = run(true);
    let different = unchanged.iter().zip(&changed).filter(|(left, right)| left != right).count();
    assert_eq!(different, 0, "changed brush geometry changed the cached tinted footprint submitted to Remove");
}

#[test]
fn review2_remove_gesture_keeps_the_brush_captured_at_press() {
    for after_each in [false, true] {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = after_each;
        app.session.tools.brush.roundness = 0.3;
        tool_event(&mut app, ToolEvent::Down { x: 45.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
        app.run("tools.setBrush", json!({"brush":{"size":40,"roundness":1.0}})).unwrap();
        tool_event(&mut app, ToolEvent::Up { x: 45.0, y: 30.0 }, egui::Modifiers::NONE);
        if !after_each {
            assert!(remove_pending_commit(&mut app));
        }
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let (_, params) = app.session.journal.last().unwrap();
        let brush = if after_each { &params["brush"] } else { &params["strokes"][0]["brush"] };
        assert_eq!(brush["size"].as_f64(), Some(16.0), "release used the changed brush");
        assert!((brush["roundness"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    }
}

#[test]
fn review2_queued_remove_does_not_survive_context_round_trips_in_one_action() {
    let sets = [
        vec![("channel.target".into(), json!({"channel":"red"})), ("channel.target".into(), json!({"channel":"composite"}))],
        vec![
            ("prefs.set".into(), json!({"path":"generativeRemove.enabled","value":true})),
            ("prefs.set".into(), json!({"path":"generativeRemove.enabled","value":false})),
        ],
    ];
    for steps in sets {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        remove_pending_add(&mut app, &[[40.0, 30.0, 1.0]]);
        app.session.actions.list.push(photocraft_engine::actions_cmds::Action::new("Context round trip", steps));
        app.run("actions.play", json!({"action":"Context round trip"})).unwrap();
        assert!(app.remove_pending.is_empty(), "nested context transitions retained the queued strokes");
    }
}

#[test]
fn review3_pending_remove_rejects_excessive_dabs_before_tint_allocation() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    app.session.tools.brush.size = 1.0;
    app.session.tools.brush.scattering.enabled = true;
    app.session.tools.brush.scattering.count = 16;
    let points: Vec<_> = (0..10_000).map(|i| [40.0 + f64::from(i % 2) * 10.0, 30.0, 1.0]).collect();
    remove_pending_add(&mut app, &points);
    assert!(app.remove_pending.is_empty());
    assert!(app.remove_trail.is_none(), "an inadmissible stroke allocated its tint renderer");
    assert!(app.ui.status_error);
    assert!(app.ui.status.contains("brush work"), "{}", app.ui.status);
}

#[test]
fn osor_effect_choice_survives_state_and_requests_preview() {
    let options: crate::state::ToolOptions = serde_json::from_value(json!({
        "remove_method": "ai", "remove_effects": true
    }))
    .unwrap();
    let params = remove_params(&options);
    assert_eq!(params["removeEffects"], true, "effects choice must reach the engine");
    assert_eq!(params["preview"], true, "effects must be reviewed before application");
    let restored: crate::state::ToolOptions = serde_json::from_value(serde_json::to_value(options).unwrap()).unwrap();
    assert_eq!(remove_params(&restored)["removeEffects"], true);
}

#[test]
fn osor_selected_model_requests_review_even_without_effects() {
    let mut app = app();
    app.session.edit_prefs(|p| p.generative_remove.selected_model = "osor-sdxl".into());
    app.ui.tool_options.remove_method = "ai".into();
    let request = removal_request(&app, &app.ui.tool_options);
    assert_eq!(request["preview"], true);
    assert!(request.get("removeEffects").is_none());
}

#[test]
fn osor_preview_canvas_and_apply_use_the_same_candidate() {
    let _guard = REMOVE_TEST_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    struct Black;
    impl photocraft_ml::RemovalBackend for Black {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(
            &self,
            _: &str,
            input: &photocraft_ml::RemovalInput,
            _: &dyn photocraft_ml::RemovalControl,
        ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
            Ok(photocraft_ml::RemovalOutput { rgb: vec![0.; input.rgb.len()], edit_alpha: None, inference_scale: 1., load_ms: 0., forward_ms: 0. })
        }
    }
    let mut app = app();
    app.run("file.new", json!({"width":100,"height":60,"background":"white"})).unwrap();
    app.background_jobs = false;
    app.session.set_removal_service(std::sync::Arc::new(photocraft_ml::RemovalService::new(None).with_backend(std::sync::Arc::new(Black))));
    app.session.edit_prefs(|p| p.generative_remove.enabled = true);
    app.ui.tool = Tool::Remove;
    let source = app.session.active().unwrap().doc.clone();
    app.run("paint.remove", json!({"method":"ai", "modelId":"lama", "points":[[50,30]], "size":16, "preview":true})).unwrap();
    let (candidate, _) = app.session.removal_preview_document().unwrap();
    let index = app.session.active_index().unwrap();
    let (shown, _) = crate::canvas::display_doc(&mut app, index);
    assert!(std::sync::Arc::ptr_eq(&shown, &candidate), "canvas must display the engine candidate");
    assert!(std::sync::Arc::ptr_eq(&source, &app.session.active().unwrap().doc), "preview leaves source unchanged");
    remove_preview_apply(&mut app);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert!(app.session.removal_preview_document().is_none());
    assert_eq!(
        photocraft_compose::render(&candidate, candidate.bounds()).px,
        photocraft_compose::render(&app.session.active().unwrap().doc, candidate.bounds()).px
    );
    app.run("edit.undo", json!({})).unwrap();
    assert_eq!(photocraft_compose::render(&source, source.bounds()).px, photocraft_compose::render(&app.session.active().unwrap().doc, source.bounds()).px);
}

#[test]
fn hybrid_first_use_offers_recommended_models_without_downloading() {
    let root = std::env::temp_dir().join("photocraft-hybrid-ui-no-models");
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1100., 800.)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = app();
        app.session.set_removal_service(std::sync::Arc::new(photocraft_ml::RemovalService::new(Some(root.clone()))));
        app.ui.tool = Tool::Remove;
        app
    });
    h.run_steps(4);
    let setup = h.state().ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("removeSetup"));
    assert!(setup.is_some(), "first native Remove use must explain model choices");
    assert!(
        matches!(setup.unwrap().fields["workflow"].as_str(), Some("recommended" | "lama" | "compact" | "classical")),
        "first-use setup must propose a supported profile for this host"
    );
    assert!(!h.state().session.has_jobs(), "offering choices never downloads weights");
}

#[test]
fn hybrid_declined_setup_does_not_prompt_again() {
    let root = std::env::temp_dir().join("photocraft-hybrid-ui-declined");
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1100., 800.)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = app();
        app.session.set_removal_service(std::sync::Arc::new(photocraft_ml::RemovalService::new(Some(root.clone()))));
        app.run("remove.setup", json!({"workflow":"classical"})).unwrap();
        app.ui.tool = Tool::Remove;
        app
    });
    h.run_steps(4);
    assert!(!h.state().ui.dialogs.iter().any(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("removeSetup")));
    assert!(!h.state().session.has_jobs());
}

#[test]
fn hybrid_queued_strokes_keep_distinct_repair_hints() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    app.ui.tool_options.remove_after_each_stroke = false;
    app.ui.tool_options.remove_repair_type = "detail".into();
    remove_pending_add(&mut app, &[[10.0, 10.0, 1.0], [30.0, 10.0, 1.0]]);
    app.ui.tool_options.remove_repair_type = "object".into();
    assert!(remove_pending_live(&mut app), "a new hint must not discard the previous region");
    remove_pending_add(&mut app, &[[70.0, 40.0, 1.0]]);
    assert_eq!(app.remove_pending.len(), 2);
    assert_eq!(app.remove_pending[0]["repairType"], "detail");
    assert_eq!(app.remove_pending[1]["repairType"], "object");
}

#[test]
fn hybrid_gesture_does_not_retarget_if_repair_hint_changes_while_pressed() {
    let mut app = app();
    app.ui.tool = Tool::Remove;
    let modifiers = egui::Modifiers::NONE;
    tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, modifiers);
    app.ui.tool_options.remove_repair_type = "object".into();
    tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 30.0 }, modifiers);
    assert!(app.ui.status_error);
    assert!(app.ui.status.contains("cancelled"), "{}", app.ui.status);
    assert!(!app.session.has_jobs());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn removal_preview_survives_tool_persistence_and_leaves_full_screen() {
    let _guard = REMOVE_TEST_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    struct Black;
    impl photocraft_ml::RemovalBackend for Black {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(
            &self,
            _: &str,
            input: &photocraft_ml::RemovalInput,
            _: &dyn photocraft_ml::RemovalControl,
        ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
            Ok(photocraft_ml::RemovalOutput { rgb: vec![0.; input.rgb.len()], edit_alpha: None, inference_scale: 1., load_ms: 0., forward_ms: 0. })
        }
    }
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = app();
        app.run("file.new", json!({"width":100,"height":60,"background":"white"})).unwrap();
        app.session.set_removal_service(std::sync::Arc::new(photocraft_ml::RemovalService::new(None).with_backend(std::sync::Arc::new(Black))));
        app.session.edit_prefs(|p| p.generative_remove.enabled = true);
        app.ui.tool = Tool::Remove;
        app.background_jobs = true;
        app
    });
    h.run_steps(4);
    let ctx = h.ctx.clone();
    let source = h.state().session.active().unwrap().doc.clone();
    let history = h.state().session.active().unwrap().history.past_len();
    let journal = h.state().session.journal.len();
    let saved_brush = h.state().session.prefs().last_brush.clone();
    let saved_options = h.state().session.prefs().tool_options.clone();
    h.state_mut().run("tools.setBrush", json!({"size":23})).unwrap();
    h.state_mut().ui.tool_options.remove_after_each_stroke = false;
    let started = h.state_mut().run("paint.remove", json!({"method":"ai","modelId":"lama","points":[[50,30]],"size":16,"preview":true})).unwrap();
    let job = photocraft_engine::jobs::JobId(started["job"].as_u64().unwrap());
    let submitted_prefs_rev = h.state().session.prefs.rev();

    // Stroke release saves the brush and options at the end of the same frame, after submission.
    // Those preferences describe future gestures; they do not change this job's frozen input.
    crate::last_brush::persist(h.state_mut(), &ctx);
    crate::tool_memory::persist(h.state_mut(), &ctx);
    assert_ne!(h.state().session.prefs().last_brush, saved_brush);
    assert_ne!(h.state().session.prefs().tool_options, saved_options);
    assert!(h.state().session.prefs.rev() > submitted_prefs_rev);
    let result = h.state_mut().session.wait_job(job).expect("saving tool state must not discard a completed Remove candidate");
    assert_eq!(result["preview"], true);
    crate::jobs_ui::tick(h.state_mut(), &ctx);
    let (candidate, preview_id) = h.state().session.removal_preview_document().unwrap();

    // The candidate also survives view-only persistence while displayed in full screen.
    crate::menus::invoke(h.state_mut(), &ctx, "view.screenMode.fullScreen", json!({})).unwrap();
    h.state_mut().ui.view.show.layer_edges = true;
    let published_prefs_rev = h.state().session.prefs.rev();
    crate::view_cmds::persist(h.state_mut(), &ctx);
    assert!(h.state().session.prefs.rev() > published_prefs_rev);
    h.run_steps(4);
    assert_eq!(h.state().session.removal_preview_status()["id"], preview_id);
    let index = h.state().session.active_index().unwrap();
    let (shown, _) = crate::canvas::display_doc(h.state_mut(), index);
    assert!(std::sync::Arc::ptr_eq(&shown, &candidate), "the canvas must keep showing the reviewed candidate");
    assert!(std::sync::Arc::ptr_eq(&source, &h.state().session.active().unwrap().doc));

    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(h.state().session.removal_preview_document().is_none(), "first Escape discards the candidate");
    assert_eq!(h.state().ui.view.screen_mode, "fullScreen");
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert_eq!(h.state().ui.view.screen_mode, "standard", "next Escape returns to the main view");
    assert!(std::sync::Arc::ptr_eq(&source, &h.state().session.active().unwrap().doc));
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    assert_eq!(h.state().session.journal.len(), journal + 1, "only changing the brush was journalled");
    assert!(!h.state().ui.status_error, "{}", h.state().ui.status);
}
