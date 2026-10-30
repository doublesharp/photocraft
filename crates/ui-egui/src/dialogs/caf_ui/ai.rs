//! UI ownership of one engine-generated LaMa preview. The engine owns pixels and admission.
use super::{CafWorkspace, PhotocraftApp, Rect, Value, json};
use photocraft_engine::jobs::JobId;

#[derive(Default)]
pub(super) struct State {
    pub(super) job: Option<(u64, JobId)>,
    pub(super) id: Option<u64>,
    params: Option<Value>,
    pub(super) applied: bool,
    visible: bool,
    applied_context: Option<(u64, u64)>,
}

pub(super) fn readiness(app: &PhotocraftApp) -> Result<(), String> {
    let prefs = &app.session.prefs().generative_remove;
    if !prefs.enabled || !prefs.enabled_models.iter().any(|id| id == "lama") {
        return Err("Enable LaMa in Models… to use local AI fill".into());
    }
    let service = app.session.removal_service();
    let status = service.status_for_fill().into_iter().find(|status| status.id == "lama");
    match status {
        Some(status) if status.installed && status.available => {
            photocraft_engine::edit_menu_cmds::validate_ai_smart_fill(&app.session, &json!({"method":"ai"})).map_err(|error| error.to_string())
        }
        Some(status) => Err(status.reason.unwrap_or_else(|| "Install LaMa in Models… to use local AI fill".into())),
        None => Err("LaMa is unavailable in this build; choose Classical".into()),
    }
}

/// Control requests must never appear to honor Classical-only sampling controls in AI mode.
/// The view retains their remembered values, and exposes them again when Classical is selected.
pub(super) fn validate_controls(app: &PhotocraftApp, value: &Value) -> Result<(), String> {
    let method = value.get("method").and_then(Value::as_str).or_else(|| app.distort.caf.as_ref().map(|w| w.opts.method.as_str()));
    if method != Some("ai") {
        return Ok(());
    }
    if ["sampling", "stroke", "brushMode", "brushSize", "colorAdaptation", "rotationAdaptation", "scale", "mirror"].iter().any(|key| value.get(*key).is_some())
        || value.get("tool").and_then(Value::as_str) == Some("brush")
    {
        return Err("LaMa uses surrounding context automatically; choose Classical for sampling-brush and adaptation controls".into());
    }
    Ok(())
}

fn error(app: &mut PhotocraftApp, message: String) {
    if let Some(w) = app.distort.caf.as_mut() {
        w.preview.finish(w.preview.want, Err(message));
    }
}

/// Cancel owned work and release only this workspace's exact candidate.
pub(super) fn clear(app: &mut PhotocraftApp) {
    let state = match app.distort.caf.as_mut() {
        Some(w) => {
            let state = std::mem::take(&mut w.ai);
            if state.job.is_some() || state.id.is_some() || state.params.is_some() || state.applied {
                w.preview.tex = None;
                w.preview.pending_image = None;
                w.preview.error = None;
                w.preview.shown = 0;
            }
            state
        }
        None => return,
    };
    let owned_id = state.id.or_else(|| {
        state.job.and_then(|(_, job)| app.session.job(job)).and_then(|job| job.result).and_then(|result| result.get("previewId").and_then(Value::as_u64))
    });
    if let Some((_, job)) = state.job {
        app.session.cancel_job(job);
    }
    // A cancelled pending job cannot publish through the job system. A completed job
    // may have been superseded, so only its recorded candidate ID belongs to us.
    if let Some(id) = owned_id {
        let _ = app.session.execute("fill.preview.discard", json!({"id":id}));
    }
}

/// A completed candidate can outlive a switch to Classical, but not changed AI inputs
/// or engine context. Its original admission remains owned until Apply, Cancel or invalidation.
fn current_candidate(app: &PhotocraftApp) -> Option<u64> {
    let w = app.distort.caf.as_ref()?;
    let id = w.ai.id?;
    if w.ai.params.as_ref() != Some(&super::preview::ai_params(w)) {
        return None;
    }
    let status = app.session.fill_preview_status();
    (status.get("ready").and_then(Value::as_bool) == Some(true) && status.get("id").and_then(Value::as_u64) == Some(id)).then_some(id)
}

/// Compare Classical with a finished AI result without starting another neural operation.
/// A pending forward is still cancelled and drained; only a completed exact candidate is kept.
pub(super) fn pause(app: &mut PhotocraftApp) {
    if current_candidate(app).is_none() || app.distort.caf.as_ref().is_some_and(|w| w.ai.job.is_some() || w.ai.applied) {
        clear(app);
        return;
    }
    if let Some(w) = app.distort.caf.as_mut()
        && w.ai.visible
    {
        w.ai.visible = false;
        w.preview.tex = None;
        w.preview.pending_image = None;
        w.preview.error = None;
        w.preview.shown = 0;
    }
}

fn image(app: &PhotocraftApp, id: u64) -> Result<(Rect, egui::ColorImage), String> {
    let w = app.distort.caf.as_ref().ok_or("Content-Aware Fill is not open")?;
    let (doc, current) = app.session.fill_preview_document().ok_or("the fill preview is stale; generate a new preview")?;
    if current != id {
        return Err("the fill preview changed; generate a new preview".into());
    }
    let rect = w.base.rect;
    let limit = w.texture_limit.clamp(1, super::PROXY_SIDE);
    let factor = (rect.width().max(rect.height()) as usize).div_ceil(limit).max(1);
    let width = (rect.width() as usize).div_ceil(factor).max(1) as u32;
    let height = (rect.height() as usize).div_ceil(factor).max(1) as u32;
    let buffer = photocraft_compose::render_reduced_rect(&doc, rect, width, height);
    let (display, _) = crate::canvas::canvas_display(app, &doc, None);
    Ok((rect, crate::canvas::display_image(display.as_deref(), &buffer)))
}

fn completed(app: &mut PhotocraftApp, generation: u64, result: Result<Value, String>) {
    let current = app.distort.caf.as_ref().is_some_and(|w| w.opts.method == "ai" && w.preview.want == generation);
    let result = result.and_then(|value| value.get("previewId").and_then(Value::as_u64).ok_or_else(|| "LaMa did not return a fill preview".to_string()));
    let id = match result {
        Ok(id) if current => id,
        Ok(id) => {
            let _ = app.session.execute("fill.preview.discard", json!({"id":id}));
            return;
        }
        Err(message) => {
            if current {
                error(app, message);
            }
            return;
        }
    };
    let rendered = image(app, id);
    if let Some(w) = app.distort.caf.as_mut() {
        w.ai.id = Some(id);
        w.ai.visible = true;
        w.preview.finish(generation, rendered);
    }
}

/// Make settings/overlay changes reuse a current candidate when inference inputs are identical.
fn reconcile(app: &mut PhotocraftApp) {
    let Some(w) = app.distort.caf.as_ref() else { return };
    let params = super::preview::plan_params(w, true);
    let same = w.ai.params.as_ref() == Some(&params);
    let status = app.session.fill_preview_status();
    let ready = status.get("ready").and_then(Value::as_bool) == Some(true);
    let id = status.get("id").and_then(Value::as_u64);
    let owned_current = current_candidate(app);
    let restore = owned_current.filter(|_| !w.ai.visible);
    let context = app.session.active().map(|st| (st.doc.id.0, st.revision));
    let applied_current = w.ai.applied && context == w.ai.applied_context;
    if let Some(id) = restore {
        let rendered = image(app, id);
        if let Some(w) = app.distort.caf.as_mut() {
            w.ai.visible = true;
            w.preview.finish(w.preview.want, rendered);
        }
        return;
    }
    if owned_current.is_some() || (same && applied_current) {
        if let Some(w) = app.distort.caf.as_mut() {
            w.preview.shown = w.preview.want;
        }
        return;
    }
    if !same || (w.ai.id.is_some() && (!ready || id != w.ai.id)) || (w.ai.applied && !applied_current) {
        clear(app);
    } else if let Some(w) = app.distort.caf.as_mut()
        && let Some((generation, _)) = &mut w.ai.job
    {
        *generation = w.preview.want;
    }
}

fn collect(app: &mut PhotocraftApp, wait: bool) {
    let owned = app.distort.caf.as_ref().and_then(|w| w.ai.job);
    let Some((generation, id)) = owned else { return };
    let outcome = if wait {
        Some(app.session.wait_job(id).map_err(|e| e.to_string()))
    } else {
        app.session.job(id).and_then(|job| match job.state {
            "done" => Some(Ok(job.result.unwrap_or(Value::Null))),
            "failed" => Some(Err(job.error.unwrap_or_else(|| "the fill preview failed".into()))),
            "cancelled" => Some(Err("the fill preview was cancelled".into())),
            _ => None,
        })
    };
    if let Some(result) = outcome {
        if let Some(w) = app.distort.caf.as_mut() {
            w.ai.job = None;
        }
        completed(app, generation, result);
    }
}

fn start(app: &mut PhotocraftApp) {
    let Some(w) = app.distort.caf.as_ref() else { return };
    if w.ai.job.is_some() || w.ai.id.is_some() || w.ai.applied || w.preview.shown == w.preview.want {
        return;
    }
    if let Err(message) = readiness(app) {
        error(app, message);
        return;
    }
    if app.session.removal_service().status().iter().any(|status| status.busy) {
        return;
    }
    let (generation, mut params) = (w.preview.want, super::preview::plan_params(w, true));
    if let Some(w) = app.distort.caf.as_mut() {
        w.ai.params = Some(params.clone());
    }
    params["preview"] = json!(true);
    match app.run(super::CMD, params) {
        Ok(result) => {
            if let Some(id) = result.get("job").and_then(Value::as_u64) {
                if let Some(w) = app.distort.caf.as_mut() {
                    w.ai.job = Some((generation, JobId(id)));
                }
            } else {
                completed(app, generation, Ok(result));
            }
        }
        Err(message) => error(app, message),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn drive(app: &mut PhotocraftApp, ctx: &egui::Context, dragging: bool) {
    reconcile(app);
    collect(app, false);
    if !dragging {
        start(app);
    }
    if app.distort.caf.as_ref().is_some_and(|w| w.preview.shown != w.preview.want) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
}

pub(super) fn wait(app: &mut PhotocraftApp) -> Result<(), String> {
    reconcile(app);
    collect(app, true);
    start(app);
    collect(app, true);
    if already_applied(app) {
        return Ok(());
    }
    let w = app.distort.caf.as_ref().ok_or("Content-Aware Fill is not open")?;
    if let Some(message) = &w.preview.error {
        return Err(message.clone());
    }
    if w.ai.id.is_none() {
        return Err("Local AI is busy; wait for the current operation to finish".into());
    }
    Ok(())
}

/// An unchanged completed fill can close with OK without creating another edit.
pub(super) fn applied_in(w: &CafWorkspace, context: Option<(u64, u64)>) -> bool {
    w.opts.method == "ai" && w.ai.applied && context == w.ai.applied_context && w.ai.params.as_ref() == Some(&super::preview::plan_params(w, true))
}

pub(super) fn already_applied(app: &PhotocraftApp) -> bool {
    let context = app.session.active().map(|st| (st.doc.id.0, st.revision));
    app.distort.caf.as_ref().is_some_and(|w| applied_in(w, context))
}

pub(super) fn can_apply(w: &CafWorkspace) -> bool {
    w.opts.method != "ai" || (w.ai.id.is_some() && w.preview.shown == w.preview.want && w.preview.error.is_none())
}

pub(super) fn apply(app: &mut PhotocraftApp) -> Result<Value, String> {
    reconcile(app);
    collect(app, false);
    let w = app.distort.caf.as_ref().ok_or("Content-Aware Fill is not open")?;
    if !can_apply(w) {
        return Err("Wait for the LaMa preview before applying".into());
    }
    let id = w.ai.id.ok_or("Generate a LaMa preview before applying")?;
    let result = app.run("fill.preview.apply", json!({"id":id}));
    if result.is_ok() {
        let applied_context = app.session.active().map(|st| (st.doc.id.0, st.revision));
        if let Some(w) = app.distort.caf.as_mut() {
            w.ai.id = None;
            w.ai.job = None;
            w.ai.applied = true;
            w.ai.applied_context = applied_context;
        }
    }
    result
}

pub(super) fn manager_open(app: &PhotocraftApp) -> bool {
    app.ui.dialogs.iter().any(|dialog| matches!(dialog.fields.get("__prefsui").and_then(Value::as_str), Some("removeModels" | "removeSetup")))
}

pub(super) fn open_models(app: &mut PhotocraftApp) {
    clear(app);
    if let Some(w) = app.distort.caf.as_mut() {
        w.stale();
    }
    crate::prefs_ui::open_remove_models(app);
}
