//! First-use choices and the explicitly authorized, sequential setup downloads.

use super::{DialogKind, JobId, Map, PhotocraftApp, RichText, Tokens, Value, VecDeque, json};

const FULL_MEMORY_MIB: u64 = 49_152;

#[derive(Default)]
pub(super) struct Runtime {
    prompted: bool,
    queue: VecDeque<String>,
    active: Option<JobId>,
    automated: bool,
    authorize: Option<fn(&str, &Value) -> photocraft_engine::Result<()>>,
    pub(super) message: String,
    pub(super) error: bool,
}

pub(super) fn busy(app: &PhotocraftApp) -> bool {
    app.prefs_rt.remove_setup.active.is_some() || !app.prefs_rt.remove_setup.queue.is_empty()
}

fn workflow(f: &Map<String, Value>) -> &str {
    f.get("workflow").and_then(Value::as_str).unwrap_or("recommended")
}

fn components(workflow: &str) -> &'static [&'static str] {
    match workflow {
        "recommended" => &["lama", "osor-sdxl"],
        "lama" => &["lama"],
        "compact" => &["migan-v1"],
        _ => &[],
    }
}

fn status(app: &mut PhotocraftApp) -> Result<Value, String> {
    app.session.execute("ai.models.list", json!({})).map_err(|error| error.to_string())
}

fn model<'a>(status: &'a Value, id: &str) -> Option<&'a Value> {
    status.get("models")?.as_array()?.iter().find(|model| model.get("id").and_then(Value::as_str) == Some(id))
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn missing(status: &Value, selected: &str) -> Vec<String> {
    components(selected).iter().filter(|id| !model(status, id).is_some_and(|m| flag(m, "installed"))).map(|id| (*id).into()).collect()
}

pub(super) fn open(app: &mut PhotocraftApp) -> u64 {
    if let Some(dialog) = app.ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("removeSetup")) {
        return dialog.id;
    }
    app.prefs_rt.remove_setup.prompted = true;
    let current = &app.session.prefs().generative_remove;
    let selected = if current.setup_complete && current.workflow != "custom" { current.workflow.as_str() } else { "recommended" };
    let fields = json!({"__prefsui":"removeSetup", "__label":"Set Up Generative Remove", "workflow":selected,
        "memoryUsageMb":u64::from(app.session.prefs().performance.memory_usage_mb).max(FULL_MEMORY_MIB), "acceptLicenses":false});
    app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default())
}

/// An explicit OSOR setup action recommends both complementary models, even after a prior
/// Classical or smaller choice. Merely opening it never changes that saved choice.
pub(super) fn open_recommended(app: &mut PhotocraftApp) -> u64 {
    let id = open(app);
    if let Some(dialog) = app.ui.dialog_mut(id) {
        dialog.fields.insert("workflow".into(), json!("recommended"));
        dialog.fields.insert("acceptLicenses".into(), json!(false));
    }
    id
}

/// Runs after the engine job events are applied. Closing a dialog never abandons an authorized
/// transfer; cancellation or failure stops the remaining queue and requires a new user choice.
pub(crate) fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if let Some(id) = app.prefs_rt.remove_setup.active {
        let job = app.session.jobs_with_recent().into_iter().find(|job| job.id == id);
        match job {
            Some(job) if job.state == "running" => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Some(job) if job.state == "done" => {
                app.prefs_rt.remove_setup.active = None;
                app.prefs_rt.remove_setup.message = tl!("Model installed and verified").into();
            }
            job => {
                app.prefs_rt.remove_setup.active = None;
                app.prefs_rt.remove_setup.queue.clear();
                app.prefs_rt.remove_setup.error = true;
                app.prefs_rt.remove_setup.message =
                    job.and_then(|job| job.error).unwrap_or_else(|| tl!("Model setup stopped. Choose a setup to try again.").into());
                app.ui.status = app.prefs_rt.remove_setup.message.clone();
                app.ui.status_error = true;
            }
        }
    }
    if app.prefs_rt.remove_setup.active.is_none()
        && let Some(id) = app.prefs_rt.remove_setup.queue.pop_front()
    {
        let old_input = app.automation_input;
        let old_gate = app.session.authorize;
        app.automation_input |= app.prefs_rt.remove_setup.automated;
        if app.session.authorize.is_none() {
            app.session.authorize = app.prefs_rt.remove_setup.authorize;
        }
        let result = super::models::model_command(app, "ai.models.install", json!({"modelId":id}));
        app.session.authorize = old_gate;
        app.automation_input = old_input;
        match result {
            Ok(result) => {
                app.prefs_rt.remove_setup.active = result.get("job").and_then(Value::as_u64).map(JobId);
                if app.prefs_rt.remove_setup.active.is_none() {
                    // Native model commands normally return a job. A synchronous completion is
                    // still success, and the next model waits until the next frame.
                    ctx.request_repaint();
                }
            }
            Err(error) => {
                app.prefs_rt.remove_setup.queue.clear();
                app.prefs_rt.remove_setup.error = true;
                app.prefs_rt.remove_setup.message = error.clone();
                app.ui.status = error;
                app.ui.status_error = true;
            }
        }
    }
    let prefs = &app.session.prefs().generative_remove;
    if app.ui.tool == crate::state::Tool::Remove
        && !prefs.setup_complete
        && !prefs.enabled
        && !app.prefs_rt.remove_setup.prompted
        && app.ui.dialogs.is_empty()
        && app.ui.shell.dialog.is_none()
        && app.session.removal_service().management_available()
    {
        open(app);
    }
}

fn validation(app: &PhotocraftApp, f: &Map<String, Value>, status: &Value) -> Result<(), String> {
    if busy(app) || app.session.jobs().iter().any(|job| job.command.starts_with("ai.models.")) {
        return Err(tl!("Wait for model setup to finish or cancel it first.").into());
    }
    let selected = workflow(f);
    if !matches!(selected, "recommended" | "lama" | "compact" | "classical") {
        return Err(tl!("Choose a Remove setup.").into());
    }
    if selected == "recommended" {
        let budget = f
            .get("memoryUsageMb")
            .and_then(Value::as_u64)
            .filter(|n| (FULL_MEMORY_MIB..=1_048_576).contains(n))
            .ok_or_else(|| tl!("The recommended setup needs a memory allowance of at least 48 GiB.").to_string())?;
        if crate::gpu_canvas::physical_memory().is_some_and(|bytes| budget.saturating_mul(1 << 20) > bytes) {
            return Err(tl!("This memory allowance exceeds the memory in this computer. Choose a smaller setup.").into());
        }
    }
    for id in components(selected) {
        let entry = model(status, id).ok_or_else(|| tl!("This model is unavailable in this build.").to_string())?;
        if !flag(entry, "available") {
            return Err(tl!("This model is unavailable in this build. Choose a smaller setup.").into());
        }
        if !flag(entry, "installed") && (!flag(status, "managementAvailable") || !flag(entry, "canDownload")) {
            return Err(tl!("Model downloads are unavailable in this session.").into());
        }
    }
    if !missing(status, selected).is_empty() && f.get("acceptLicenses").and_then(Value::as_bool) != Some(true) {
        return Err(tl!("Review and accept the model licenses before downloading.").into());
    }
    Ok(())
}

pub(crate) fn owns(f: &Map<String, Value>) -> bool {
    f.get("__prefsui").and_then(Value::as_str) == Some("removeSetup")
}

fn setup_params(f: &Map<String, Value>) -> Value {
    let mut params = json!({"workflow":workflow(f)});
    if workflow(f) == "recommended" {
        params["memoryUsageMb"] = f.get("memoryUsageMb").cloned().unwrap_or(Value::Null);
    }
    params
}

pub(crate) fn preflight(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<(), String> {
    if !owns(f) {
        return Ok(());
    }
    let status = status(app)?;
    validation(app, f, &status)?;
    let commands = std::iter::once(("remove.setup", setup_params(f)))
        .chain(missing(&status, workflow(f)).into_iter().map(|id| ("ai.models.install", json!({"modelId":id}))));
    for (command, params) in commands {
        if let Some(gate) = app.session.authorize {
            gate(command, &params).map_err(|error| error.to_string())?;
        }
        if app.automation_input
            && let Some(gate) = app.services.automation_command.as_ref()
        {
            gate(command, &params)?;
        }
    }
    Ok(())
}

pub(crate) fn can_confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> bool {
    !owns(f) || status(app).is_ok_and(|status| validation(app, f, &status).is_ok())
}

pub(crate) fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    preflight(app, f)?;
    let status = status(app)?;
    let selected = workflow(f);
    let queue = missing(&status, selected);
    let result = app.run("remove.setup", setup_params(f))?;
    app.ui.tool_options.remove_method = "auto".into();
    app.ui.tool_options.remove_effects = selected == "recommended";
    app.prefs_rt.remove_setup.queue = queue.into();
    app.prefs_rt.remove_setup.automated = app.automation_input;
    app.prefs_rt.remove_setup.authorize = app.session.authorize;
    app.prefs_rt.remove_setup.error = false;
    app.prefs_rt.remove_setup.message = if busy(app) {
        tl!("Model setup is starting. Downloads continue while this window is closed.").into()
    } else {
        tl!("Remove setup saved. You can change it in Models.").into()
    };
    app.ui.status = app.prefs_rt.remove_setup.message.clone();
    app.ui.status_error = false;
    Ok(result)
}

pub(super) fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    // Keep the action row outside the scroll area, including on a 600 pt laptop window.
    let height = (ui.ctx().content_rect().height() - 160.0).max(160.0);
    egui::ScrollArea::vertical().id_salt("remove-setup-options").max_height(height).show(ui, |ui| body_contents(app, ui, f));
}

fn body_contents(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let status = match status(app) {
        Ok(status) => status,
        Err(error) => {
            ui.label(error);
            return;
        }
    };
    ui.label(tl!("Choose local models for Remove. Your images stay on this computer. A brush stroke never downloads a model."));
    ui.add_space(8.0);
    let mut selected = workflow(f).to_string();
    for (id, label, description) in [
        (
            "recommended",
            tl!("Recommended: LaMa + OSOR"),
            tl!("About 8.6 GB download. LaMa repairs small marks; OSOR removes objects, shadows and reflections. Every result opens as a preview."),
        ),
        ("lama", tl!("Smaller: LaMa"), tl!("About 208 MB download. Local removal with a smaller memory requirement.")),
        ("compact", tl!("Compact: MI-GAN"), tl!("About 30 MB download. The smallest AI model.")),
        ("classical", tl!("Classical: no download"), tl!("Use the built-in removal algorithm. You can install AI models later.")),
    ] {
        if ui.radio_value(&mut selected, id.into(), label).changed() {
            f.insert("acceptLicenses".into(), json!(false));
        }
        ui.indent(id, |ui| {
            ui.label(RichText::new(description).color(t.text_dim));
        });
        ui.add_space(4.0);
    }
    f.insert("workflow".into(), json!(selected));
    if selected == "recommended" {
        ui.separator();
        ui.label(tl!(
            "OSOR setup needs about 19.1 GB of free disk space and 40 GiB of memory to prepare the weights. Installed OSOR weights use about 10.6 GB."
        ));
        ui.horizontal(|ui| {
            let label = ui.label(tl!("PhotoCraft memory allowance (MiB):"));
            let mut budget = f.get("memoryUsageMb").and_then(Value::as_u64).unwrap_or(FULL_MEMORY_MIB);
            ui.add(egui::DragValue::new(&mut budget).range(FULL_MEMORY_MIB..=1_048_576).speed(1024)).labelled_by(label.id);
            f.insert("memoryUsageMb".into(), json!(budget));
        });
        ui.label(RichText::new(tl!("Confirming this setup also saves this memory allowance in Preferences.")).color(t.text_dim));
        if let Some(ram) = crate::gpu_canvas::physical_memory() {
            ui.label(format!("{}: {:.0} GiB", tl!("Computer memory"), ram as f64 / (1_u64 << 30) as f64));
        }
    }
    if !missing(&status, &selected).is_empty() {
        ui.separator();
        ui.label(tl!("Download sources and licenses:"));
        for id in components(&selected) {
            if let Some(entry) = model(&status, id) {
                let label = entry.get("label").and_then(Value::as_str).unwrap_or(id);
                let url = entry
                    .get("licenseUrl")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .or_else(|| entry.get("source").and_then(Value::as_str))
                    .unwrap_or("");
                let license = entry.get("license").and_then(Value::as_str).unwrap_or("");
                ui.hyperlink_to(format!("{label}: {license}"), url);
            }
        }
        if selected == "recommended" {
            ui.label(tl!("OSOR weights use CreativeML Open RAIL++-M, which includes use restrictions."));
        }
        let mut accepted = f.get("acceptLicenses").and_then(Value::as_bool).unwrap_or(false);
        crate::widgets::checkbox(ui, &mut accepted, tl!("I have read and accept the model licenses."));
        f.insert("acceptLicenses".into(), json!(accepted));
    }
    if let Err(reason) = validation(app, f, &status) {
        ui.add_space(6.0);
        ui.label(RichText::new(reason).color(t.text_dim));
    }
    f.insert("__download".into(), json!(!missing(&status, &selected).is_empty()));
}

pub(crate) fn confirm_label(f: &Map<String, Value>) -> &'static str {
    if f.get("__download").and_then(Value::as_bool) == Some(true) { tl!("Download and use") } else { tl!("Use this setup") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct Ready;
    impl photocraft_ml::RemovalBackend for Ready {
        fn ready(&self, _: &str) -> bool {
            true
        }
        fn infer(
            &self,
            _: &str,
            _: &photocraft_ml::RemovalInput,
            _: &dyn photocraft_ml::RemovalControl,
        ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
            Err(photocraft_ml::Error::Inference("setup must never infer".into()))
        }
    }

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.set_removal_service(Arc::new(photocraft_ml::RemovalService::new(None).with_backend(Arc::new(Ready))));
        app
    }

    #[test]
    fn hybrid_setup_choices_save_without_inference_or_unrelated_memory_changes() {
        for (workflow, enabled, ids) in [("lama", true, vec!["lama"]), ("compact", true, vec!["migan-v1"]), ("classical", false, vec![])] {
            let mut app = app();
            let budget = app.session.prefs().performance.memory_usage_mb;
            let fields = json!({"workflow":workflow}).as_object().unwrap().clone();
            confirm(&mut app, &fields).unwrap();
            assert_eq!(app.session.prefs().generative_remove.workflow, workflow);
            assert!(app.session.prefs().generative_remove.setup_complete);
            assert_eq!(app.session.prefs().generative_remove.enabled, enabled);
            assert_eq!(app.session.prefs().generative_remove.enabled_models, ids);
            assert_eq!(app.session.prefs().performance.memory_usage_mb, budget);
            assert_eq!(app.ui.tool_options.remove_method, "auto");
            assert!(!busy(&app));
            assert!(!app.session.has_jobs());
        }
    }

    #[test]
    fn hybrid_setup_requires_license_consent_only_when_weights_are_missing() {
        let app = app();
        let status = json!({"managementAvailable":true,"models":[{"id":"lama","available":true,"canDownload":true,"installed":false}]});
        let mut fields = json!({"workflow":"lama"}).as_object().unwrap().clone();
        assert!(validation(&app, &fields, &status).unwrap_err().contains("licenses"));
        fields.insert("acceptLicenses".into(), json!(true));
        assert!(validation(&app, &fields, &status).is_ok());
        fields.insert("workflow".into(), json!("classical"));
        fields.insert("acceptLicenses".into(), json!(false));
        assert!(validation(&app, &fields, &status).is_ok());
    }

    #[test]
    fn hybrid_setup_unsupported_full_profile_can_be_replaced_with_classical() {
        let app = app();
        let status = json!({"managementAvailable":true,"models":[{"id":"lama","available":true,"installed":true},{"id":"osor-sdxl","available":false}]});
        let mut fields = json!({"workflow":"recommended","memoryUsageMb":49152,"acceptLicenses":true}).as_object().unwrap().clone();
        assert!(validation(&app, &fields, &status).is_err());
        fields.insert("workflow".into(), json!("classical"));
        assert!(validation(&app, &fields, &status).is_ok());
    }

    #[test]
    fn hybrid_setup_denied_configuration_has_no_queued_side_effects() {
        let mut app = app();
        let before = app.session.prefs_value();
        app.session.authorize =
            Some(|command, _| if command == "remove.setup" { Err(photocraft_engine::EngineError::Other("setup denied".into())) } else { Ok(()) });
        assert!(confirm(&mut app, json!({"workflow":"lama"}).as_object().unwrap()).unwrap_err().contains("setup denied"));
        assert_eq!(app.session.prefs_value(), before);
        assert!(!busy(&app));
        assert!(!app.session.has_jobs());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hybrid_setup_failed_job_does_not_start_the_next_download() {
        let mut app = app();
        let started = app
            .session
            .start_job(
                "ai.models.install",
                json!({"modelId":"lama"}),
                "Download model",
                false,
                |_| -> photocraft_engine::Result<()> { Err(photocraft_engine::EngineError::Other("verified download failed".into())) },
                |_, ()| Ok(json!({})),
            )
            .unwrap();
        let photocraft_engine::jobs::Started::Job(id) = started else {
            panic!("native job");
        };
        app.prefs_rt.remove_setup.active = Some(id);
        app.prefs_rt.remove_setup.queue.push_back("osor-sdxl".into());
        assert!(app.session.wait_job(id).is_err());
        tick(&mut app, &egui::Context::default());
        assert!(!busy(&app));
        assert_eq!(app.session.jobs_with_recent().len(), 1);
        assert!(app.prefs_rt.remove_setup.error);
        assert!(app.prefs_rt.remove_setup.message.contains("verified download failed"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hybrid_setup_cancelled_job_stops_the_remaining_queue_even_when_dialog_is_closed() {
        let mut app = app();
        let (done, wait) = std::sync::mpsc::channel();
        let started = app
            .session
            .start_job(
                "ai.models.install",
                json!({"modelId":"lama"}),
                "Download model",
                false,
                move |_| {
                    wait.recv().map_err(|error| photocraft_engine::EngineError::Other(error.to_string()))?;
                    Ok(())
                },
                |_, ()| Ok(json!({})),
            )
            .unwrap();
        let photocraft_engine::jobs::Started::Job(id) = started else {
            panic!("native job");
        };
        app.prefs_rt.remove_setup.active = Some(id);
        app.prefs_rt.remove_setup.queue.push_back("osor-sdxl".into());
        let dialog = open(&mut app);
        crate::dialogs::cancel(&mut app, dialog).unwrap();
        assert!(busy(&app), "closing the setup must not lose its authorized queue");
        assert!(app.session.cancel_job(id));
        tick(&mut app, &egui::Context::default());
        assert!(!busy(&app));
        assert!(app.session.jobs().iter().all(|job| job.command != "ai.models.install"));
        assert_eq!(app.session.jobs_with_recent().len(), 1);
        done.send(()).unwrap();
        app.session.join_cancelled_jobs();
    }

    #[test]
    fn hybrid_setup_cancel_does_not_nag_again_in_this_session() {
        let mut app = app();
        app.ui.tool = crate::state::Tool::Remove;
        let dialog = open(&mut app);
        crate::dialogs::cancel(&mut app, dialog).unwrap();
        tick(&mut app, &egui::Context::default());
        assert!(app.ui.dialogs.is_empty());
        assert!(!app.session.prefs().generative_remove.setup_complete, "Cancel preserves preferences");
    }

    #[test]
    fn hybrid_setup_small_window_keeps_confirmation_reachable() {
        use egui_kittest::kittest::Queryable;
        let mut harness = egui_kittest::Harness::builder().with_size(egui::vec2(650.0, 600.0)).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            open(&mut app);
            app
        });
        harness.run_steps(4);
        let button = harness.get_by_label("Download and use").rect();
        assert!(button.top() >= 0.0 && button.bottom() <= 600.0, "setup confirmation must remain on screen: {button:?}");
    }

    #[test]
    #[ignore = "offscreen visual evidence; set PHOTOCRAFT_SETUP_SNAPSHOT to a Scratch PNG path"]
    fn hybrid_setup_offscreen_snapshot() {
        let _gpu = photocraft_testkit::gpu_lock();
        let path = std::env::var("PHOTOCRAFT_SETUP_SNAPSHOT").unwrap();
        let size = if std::env::var_os("PHOTOCRAFT_SETUP_SMALL_SNAPSHOT").is_some() { egui::vec2(650.0, 600.0) } else { egui::vec2(1100.0, 850.0) };
        let mut harness = egui_kittest::Harness::builder().with_size(size).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            open(&mut app);
            app
        });
        harness.run_steps(4);
        harness.render().unwrap().save(path).unwrap();
    }
}
