//! Local Remove setup, explicit downloads, and advanced manual model choices.

use super::{DialogKind, Map, PhotocraftApp, RichText, Tokens, Value, json};
use crate::file_dialog::{FileDialogAnswer, FileDialogRequest};

/// Opening the manager only reports state; configuration and downloads require an action.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    if let Some(dialog) = app.ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("removeModels")) {
        return dialog.id;
    }
    let fields = json!({"__prefsui":"removeModels","__label":"Generative Remove Models","message":"","advanced":false});
    app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default())
}

fn flag(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn model<'a>(status: &'a Value, id: &str) -> Option<&'a Value> {
    status.get("models")?.as_array()?.iter().find(|model| text(model, "id") == id)
}

fn download_bytes(model: &Value) -> u64 {
    model.get("downloadBytes").or_else(|| model.get("bytes")).and_then(Value::as_u64).unwrap_or(0)
}

fn size_label(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.1} GB", bytes as f64 / 1_000_000_000.0) } else { format!("{:.0} MB", bytes as f64 / 1_000_000.0) }
}

fn profile_for_model(id: &str) -> &'static str {
    match id {
        "lama" => "lama",
        "migan-v1" => "compact",
        _ => "recommended",
    }
}

fn remember_result(f: &mut Map<String, Value>, result: Result<Value, String>) {
    f.remove("job");
    f.insert("stopped".into(), json!(false));
    match result {
        Ok(result) => {
            let message = if let Some(job) = result.get("job").and_then(Value::as_u64) {
                f.insert("job".into(), json!(job));
                tl!("The model operation continues while this window is closed.")
            } else if flag(&result, "restartRequired") {
                tl!("Restart PhotoCraft to use the selected model folder.")
            } else if flag(&result, "modelFolderPicker") {
                tl!("Choose a folder containing your installed models.")
            } else if result.get("fileDialog").and_then(Value::as_str) == Some("folder") {
                tl!("Choose a local model bundle.")
            } else if result.get("fileDialog").is_some() {
                tl!("Choose a model file to install.")
            } else {
                tl!("Settings take effect immediately.")
            };
            f.insert("message".into(), json!(message));
            f.insert("error".into(), json!(false));
        }
        Err(error) => {
            f.insert("message".into(), json!(error));
            f.insert("error".into(), json!(true));
        }
    }
}

/// Check every permission before changing any part of the manual configuration.
fn configure(app: &mut PhotocraftApp, commands: Vec<(&str, Value)>) -> Result<Value, String> {
    for (command, params) in &commands {
        if let Some(gate) = app.session.authorize {
            gate(command, params).map_err(|error| error.to_string())?;
        }
        if app.automation_input
            && let Some(gate) = app.services.automation_command.as_ref()
        {
            gate(command, params)?;
        }
    }
    let mut result = Value::Null;
    for (command, params) in commands {
        result = app.run(command, params)?;
    }
    Ok(result)
}

fn enabled_model(app: &mut PhotocraftApp, id: &str, enabled: bool) -> Result<Value, String> {
    let mut ids = app.session.prefs().generative_remove.enabled_models.clone();
    ids.retain(|value| value != id);
    if enabled {
        ids.push(id.to_string());
    }
    configure(app, vec![("remove.setup", json!({"workflow":"custom"})), ("prefs.set", json!({"path":"generativeRemove.enabledModels","value":ids}))])
}

fn select_model(app: &mut PhotocraftApp, id: &str) -> Result<Value, String> {
    if !app.session.removal_service().catalog().iter().any(|model| model.id == id) {
        return Err(tl!("Choose an available removal model.").into());
    }
    let mut ids = app.session.prefs().generative_remove.enabled_models.clone();
    if !ids.iter().any(|value| value == id) {
        ids.push(id.to_string());
    }
    let result = configure(
        app,
        vec![
            ("remove.setup", json!({"workflow":"custom","modelId":id})),
            ("prefs.set", json!({"path":"generativeRemove.enabled","value":true})),
            ("prefs.set", json!({"path":"generativeRemove.enabledModels","value":ids})),
            ("prefs.set", json!({"path":"generativeRemove.selectedModel","value":id})),
        ],
    )?;
    // Custom Auto intentionally preserves the legacy Classical fallback. A manual model action
    // is an explicit choice to use AI, and must update the tool's method as well.
    app.ui.tool_options.remove_method = "ai".into();
    Ok(result)
}

// Transfers run in the background even for a host that normally runs commands inline.
pub(super) fn model_command(app: &mut PhotocraftApp, id: &str, params: Value) -> Result<Value, String> {
    let background = app.background_jobs;
    app.background_jobs = true;
    let result = app.run(id, params);
    app.background_jobs = background;
    result
}

/// Ask only for the path here; validation and copying happen on the engine worker.
fn import_model(app: &mut PhotocraftApp, id: &str) -> Result<Value, String> {
    let id = id.to_string();
    let request = if id == "osor-sdxl" {
        FileDialogRequest::Folder { initial_dir: None }
    } else {
        FileDialogRequest::Open { multiple: false, initial_dir: None, extensions: Some(vec!["onnx".into()]) }
    };
    app.ask_file(request, move |app, answer| {
        let result = match answer {
            FileDialogAnswer::Paths(paths) => match paths.first() {
                Some(path) => model_command(app, "ai.models.import", json!({"modelId":id,"path":path})),
                None => Err(tl!("No model file was selected").into()),
            },
            _ => Err(tl!("Local model installation requires a desktop file path").into()),
        };
        if let Some(dialog) = app.ui.dialogs.iter_mut().find(|dialog| dialog.fields.get("__prefsui").and_then(Value::as_str) == Some("removeModels")) {
            remember_result(&mut dialog.fields, result.clone());
        }
        result
    })
}

pub(super) fn model_folder_pending(app: &PhotocraftApp) -> bool {
    app.session.prefs().generative_remove.model_directory != app.prefs_rt.model_directory_at_launch
}

fn set_model_folder(app: &mut PhotocraftApp, folder: &str) -> Result<Value, String> {
    configure(app, vec![("prefs.set", json!({"path":"generativeRemove.modelDirectory","value":folder}))])?;
    Ok(json!({"restartRequired":model_folder_pending(app)}))
}

fn choose_model_folder(app: &mut PhotocraftApp) -> Result<Value, String> {
    // A native folder picker requires an interactive user; remote clients cannot use a
    // delayed dialog reply to bypass their filesystem preference authorization.
    if app.automation_input || app.session.authorize.is_some() {
        return Err(tl!("Choose the model folder locally in PhotoCraft.").into());
    }
    let previous = app.session.prefs().generative_remove.model_directory.clone();
    let initial_dir = app.session.removal_service().model_directory().map(|path| path.to_string_lossy().into_owned());
    app.ask_file(FileDialogRequest::Folder { initial_dir }, move |app, answer| {
        if app.session.prefs().generative_remove.model_directory != previous {
            return Err(tl!("The model folder changed while the picker was open. Choose it again.").into());
        }
        match answer {
            FileDialogAnswer::FolderTo(path) => set_model_folder(app, &path),
            _ => Err(tl!("Choose a model folder on this computer.").into()),
        }
    })?;
    Ok(json!({"fileDialog":"folder","modelFolderPicker":true}))
}

fn render_model_folder(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, busy: bool) {
    if cfg!(target_arch = "wasm32") || app.services.file_dialog.is_none() {
        return;
    }
    ui.separator();
    ui.label(RichText::new(tl!("Model folder")).strong());
    let folder = app.session.prefs().generative_remove.model_directory.clone();
    ui.label(if folder.is_empty() { tl!("Automatic (default folder)") } else { folder.as_str() });
    if let Some(active) = status.get("modelDirectory").and_then(Value::as_str)
        && (folder.is_empty() || folder != active)
    {
        ui.label(crate::i18n::fmt(tl!("In use: {folder}"), &[("folder", active)]));
    }
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(!busy, egui::Button::new(tl!("Choose model folder…"))).clicked() {
            let result = choose_model_folder(app);
            remember_result(f, result);
        }
        if ui.add_enabled(!busy && !folder.is_empty(), egui::Button::new(tl!("Use default folder"))).clicked() {
            let result = set_model_folder(app, "");
            remember_result(f, result);
        }
    });
    if model_folder_pending(app) {
        ui.label(tl!("Restart PhotoCraft to use the selected model folder."));
    }
    ui.label(tl!("Existing model files stay in their current folders. Choose a folder containing your installed models to reuse them."));
}

pub(super) fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let status = match app.session.execute("ai.models.list", json!({})) {
        Ok(status) => status,
        Err(error) => {
            ui.label(RichText::new(error.to_string()).color(Tokens::get(ui.ctx()).text_dim));
            return;
        }
    };
    render(app, ui, f, &status);
}

struct CurrentSetup {
    label: String,
    ready: bool,
    available: bool,
    installed: usize,
    total: usize,
    remaining_bytes: u64,
    proposal: String,
    disabled: bool,
    attention: Option<String>,
    configuration_reason: Option<String>,
    memory_warning: Option<&'static str>,
}

fn installed_setup(app: &PhotocraftApp, status: &Value) -> CurrentSetup {
    let prefs = &app.session.prefs().generative_remove;
    if prefs.workflow != "custom" {
        let summary = super::remove_setup::profile_summary(status, &prefs.workflow);
        let configured = super::remove_setup::current_profile_ready(status, &prefs.workflow);
        let disabled = prefs.workflow != "classical"
            && (!prefs.enabled
                || !super::remove_setup::components(&prefs.workflow).iter().all(|required| prefs.enabled_models.iter().any(|id| id == required)));
        return CurrentSetup {
            label: super::remove_setup::workflow_label(&prefs.workflow).into(),
            ready: configured && !disabled,
            available: summary.available,
            installed: summary.installed,
            total: summary.total,
            remaining_bytes: summary.remaining_bytes,
            proposal: prefs.workflow.clone(),
            disabled,
            attention: status
                .get("models")
                .and_then(Value::as_array)
                .and_then(|models| models.iter().find(|entry| asset_attention(entry) && used_by_current_setup(app, text(entry, "id"))))
                .map(|entry| text(entry, "id").into()),
            configuration_reason: (summary.ready() && !configured)
                .then(|| shared_configuration_reason(status, super::remove_setup::components(&prefs.workflow))),
            memory_warning: None,
        };
    }
    let selected = model(status, &prefs.selected_model);
    let installed = selected.is_some_and(|model| flag(model, "installed"));
    let available = selected.is_some_and(|model| flag(model, "available"));
    let disabled = !prefs.enabled || !prefs.enabled_models.iter().any(|id| id == &prefs.selected_model);
    let configured = super::remove_setup::configured_model_ready(status, &prefs.selected_model);
    let selected_label = selected.map(|model| text(model, "label")).filter(|label| !label.is_empty()).unwrap_or(&prefs.selected_model);
    CurrentSetup {
        label: format!("{}: {selected_label}", super::remove_setup::workflow_label("custom")),
        ready: configured && !disabled,
        available,
        installed: usize::from(installed),
        total: 1,
        remaining_bytes: selected.filter(|_| !installed).map(download_bytes).unwrap_or(0),
        proposal: profile_for_model(&prefs.selected_model).into(),
        disabled,
        attention: selected.filter(|entry| asset_attention(entry)).map(|_| prefs.selected_model.clone()),
        configuration_reason: (installed && available && !configured).then(|| shared_configuration_reason(status, &[&prefs.selected_model])),
        memory_warning: None,
    }
}

fn shared_configuration_reason(status: &Value, ids: &[&str]) -> String {
    ids.iter()
        .filter_map(|id| model(status, id))
        .find_map(|entry| entry.get("configuredReason").and_then(Value::as_str).filter(|reason| !reason.is_empty()))
        .unwrap_or(tl!("The installed models need a shared AI setup choice."))
        .to_owned()
}

fn current_setup(app: &PhotocraftApp, status: &Value) -> CurrentSetup {
    current_setup_with_memory(app, status, crate::gpu_canvas::physical_memory())
}

fn current_setup_with_memory(app: &PhotocraftApp, status: &Value, physical_memory: Option<u64>) -> CurrentSetup {
    let mut current = installed_setup(app, status);
    if !current.ready || app.session.prefs().generative_remove.workflow == "classical" {
        return current;
    }
    let prefs = &app.session.prefs().generative_remove;
    let full = prefs.workflow == "recommended" || (prefs.workflow == "custom" && prefs.selected_model == "osor-sdxl");
    let selected = if prefs.workflow == "compact" {
        "migan-v1"
    } else if prefs.workflow == "lama" {
        "lama"
    } else {
        &prefs.selected_model
    };
    let required = model(status, selected).and_then(|entry| entry.get("memoryMib")).and_then(Value::as_u64).unwrap_or(0);
    let allowance = u64::from(app.session.prefs().performance.memory_usage_mb);
    current.memory_warning = if full && physical_memory.is_some_and(|bytes| bytes < 49_152_u64.saturating_mul(1 << 20)) {
        Some(tl!("This computer needs more memory for OSOR. Choose a smaller setup."))
    } else if (full && allowance < 49_152) || (!full && required > 0 && allowance <= required) {
        Some(tl!("The memory allowance is too low for this setup. Choose a smaller setup or increase Memory Usage in Preferences > Performance."))
    } else {
        None
    };
    current.ready = current.memory_warning.is_none();
    current
}

fn asset_attention(entry: &Value) -> bool {
    flag(entry, "installed") && flag(entry, "available") && entry.get("canAttempt").and_then(Value::as_bool) == Some(false)
}

fn manage_files(f: &mut Map<String, Value>, id: &str) {
    f.insert("advanced".into(), json!(true));
    f.insert("focusModelId".into(), json!(id));
}

fn refresh_feedback(app: &PhotocraftApp, f: &mut Map<String, Value>, status: &Value) {
    let jobs = app.session.jobs_with_recent();
    let running = jobs.iter().filter(|job| job.command.starts_with("ai.models.") && job.state == "running").max_by_key(|job| job.id.0);
    let tracked = f.get("job").and_then(Value::as_u64).and_then(|id| jobs.iter().find(|job| job.id.0 == id));
    let recent = || jobs.iter().filter(|job| job.command.starts_with("ai.models.")).max_by_key(|job| job.id.0);
    let message_empty = f.get("message").and_then(Value::as_str).unwrap_or("").is_empty();
    let job = running.or(tracked).or_else(|| if message_empty { recent() } else { None });
    let Some(job) = job else { return };
    f.insert("job".into(), json!(job.id.0));
    let draining =
        job.state == "cancelled" && status.get("models").and_then(Value::as_array).is_some_and(|models| models.iter().any(|model| flag(model, "busy")));
    let message = match job.state {
        "running" => "".to_string(),
        "failed" => job.error.clone().unwrap_or_else(|| tl!("Model setup could not finish.").into()),
        "cancelled" if draining => tl!("Cancellation requested. Waiting for the current step to stop.").into(),
        "cancelled" => tl!("Model installation cancelled. Installed models were kept.").into(),
        "done" if job.command == "ai.models.remove" => tl!("Model removed. Finish setup to install it again.").into(),
        "done" => tl!("Model installed and verified").into(),
        _ => return,
    };
    f.insert("message".into(), json!(message));
    f.insert("error".into(), json!(job.state == "failed"));
    f.insert("stopped".into(), json!(matches!(job.state, "failed" | "cancelled")));
    f.insert("stopping".into(), json!(draining));
}

fn render_summary(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, blocked: bool) {
    let t = Tokens::get(ui.ctx());
    let current = current_setup(app, status);
    ui.label(RichText::new(format!("{}: {}", tl!("Current setup"), current.label)).strong());
    if app.session.prefs().generative_remove.workflow == "custom" && app.ui.tool_options.remove_method == "auto" {
        ui.label(RichText::new(tl!("Auto currently uses Classical. Choose a setup, or choose a manual AI model in Advanced.")).color(t.text_dim));
    } else if app.session.prefs().generative_remove.workflow != "classical" && app.ui.tool_options.remove_method == "classical" {
        ui.label(RichText::new(tl!("The Remove method is Classical. Choose a setup to switch back to Auto.")).color(t.text_dim));
    }
    if current.ready {
        ui.label(tl!("Ready"));
    } else if let Some(id) = current.attention.as_deref() {
        ui.label(tl!("Needs attention"));
        if let Some(entry) = model(status, id) {
            ui.label(RichText::new(tl!(text(entry, "reason"))).color(t.danger));
        }
        if ui.add_enabled(!blocked, egui::Button::new(tl!("Manage model files…"))).clicked() {
            manage_files(f, id);
        }
    } else if let Some(warning) = current.memory_warning {
        ui.label(tl!("Needs attention"));
        ui.label(RichText::new(warning).color(t.text_dim));
        if ui.add_enabled(!blocked, egui::Button::new(tl!("Review setup…"))).clicked() {
            super::remove_setup::open_profile(app, &current.proposal);
        }
    } else {
        ui.label(tl!("Setup incomplete"));
        if current.disabled {
            ui.label(RichText::new(tl!("Local AI is off or a model required by this setup is disabled.")).color(t.text_dim));
        } else if !current.available {
            ui.label(RichText::new(tl!("This setup is unavailable in this build. Choose another setup.")).color(t.text_dim));
        } else if let Some(reason) = current.configuration_reason.as_deref() {
            ui.label(RichText::new(tl!(reason)).color(t.text_dim));
            ui.label(tl!("Finish setup to confirm the shared AI choice and reuse installed models. No download is needed."));
        } else {
            ui.label(RichText::new(format!("{}: {}/{}", tl!("Models installed"), current.installed, current.total)).color(t.text_dim));
        }
        let transferring = super::remove_setup::busy(app) || app.session.jobs().iter().any(|job| job.command.starts_with("ai.models."));
        if current.remaining_bytes > 0 && !transferring {
            ui.label(RichText::new(crate::i18n::fmt(tl!("{size} download remaining"), &[("size", &size_label(current.remaining_bytes))])).color(t.text_dim));
        }
    }
    ui.horizontal_wrapped(|ui| {
        if !current.ready
            && current.attention.is_none()
            && current.memory_warning.is_none()
            && ui.add_enabled(!blocked && current.available, egui::Button::new(tl!("Finish setup…"))).clicked()
        {
            super::remove_setup::open_profile(app, &current.proposal);
        }
        if ui.add_enabled(!blocked, egui::Button::new(tl!("Change setup…"))).clicked() {
            super::remove_setup::open(app);
        }
    });
}

fn render_progress(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &Map<String, Value>, model_busy: bool) {
    let jobs: Vec<_> = app.session.jobs().into_iter().filter(|job| job.command.starts_with("ai.models.")).collect();
    for job in &jobs {
        ui.horizontal_wrapped(|ui| {
            ui.label(tl!(&job.label));
            ui.add(egui::ProgressBar::new(job.progress).desired_width(150.0).show_percentage());
            if ui.button(tl!("Cancel model operation")).clicked() {
                crate::jobs_ui::cancel(app, job.id);
            }
        });
        if !job.message.is_empty() {
            ui.label(tl!(&job.message));
        }
    }
    if !jobs.is_empty() {
        ui.label(tl!("You can close this window and keep working while setup continues."));
    } else if f.get("stopping").and_then(Value::as_bool) == Some(true) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(tl!("Stopping model setup…"));
        });
    } else if super::remove_setup::busy(app) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(tl!("Starting model setup…"));
        });
    } else if model_busy {
        ui.label(tl!("Finish or cancel the current removal before changing models."));
    }
    if !jobs.is_empty() || model_busy || super::remove_setup::busy(app) {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
}

fn render_feedback(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &Map<String, Value>, blocked: bool) {
    let t = Tokens::get(ui.ctx());
    let setup_message = &app.prefs_rt.remove_setup.message;
    if !setup_message.is_empty() {
        ui.label(RichText::new(setup_message).color(if app.prefs_rt.remove_setup.error { t.danger } else { t.text_dim }));
    }
    if let Some(message) = f.get("message").and_then(Value::as_str).filter(|message| !message.is_empty() && *message != setup_message.as_str()) {
        ui.label(RichText::new(message).color(if f.get("error").and_then(Value::as_bool) == Some(true) { t.danger } else { t.text_dim }));
    }
    if f.get("stopped").and_then(Value::as_bool) == Some(true) && ui.add_enabled(!blocked, egui::Button::new(tl!("Review setup…"))).clicked() {
        super::remove_setup::open(app);
    }
}

fn used_by_current_setup(app: &PhotocraftApp, id: &str) -> bool {
    let prefs = &app.session.prefs().generative_remove;
    match prefs.workflow.as_str() {
        "recommended" => matches!(id, "lama" | "osor-sdxl"),
        "lama" => id == "lama",
        "compact" => id == "migan-v1",
        "classical" => false,
        _ => prefs.enabled && prefs.selected_model == id,
    }
}

fn render_remove_confirmation(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, blocked: bool) {
    let Some(id) = f.get("removeModelId").and_then(Value::as_str).map(str::to_owned) else { return };
    let Some(entry) = model(status, &id) else {
        f.remove("removeModelId");
        return;
    };
    ui.group(|ui| {
        ui.label(RichText::new(crate::i18n::fmt(tl!("Remove {model}?"), &[("model", text(entry, "label"))])).strong());
        ui.label(tl!("PhotoCraft-managed model files will be deleted. Imported originals are kept."));
        if used_by_current_setup(app, &id) {
            ui.label(tl!("The current setup will become incomplete. Download or import this model again before using that setup."));
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(tl!("Keep model")).clicked() {
                f.remove("removeModelId");
            }
            if ui.add_enabled(!blocked && flag(status, "managementAvailable"), egui::Button::new(tl!("Remove model"))).clicked() {
                f.remove("removeModelId");
                let result = model_command(app, "ai.models.remove", json!({"modelId":id}));
                remember_result(f, result);
            }
        });
    });
}

fn render_card(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, entry: &Value, management: bool, blocked: bool) {
    let t = Tokens::get(ui.ctx());
    let id = text(entry, "id");
    let installed = flag(entry, "installed");
    ui.push_id(id, |ui| {
        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(text(entry, "label")).strong());
                ui.label(RichText::new(if installed { tl!("Installed") } else { tl!("Not installed") }).color(t.text_dim));
                if !installed {
                    ui.label(RichText::new(size_label(download_bytes(entry))).color(t.text_dim));
                }
            });
            ui.label(tl!(text(entry, "description")));
            let reason = text(entry, "reason");
            let duplicate = reason == "model is not installed"
                || (!flag(entry, "available")
                    && matches!(reason, "native AI inference is unavailable in this build" | "AI removal is unavailable in the browser"));
            if !reason.is_empty() && !duplicate {
                ui.label(RichText::new(tl!(reason)).color(t.text_dim));
            }
            if asset_attention(entry) {
                if ui.add_enabled(!blocked, egui::Button::new(tl!("Manage model files…"))).clicked() {
                    manage_files(f, id);
                }
            } else {
                let allowed = !blocked && flag(entry, "available") && (installed || (management && flag(entry, "canDownload")));
                let label = if installed { tl!("Use this setup…") } else { tl!("Download and use…") };
                if ui.add_enabled(allowed, egui::Button::new(label)).clicked() {
                    super::remove_setup::open_profile(app, profile_for_model(id));
                }
            }
        });
    });
}

fn render_details(ui: &mut egui::Ui, entry: &Value) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(text(entry, "license")).color(t.text_dim));
        if !text(entry, "source").is_empty() {
            ui.hyperlink_to(tl!("Model source and license"), text(entry, "source"));
        }
        if !text(entry, "url").is_empty() {
            ui.hyperlink_to(tl!("Model download source"), text(entry, "url"));
        }
    });
}

fn render_manual_model(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, entry: &Value, management: bool, blocked: bool) {
    let id = text(entry, "id");
    let prefs = app.session.prefs().generative_remove.clone();
    let mut enabled = prefs.enabled_models.iter().any(|value| value == id);
    let response = ui.push_id(("manual", id), |ui| {
        ui.group(|ui| {
            ui.label(RichText::new(text(entry, "label")).strong());
            ui.add_enabled_ui(!blocked && flag(entry, "available"), |ui| {
                if crate::widgets::checkbox(ui, &mut enabled, tl!("Allow this model")).changed() {
                    let result = enabled_model(app, id, enabled);
                    remember_result(f, result);
                }
                if ui.button(tl!("Use this model")).clicked() {
                    let result = select_model(app, id);
                    let success = result.is_ok();
                    remember_result(f, result);
                    if success {
                        let message = if flag(entry, "installed") {
                            tl!("Model selected. Remove now uses AI.")
                        } else {
                            tl!("Model selected. Download its setup before using Remove.")
                        };
                        f.insert("message".into(), json!(message));
                    }
                }
            });
            if prefs.workflow == "custom" && prefs.selected_model == id && prefs.enabled && app.ui.tool_options.remove_method == "ai" {
                ui.label(tl!("Manual AI model"));
            }
            render_details(ui, entry);
            ui.add_enabled_ui(!blocked && management && flag(entry, "available"), |ui| {
                ui.horizontal_wrapped(|ui| {
                    let label = if id == "osor-sdxl" { tl!("Install from folder…") } else { tl!("Install from file…") };
                    if ui.add_enabled(app.services.file_dialog.is_some(), egui::Button::new(label)).clicked() {
                        let result = import_model(app, id);
                        remember_result(f, result);
                    }
                    if flag(entry, "installed") && ui.button(tl!("Remove installed model…")).clicked() {
                        f.insert("removeModelId".into(), json!(id));
                    }
                });
            });
        });
    });
    if f.get("focusModelId").and_then(Value::as_str) == Some(id) {
        response.response.scroll_to_me(Some(egui::Align::TOP));
        f.remove("focusModelId");
    }
}

fn render_advanced(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, blocked: bool) {
    let expanded = f.get("advanced").and_then(Value::as_bool).unwrap_or(false);
    let response = egui::CollapsingHeader::new(tl!("Advanced")).id_salt("remove-models-advanced").open(Some(expanded)).show(ui, |ui| {
        ui.label(tl!("Manual controls use Custom setup. Use this model enables local AI and switches Remove to AI; it does not download files."));
        ui.label(tl!("Allowing a model alone keeps the current method. Auto in Custom uses Classical."));
        let mut enabled = app.session.prefs().generative_remove.enabled;
        ui.add_enabled_ui(!blocked && flag(status, "runtimeAvailable"), |ui| {
            if crate::widgets::checkbox(ui, &mut enabled, tl!("Enable local AI removal")).changed() {
                let result = configure(
                    app,
                    vec![("remove.setup", json!({"workflow":"custom"})), ("prefs.set", json!({"path":"generativeRemove.enabled","value":enabled}))],
                );
                remember_result(f, result);
            }
        });
        if let Some(models) = status.get("models").and_then(Value::as_array) {
            for entry in models {
                render_manual_model(app, ui, f, entry, flag(status, "managementAvailable"), blocked);
                ui.add_space(6.0);
            }
        }
    });
    if response.header_response.clicked() {
        f.insert("advanced".into(), json!(!expanded));
        ui.ctx().request_repaint();
    }
}

fn render_execution(ui: &mut egui::Ui, status: &Value) {
    let Some(info) = status.get("lastExecution").filter(|value| value.is_object()) else { return };
    let id = text(info, "modelId");
    let model = model(status, id).map(|entry| text(entry, "label")).unwrap_or(id);
    let template = if flag(info, "usesGpu") {
        if info.get("graph").and_then(|graph| graph.get("cpuNodes")).and_then(Value::as_u64).is_some_and(|count| count > 0) {
            tl!("Last run: {model} used GPU and CPU operations.")
        } else {
            tl!("Last run: {model} used the GPU.")
        }
    } else {
        tl!("Last run: {model} used the CPU.")
    };
    let label = crate::i18n::fmt(template, &[("model", model)]);
    ui.label(label).on_hover_text(text(info, "device"));
    if let Some(reason) = info.get("fallbackReason").and_then(Value::as_str) {
        ui.label(tl!("GPU unavailable for this run; used the same model on CPU.")).on_hover_text(reason);
    }
}

fn render(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value) {
    let top = ui.cursor().top();
    refresh_feedback(app, f, status);
    let models_busy = status.get("models").and_then(Value::as_array).is_some_and(|models| models.iter().any(|model| flag(model, "busy")));
    let busy = models_busy || super::remove_setup::busy(app) || app.session.jobs().iter().any(|job| job.command.starts_with("ai.models."));
    let blocked = busy || model_folder_pending(app);
    render_model_folder(app, ui, f, status, busy);
    render_summary(app, ui, f, status, blocked);
    render_progress(app, ui, f, models_busy);
    render_feedback(app, ui, f, blocked);
    render_execution(ui, status);
    render_remove_confirmation(app, ui, f, status, blocked);
    if !flag(status, "runtimeAvailable") {
        ui.label(tl!("Local AI inference is unavailable in this build. Classical Remove remains available."));
    }
    if !flag(status, "managementAvailable") {
        ui.label(tl!("Model installation is unavailable in this session."));
    }
    let used_height = (ui.cursor().top() - top).max(0.0);
    // Reserve the dialog chrome and the privacy footer so Close stays outside the scrolling list.
    let height = (ui.ctx().content_rect().height() - used_height - 200.0).clamp(40.0, 440.0);
    egui::ScrollArea::vertical().max_height(height).id_salt("remove-models-list").show(ui, |ui| {
        if let Some(models) = status.get("models").and_then(Value::as_array) {
            for entry in models {
                render_card(app, ui, f, entry, flag(status, "managementAvailable"), blocked);
                ui.add_space(6.0);
            }
        }
        render_advanced(app, ui, f, status, blocked);
    });
    ui.label(RichText::new(tl!("Your images stay on this computer. A brush stroke never downloads a model.")).color(Tokens::get(ui.ctx()).text_dim));
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_ux_review_disabled_required_model_is_not_ready() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"lama"})).unwrap();
        state.app.session.edit_prefs(|prefs| prefs.generative_remove.enabled_models.clear());
        state.status["models"][0]["id"] = json!("lama");
        state.status["models"][0]["label"] = json!("LaMa");
        state.status["models"][0]["installed"] = json!(true);
        state.status["models"][0]["canAttempt"] = json!(true);
        state.status["models"][0]["memoryMib"] = json!(3072);
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        assert!(h.query_by_label("Ready").is_none(), "a setup with its required model disabled must not say Ready");
    }

    use super::*;
    use egui::vec2;

    #[test]
    fn opening_remove_models_has_no_download_or_preferences_side_effects() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let before = app.session.prefs_value();
        let id = open(&mut app);
        assert_eq!(open(&mut app), id, "reuse the existing model manager");
        let fields = app.ui.dialogs.iter().find(|dialog| dialog.id == id).unwrap().fields.clone();
        assert_eq!(fields["__prefsui"], "removeModels");
        assert_eq!(app.session.prefs_value(), before);
        assert!(!app.session.has_jobs());
        assert!(app.session.journal.is_empty());
        super::super::confirm(&mut app, &fields).unwrap();
        assert_eq!(app.session.prefs_value(), before, "closing the manager commits no unrelated draft");
    }

    #[test]
    fn model_folder_selection_persists_without_replacing_the_active_runtime() {
        use std::{cell::RefCell, rc::Rc, sync::Arc};
        let replies: Rc<RefCell<Vec<crate::file_dialog::FileDialogReply>>> = Default::default();
        let held = replies.clone();
        let services = crate::Services {
            file_dialog: Some(Box::new(move |request, _, reply| {
                assert!(matches!(request, FileDialogRequest::Folder { .. }));
                held.borrow_mut().push(reply);
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let active = app.session.removal_service();
        let folder = std::env::temp_dir().join("existing-ai-models").to_string_lossy().into_owned();
        let ctx = egui::Context::default();
        choose_model_folder(&mut app).unwrap();
        app.poll_file_dialog(&ctx, None);
        replies.borrow_mut().pop().unwrap().send(None);
        app.poll_file_dialog(&ctx, None);
        assert!(app.session.prefs().generative_remove.model_directory.is_empty());
        assert!(!model_folder_pending(&app));
        choose_model_folder(&mut app).unwrap();
        app.poll_file_dialog(&ctx, None);
        replies.borrow_mut().pop().unwrap().send(Some(FileDialogAnswer::FolderTo(folder.clone())));
        app.poll_file_dialog(&ctx, None);
        assert_eq!(app.session.prefs().generative_remove.model_directory, folder);
        assert!(model_folder_pending(&app));
        assert!(Arc::ptr_eq(&active, &app.session.removal_service()), "folder changes must not replace a running service");
        assert!(!app.session.has_jobs(), "choosing a folder must never download models");
        let saved = app.session.prefs_to_json();
        let restarted = PhotocraftApp::new(
            photocraft_engine::Session::new(),
            crate::Services { load_prefs: Some(Box::new(move || Some(saved.clone()))), ..Default::default() },
        );
        assert_eq!(restarted.session.prefs().generative_remove.model_directory, folder);
        assert!(!model_folder_pending(&restarted));
        // A delayed picker answer must not overwrite a newer preference.
        choose_model_folder(&mut app).unwrap();
        app.poll_file_dialog(&ctx, None);
        set_model_folder(&mut app, "").unwrap();
        replies.borrow_mut().pop().unwrap().send(Some(FileDialogAnswer::FolderTo(folder)));
        app.poll_file_dialog(&ctx, None);
        assert!(app.session.prefs().generative_remove.model_directory.is_empty());
        assert!(app.ui.status.contains("changed while the picker was open"));
        app.automation_input = true;
        assert!(choose_model_folder(&mut app).is_err());
        assert!(!app.file_dialog_open());
    }

    #[test]
    fn process_toggle_preserves_other_enabled_models() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        enabled_model(&mut app, "migan-v1", true).unwrap();
        enabled_model(&mut app, "lama", false).unwrap();
        assert_eq!(app.session.prefs_value().pointer("/generativeRemove/enabledModels"), Some(&json!(["migan-v1"])));
        enabled_model(&mut app, "lama", true).unwrap();
        enabled_model(&mut app, "lama", true).unwrap();
        assert_eq!(app.session.prefs_value().pointer("/generativeRemove/enabledModels"), Some(&json!(["migan-v1", "lama"])));
        assert_eq!(app.session.prefs_value().pointer("/generativeRemove/enabled"), Some(&json!(false)));
    }

    #[test]
    fn import_requires_a_native_path_and_surfaces_chooser_errors_in_the_manager() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let held = Rc::new(RefCell::new(Vec::new()));
        let requests = held.clone();
        let services =
            crate::Services { file_dialog: Some(Box::new(move |request, _, reply| requests.borrow_mut().push((request, reply)))), ..Default::default() };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let id = open(&mut app);
        let ctx = egui::Context::default();
        assert_eq!(import_model(&mut app, "lama").unwrap()["fileDialog"], "open");
        app.poll_file_dialog(&ctx, None);
        let (request, reply) = held.borrow_mut().pop().unwrap();
        assert!(matches!(request, FileDialogRequest::Open { multiple: false, extensions: Some(ext), .. } if ext == ["onnx"]));
        reply.send(Some(FileDialogAnswer::Contents("model.onnx".into(), vec![1, 2, 3])));
        app.poll_file_dialog(&ctx, None);
        let fields = &app.ui.dialogs.iter().find(|dialog| dialog.id == id).unwrap().fields;
        assert_eq!(fields["error"], true);
        assert!(fields["message"].as_str().unwrap().contains("desktop file path"));
        assert!(!app.session.has_jobs(), "browser-style file contents must not start an import");
    }

    struct ModelUi {
        app: PhotocraftApp,
        fields: Map<String, Value>,
        status: Value,
    }

    fn fixture() -> ModelUi {
        ModelUi {
            app: PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default()),
            fields: Map::new(),
            status: json!({"runtimeAvailable":true,"managementAvailable":true,"models":[{
                "id":"migan-v1", "label":"MI-GAN v1", "description":"Compact model", "bytes":29546882,
                "license":"MIT", "source":"https://example.org/source", "url":"https://example.org/model",
                "installed":false, "available":true, "canDownload":true, "busy":false
            }]}),
        }
    }

    #[test]
    fn uninstalled_model_can_be_enabled_selected_and_offered_for_download() {
        use egui_kittest::{
            Harness,
            kittest::{NodeT, Queryable},
        };
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), fixture());
        h.run_steps(3);
        assert!(!h.get_by_label("Download and use…").accesskit_node().is_disabled(), "an absent model can be installed");
        h.get_by_label("Advanced").click();
        h.run_steps(8);
        h.get_by_label("Allow this model").click();
        h.run_steps(3);
        assert_eq!(h.state().app.session.prefs_value().pointer("/generativeRemove/enabledModels"), Some(&json!(["lama", "migan-v1"])));
        h.get_by_label("Use this model").click();
        h.run_steps(3);
        assert_eq!(h.state().app.session.prefs_value().pointer("/generativeRemove/selectedModel"), Some(&json!("migan-v1")));
        assert!(!h.state().app.session.has_jobs(), "selecting a model does not download it");
    }

    #[test]
    fn manual_model_selection_uses_ai_instead_of_classical_auto() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152})).unwrap();
        state.app.ui.tool_options.remove_method = "auto".into();
        state.status["models"][0]["id"] = json!("osor-sdxl");
        state.status["models"][0]["label"] = json!("OSOR-SDXL Phase II");
        state.status["models"][0]["installed"] = json!(true);
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Advanced").click();
        h.run_steps(8);
        h.get_by_label("Use this model").click();
        h.run_steps(3);
        assert_eq!(h.state().app.session.prefs().generative_remove.selected_model, "osor-sdxl");
        assert_eq!(h.state().app.session.prefs().generative_remove.workflow, "custom");
        assert_eq!(
            h.state().app.ui.tool_options.remove_method,
            "ai",
            "an explicit model choice must use that model, rather than Custom Auto's Classical fallback"
        );
        assert!(!h.state().app.session.has_jobs(), "model selection must not download or infer");
    }

    #[test]
    fn model_downloads_propose_their_setup_after_a_classical_choice() {
        use egui_kittest::{Harness, kittest::Queryable};
        for (model_id, profile) in [("lama", "lama"), ("migan-v1", "compact"), ("osor-sdxl", "recommended")] {
            let mut state = fixture();
            state.app.run("remove.setup", json!({"workflow":"classical"})).unwrap();
            state.status["models"][0]["id"] = json!(model_id);
            let mut h = Harness::builder()
                .with_size(vec2(700.0, 650.0))
                .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
            h.run_steps(3);
            h.get_by_label("Download and use…").click();
            h.run_steps(3);
            let app = &h.state().app;
            let setup = app.ui.dialogs.iter().find(|dialog| dialog.fields.get("__prefsui").and_then(Value::as_str) == Some("removeSetup")).unwrap();
            assert_eq!(setup.fields["workflow"], profile);
            assert_eq!(app.session.prefs().generative_remove.workflow, "classical", "a proposal changes no preference");
            assert!(!app.session.has_jobs(), "confirmation and license consent precede downloads");
        }
    }

    #[test]
    fn model_availability_is_per_adapter() {
        use egui_kittest::{
            Harness,
            kittest::{NodeT, Queryable},
        };
        let mut state = fixture();
        state.status["models"][0]["available"] = json!(false);
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        assert!(h.get_by_label("Download and use…").accesskit_node().is_disabled(), "another compiled adapter must not enable this model");
        h.get_by_label("Advanced").click();
        h.run_steps(8);
        assert!(h.get_by_label("Allow this model").accesskit_node().is_disabled());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_job_progress_can_be_cancelled_and_reports_the_result() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        let (done, wait) = std::sync::mpsc::channel();
        let started = state
            .app
            .session
            .start_job(
                "ai.models.install",
                json!({}),
                "Download model",
                false,
                move |ctx| {
                    ctx.progress(0.4, "Downloading");
                    wait.recv().map_err(|e| photocraft_engine::EngineError::Other(e.to_string()))?;
                    Ok(())
                },
                |_, ()| Ok(json!({})),
            )
            .unwrap();
        let photocraft_engine::jobs::Started::Job(id) = started else { panic!("native job should run in the background") };
        state.fields.insert("job".into(), json!(id.0));
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Cancel model operation").click();
        h.run_steps(3);
        assert!(h.state().app.session.jobs_with_recent().iter().any(|job| job.id == id && job.state == "cancelled"));
        assert_eq!(h.state().fields["message"], "Model installation cancelled. Installed models were kept.");
        assert_eq!(h.state().fields["error"], false);
        h.state_mut().fields = Map::new();
        h.run_steps(3);
        h.get_by_label("Model installation cancelled. Installed models were kept.");
        done.send(()).unwrap();
        h.state_mut().app.session.join_cancelled_jobs();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_job_failure_is_visible_in_manager() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        let started = state
            .app
            .session
            .start_job(
                "ai.models.import",
                json!({}),
                "Import model",
                false,
                |_| -> photocraft_engine::Result<()> { Err(photocraft_engine::EngineError::Other("model verification failed".into())) },
                |_, ()| Ok(json!({})),
            )
            .unwrap();
        let photocraft_engine::jobs::Started::Job(id) = started else { panic!("native job should run in the background") };
        assert!(state.app.session.wait_job(id).is_err());
        state.fields.insert("job".into(), json!(id.0));
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        assert_eq!(h.state().fields["error"], true);
        assert!(h.state().fields["message"].as_str().unwrap().contains("model verification failed"));
        h.get_by_label(h.state().fields["message"].as_str().unwrap());
    }

    #[test]
    fn denied_manual_selection_keeps_the_whole_existing_setup() {
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152})).unwrap();
        state.app.ui.tool_options.remove_method = "auto".into();
        let before = state.app.session.prefs_value();
        state.app.session.authorize = Some(|command, params| {
            if command == "prefs.set" && params.get("path").and_then(Value::as_str) == Some("generativeRemove.selectedModel") {
                Err(photocraft_engine::EngineError::Other("selection denied".into()))
            } else {
                Ok(())
            }
        });
        assert!(select_model(&mut state.app, "osor-sdxl").unwrap_err().contains("selection denied"));
        assert_eq!(state.app.session.prefs_value(), before);
        assert_eq!(state.app.ui.tool_options.remove_method, "auto");
        assert!(!state.app.session.has_jobs());
    }

    #[test]
    fn automation_denial_precedes_manual_processing_opt_in() {
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"classical"})).unwrap();
        state.app.ui.tool_options.remove_method = "auto".into();
        state.app.services.automation_command = Some(Box::new(|command, params| {
            if command == "prefs.set" && params.get("path").and_then(Value::as_str) == Some("generativeRemove.enabled") {
                Err("processing opt-in denied".into())
            } else {
                Ok(())
            }
        }));
        state.app.automation_input = true;
        let before = state.app.session.prefs_value();
        assert!(select_model(&mut state.app, "migan-v1").unwrap_err().contains("processing opt-in denied"));
        assert_eq!(state.app.session.prefs_value(), before);
        assert_eq!(state.app.ui.tool_options.remove_method, "auto");
        assert!(!state.app.session.has_jobs());
    }

    #[test]
    fn current_profile_reports_friendly_readiness_and_remaining_download() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"lama"})).unwrap();
        state.status["models"] = json!([
            {"id":"lama","label":"LaMa","installed":true,"available":true,"canAttempt":true,"downloadBytes":208044816},
            {"id":"osor-sdxl","label":"OSOR-SDXL Phase II","installed":true,"available":true,"canAttempt":true,"downloadBytes":8415101006_u64}
        ]);
        let label = format!("Current setup: {}", super::super::remove_setup::workflow_label("lama"));
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label(&label);
        h.get_by_label("Ready");
        assert!(h.query_by_label("Selected model").is_none(), "normal setups do not display a misleading selected-model radio");
        assert!(h.query_by_label("Use this model").is_none(), "manual controls begin collapsed");
        h.state_mut().app.run("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152})).unwrap();
        h.state_mut().status["models"][1]["installed"] = json!(false);
        h.state_mut().status["models"][1]["canAttempt"] = json!(false);
        h.run_steps(3);
        h.get_by_label("Setup incomplete");
        h.get_by_label("8.4 GB download remaining");
        h.get_by_label("Finish setup…").click();
        h.run_steps(3);
        assert!(h.state().app.ui.dialogs.iter().any(|dialog| dialog.fields.get("workflow") == Some(&json!("recommended"))));
        assert!(!h.state().app.session.has_jobs());
    }

    #[test]
    fn installed_models_need_shared_setup_confirmation_before_showing_ready() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"lama"})).unwrap();
        state.status["sharedSetupRequired"] = json!(true);
        state.status["models"] = json!([{
            "id":"lama","label":"LaMa","installed":true,"available":true,"canAttempt":true,
            "configuredReady":false,"configuredReason":"Confirm the shared AI setup to reuse these installed models."
        }]);
        assert!(!current_setup(&state.app, &state.status).ready, "installed weights do not establish a shared tool binding");
        let before = state.app.session.prefs_value();
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Setup incomplete");
        assert!(h.query_by_label("Ready").is_none());
        h.get_by_label("Finish setup…").click();
        h.run_steps(3);
        assert!(h.state().app.ui.dialogs.iter().any(|dialog| dialog.fields.get("workflow") == Some(&json!("lama"))));
        assert_eq!(h.state().app.session.prefs_value(), before, "opening confirmation changes no settings");
        assert!(!h.state().app.session.has_jobs(), "binding recovery never starts a download");
        h.state_mut().status["sharedSetupRequired"] = json!(false);
        h.state_mut().status["models"][0]["configuredReady"] = json!(true);
        assert!(current_setup(&h.state().app, &h.state().status).ready);
    }

    #[test]
    fn custom_model_readiness_obeys_shared_choice_and_accepts_legacy_metadata() {
        let mut state = fixture();
        state.app.session.edit_prefs(|prefs| {
            prefs.generative_remove.workflow = "custom".into();
            prefs.generative_remove.setup_complete = true;
            prefs.generative_remove.enabled = true;
            prefs.generative_remove.enabled_models = vec!["migan-v1".into()];
            prefs.generative_remove.selected_model = "migan-v1".into();
        });
        state.status["models"][0]["installed"] = json!(true);
        state.status["models"][0]["canAttempt"] = json!(true);
        assert!(current_setup(&state.app, &state.status).ready, "older metadata still represents ready installed data");
        state.status["models"][0]["configuredReady"] = json!(false);
        assert!(!current_setup(&state.app, &state.status).ready, "a shared choice of another model is authoritative");
        state.status["models"][0]["configuredReady"] = json!(true);
        assert!(current_setup(&state.app, &state.status).ready);
        assert!(!state.app.session.has_jobs());
    }

    #[test]
    fn installed_models_are_not_ready_when_the_saved_memory_allowance_is_too_low() {
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152})).unwrap();
        state.status["models"] = json!([
            {"id":"lama","installed":true,"available":true,"canAttempt":true},
            {"id":"osor-sdxl","installed":true,"available":true,"canAttempt":true}
        ]);
        let ample_memory = Some(128_u64 << 30);
        assert!(current_setup_with_memory(&state.app, &state.status, ample_memory).ready);
        state.app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 4096);
        let current = current_setup_with_memory(&state.app, &state.status, ample_memory);
        assert!(!current.ready);
        assert!(current.memory_warning.unwrap().contains("memory allowance"));
        state.app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 49152);
        let current = current_setup_with_memory(&state.app, &state.status, Some(16_u64 << 30));
        assert!(!current.ready);
        assert!(current.memory_warning.unwrap().contains("more memory"));
        assert!(!state.app.session.has_jobs());
    }

    #[test]
    fn corrupt_installed_model_offers_manage_files_without_download_or_deletion() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"compact"})).unwrap();
        state.status["models"][0]["installed"] = json!(true);
        state.status["models"][0]["canAttempt"] = json!(false);
        state.status["models"][0]["reason"] = json!("The installed model has an unexpected size; remove it and install again");
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Needs attention");
        assert!(h.query_by_label("Ready").is_none());
        assert!(h.query_by_label("Download and use…").is_none());
        h.get_all_by_label("Manage model files…").next().unwrap().click();
        h.run_steps(8);
        assert_eq!(h.state().fields["advanced"], true);
        h.get_by_label("Remove installed model…");
        assert!(!h.state().app.session.has_jobs());
    }

    #[test]
    fn shared_busy_state_does_not_claim_missing_models_are_in_use() {
        use egui_kittest::{
            Harness,
            kittest::{NodeT, Queryable},
        };
        let mut state = fixture();
        state.status["models"][0]["busy"] = json!(true);
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Not installed");
        assert!(h.query_by_label("In use").is_none());
        assert!(h.get_by_label("Download and use…").accesskit_node().is_disabled());
        assert!(h.get_by_label("Change setup…").accesskit_node().is_disabled());
    }

    #[test]
    fn removing_a_model_requires_confirmation_and_warns_about_current_setup() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"compact"})).unwrap();
        state.status["models"][0]["installed"] = json!(true);
        let before = state.app.session.prefs_value();
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Advanced").click();
        h.run_steps(8);
        h.get_by_label("Remove installed model…").click();
        h.run_steps(3);
        h.get_by_label("The current setup will become incomplete. Download or import this model again before using that setup.");
        h.get_by_label("Keep model");
        assert!(!h.state().app.session.has_jobs(), "opening confirmation performs no model operation");
        assert_eq!(h.state().app.session.prefs_value(), before);
        h.get_by_label("Keep model").click();
        h.run_steps(3);
        assert!(h.state().fields.get("removeModelId").is_none());
        assert!(!h.state().app.session.has_jobs());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn small_model_manager_keeps_progress_cancel_and_close_inside_the_window() {
        use egui_kittest::{
            Harness,
            kittest::{NodeT, Queryable},
        };
        let (done, wait) = std::sync::mpsc::channel();
        let mut h = Harness::builder().with_size(vec2(650.0, 600.0)).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.session
                .start_job(
                    "ai.models.install",
                    json!({}),
                    "Download LaMa",
                    false,
                    move |ctx| {
                        ctx.progress(0.4, "Downloading LaMa");
                        wait.recv().map_err(|error| photocraft_engine::EngineError::Other(error.to_string()))?;
                        Ok(())
                    },
                    |_, ()| Ok(json!({})),
                )
                .unwrap();
            open(&mut app);
            app
        });
        h.run_steps(8);
        let screen = h.ctx.content_rect();
        for label in ["Cancel model operation", "Close"] {
            // Windows/Linux have a title-bar Close as well as the footer button.
            for node in h.get_all_by_role_and_label(egui::accesskit::Role::Button, label) {
                assert!(!node.accesskit_node().is_disabled());
                assert!(screen.contains_rect(node.rect()), "{label} must remain inside the small window");
            }
        }
        // Exercise the footer Close, below the title-bar control on platforms that have both.
        h.get_all_by_role_and_label(egui::accesskit::Role::Button, "Close").max_by(|a, b| a.rect().center().y.total_cmp(&b.rect().center().y)).unwrap().click();
        h.run_steps(3);
        assert!(h.state().ui.dialogs.iter().all(|dialog| dialog.fields.get("__prefsui").and_then(Value::as_str) != Some("removeModels")));
        assert!(h.state().session.has_jobs(), "closing the manager leaves the download running");
        let job = h.state().session.jobs().into_iter().next().unwrap().id;
        done.send(()).unwrap();
        h.state_mut().session.wait_job(job).unwrap();
    }

    #[test]
    #[ignore = "offscreen fixture evidence; set PHOTOCRAFT_MODELS_SNAPSHOT and optional STATE=partial|ready"]
    fn model_ux_models_state_snapshot() {
        let _gpu = photocraft_testkit::gpu_lock();
        let path = std::env::var("PHOTOCRAFT_MODELS_SNAPSHOT").unwrap();
        let phase = std::env::var("PHOTOCRAFT_MODELS_SNAPSHOT_STATE").unwrap_or_else(|_| "fresh".into());
        let mut state = fixture();
        state.app.run("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152})).unwrap();
        state.status["models"] = Value::Array(
            state
                .app
                .session
                .removal_service()
                .catalog()
                .iter()
                .map(|spec| {
                    let installed = phase == "ready" || (phase == "partial" && spec.id == "lama");
                    json!({"id":spec.id,"label":spec.label,"description":spec.description,"downloadBytes":spec.download_bytes,
                "memoryMib":spec.memory_mib,"installed":installed,"canAttempt":installed,"available":true,
                "canDownload":true,"license":spec.license,"source":spec.source})
                })
                .collect(),
        );
        let size = if std::env::var_os("PHOTOCRAFT_MODELS_SMALL_SNAPSHOT").is_some() { vec2(650.0, 600.0) } else { vec2(850.0, 900.0) };
        let mut h = egui_kittest::Harness::builder().with_size(size).wgpu().build_ui_state(
            |ui, state: &mut ModelUi| {
                ui.set_width(ui.available_width().min(620.0));
                ui.heading("Generative Remove Models");
                ui.separator();
                render(&mut state.app, ui, &mut state.fields, &state.status);
                ui.separator();
                let _ = ui.button("Close");
            },
            state,
        );
        PhotocraftApp::setup_context(&h.ctx, Default::default());
        h.run_steps(4);
        h.render().unwrap().save(path).unwrap();
    }

    #[test]
    fn core_remove_labels_are_translated_in_the_complete_catalogs() {
        for code in ["ja", "zh-hans", "zh-hant", "es", "ru", "uk", "sr", "cs", "fr", "id", "ko", "pl", "de", "pt-br", "el", "nl", "it", "hu", "tr", "vi", "hi"]
        {
            let lang = crate::i18n::Lang::from_code(code).unwrap();
            for label in [
                "Generative Remove Models…",
                "Classical",
                "Models…",
                "Enable local AI removal",
                "Not installed",
                "Download",
                "Install from file…",
                "Local AI inference is unavailable in this build. Classical Remove remains available.",
                "local AI processing is disabled",
                "Local background reconstruction; 512 px context, CPU processing",
                "Compact local reconstruction; exact older MI-GAN pipeline, 512 px CPU context",
            ] {
                assert_ne!(crate::i18n::tr(lang, label), label, "{code}: {label}");
            }
        }
    }

    #[test]
    fn model_cards_show_distinct_errors_without_repeating_the_install_status() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut state = fixture();
        state.status["models"][0]["reason"] = json!("model is not installed");
        let mut h = Harness::builder()
            .with_size(vec2(700.0, 650.0))
            .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
        h.run_steps(3);
        h.get_by_label("Not installed");
        assert!(h.query_by_label("model is not installed").is_none());
        h.state_mut().status["runtimeAvailable"] = json!(false);
        h.state_mut().status["models"][0]["reason"] = json!("Model storage is read-only");
        h.run_steps(3);
        h.get_by_label("Model storage is read-only");
    }
    #[test]
    fn model_gpu_execution_reports_mixed_and_cpu_fallback() {
        use egui_kittest::{Harness, kittest::Queryable};
        for (uses_gpu, fallback, expected) in [
            (true, None, "Last run: MI-GAN v1 used GPU and CPU operations."),
            (false, Some("GPU arena exceeds the captured budget"), "Last run: MI-GAN v1 used the CPU."),
        ] {
            let mut state = fixture();
            state.status["lastExecution"] = json!({"modelId":"migan-v1","usesGpu":uses_gpu,"device":"test device",
                "graph":{"gpuNodes":8,"cpuNodes":2,"transferNodes":2,"gpuArenaBytes":4096},"fallbackReason":fallback});
            let mut harness = Harness::builder()
                .with_size(vec2(700.0, 650.0))
                .build_ui_state(|ui, state: &mut ModelUi| render(&mut state.app, ui, &mut state.fields, &state.status), state);
            harness.run_steps(3);
            assert!(harness.query_by_label(expected).is_some(), "completed backend must be visible: {expected}");
            assert_eq!(harness.query_by_label("GPU unavailable for this run; used the same model on CPU.").is_some(), fallback.is_some());
        }
    }
}

#[cfg(test)]
mod osor_chooser_tests {
    use super::*;
    #[test]
    fn osor_bundle_import_asks_for_a_directory() {
        let services = crate::Services { file_dialog: Some(Box::new(|_, _, _| {})), ..Default::default() };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let reply = import_model(&mut app, "osor-sdxl").unwrap();
        assert_eq!(reply["fileDialog"], "folder", "OSOR is a verified four-file bundle");
        assert!(!app.session.has_jobs(), "choosing a bundle starts no inference");
    }
}
