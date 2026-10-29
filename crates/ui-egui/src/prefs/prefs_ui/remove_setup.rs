//! First-use choices and the explicitly authorized, sequential setup downloads.

use super::{DialogKind, JobId, Map, PhotocraftApp, RichText, Tokens, Value, VecDeque, json};

const FULL_MEMORY_MIB: u64 = 49_152;

#[derive(Default)]
pub(super) struct Runtime {
    /// First-use/recovery is considered once per session, after the Remove tool is selected.
    prompted: bool,
    queue: VecDeque<String>,
    active: Option<JobId>,
    active_model: Option<String>,
    workflow: String,
    total: usize,
    completed: usize,
    automated: bool,
    authorize: Option<fn(&str, &Value) -> photocraft_engine::Result<()>>,
    pub(super) message: String,
    pub(super) error: bool,
}

/// Metadata only: computing the summary never opens weights or starts a model operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ProfileSummary {
    pub total: usize,
    pub installed: usize,
    pub missing: Vec<String>,
    pub remaining_bytes: u64,
    /// Required adapters are available and any present models can be attempted. RAM is separate.
    pub available: bool,
}

impl ProfileSummary {
    pub(super) fn ready(&self) -> bool {
        self.available && self.missing.is_empty()
    }
}

pub(super) fn busy(app: &PhotocraftApp) -> bool {
    app.prefs_rt.remove_setup.active.is_some() || !app.prefs_rt.remove_setup.queue.is_empty()
}

fn workflow(f: &Map<String, Value>) -> &str {
    f.get("workflow").and_then(Value::as_str).unwrap_or("recommended")
}

fn known_profile(workflow: &str) -> bool {
    matches!(workflow, "recommended" | "lama" | "compact" | "classical")
}

pub(super) fn components(workflow: &str) -> &'static [&'static str] {
    match workflow {
        "recommended" => &["lama", "osor-sdxl"],
        "lama" => &["lama"],
        "compact" => &["migan-v1"],
        _ => &[],
    }
}

pub(super) fn workflow_label(workflow: &str) -> &'static str {
    match workflow {
        "recommended" => tl!("Recommended: LaMa + OSOR"),
        "lama" => tl!("Smaller: LaMa"),
        "compact" => tl!("Compact: MI-GAN"),
        "classical" => tl!("Classical: no download"),
        _ => tl!("Custom"),
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

pub(super) fn profile_summary(status: &Value, workflow: &str) -> ProfileSummary {
    let ids = components(workflow);
    let mut summary = ProfileSummary { total: ids.len(), installed: 0, missing: Vec::new(), remaining_bytes: 0, available: known_profile(workflow) };
    for id in ids {
        let entry = model(status, id);
        summary.available &= entry
            .is_some_and(|entry| flag(entry, "available") && (!flag(entry, "installed") || entry.get("canAttempt").and_then(Value::as_bool).unwrap_or(true)));
        if entry.is_some_and(|entry| flag(entry, "installed")) {
            summary.installed = summary.installed.saturating_add(1);
        } else {
            summary.missing.push((*id).into());
            let bytes = entry.and_then(|entry| entry.get("downloadBytes").or_else(|| entry.get("bytes"))).and_then(Value::as_u64).unwrap_or(0);
            summary.remaining_bytes = summary.remaining_bytes.saturating_add(bytes);
        }
    }
    summary
}

fn enough_full_memory(physical_memory: Option<u64>) -> bool {
    physical_memory.is_none_or(|bytes| bytes >= FULL_MEMORY_MIB.saturating_mul(1 << 20))
}

pub(super) fn best_compatible_workflow(status: &Value, physical_memory: Option<u64>) -> &'static str {
    if enough_full_memory(physical_memory) && profile_summary(status, "recommended").available {
        "recommended"
    } else if profile_summary(status, "lama").available {
        "lama"
    } else if profile_summary(status, "compact").available {
        "compact"
    } else {
        "classical"
    }
}

fn missing(status: &Value, selected: &str) -> Vec<String> {
    profile_summary(status, selected).missing
}

/// Open a proposal for a specific profile. Existing settings and downloads are unchanged.
pub(super) fn open_profile(app: &mut PhotocraftApp, selected: &str) -> u64 {
    app.prefs_rt.remove_setup.prompted = true;
    if let Some(id) = app.ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| d.id) {
        if let Some(dialog) = app.ui.dialog_mut(id) {
            dialog.fields.insert("workflow".into(), json!(selected));
            dialog.fields.insert("acceptLicenses".into(), json!(false));
        }
        return id;
    }
    let fields = json!({"__prefsui":"removeSetup", "__label":"Set Up Generative Remove", "workflow":selected,
        "memoryUsageMb":u64::from(app.session.prefs().performance.memory_usage_mb).max(FULL_MEMORY_MIB), "acceptLicenses":false});
    app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default())
}

pub(super) fn open(app: &mut PhotocraftApp) -> u64 {
    if let Some(dialog) = app.ui.dialogs.iter().find(|d| owns(&d.fields)) {
        return dialog.id;
    }
    let current = &app.session.prefs().generative_remove;
    let selected = if current.setup_complete && known_profile(&current.workflow) {
        current.workflow.clone()
    } else {
        status(app).map(|status| best_compatible_workflow(&status, crate::gpu_canvas::physical_memory()).to_owned()).unwrap_or_else(|_| "classical".into())
    };
    open_profile(app, &selected)
}

/// An explicit OSOR setup action proposes both complementary models without changing settings.
#[cfg(test)]
pub(super) fn open_recommended(app: &mut PhotocraftApp) -> u64 {
    open_profile(app, "recommended")
}

fn model_label(app: &PhotocraftApp, id: Option<&str>) -> String {
    id.and_then(|id| app.session.removal_service().catalog().iter().find(|model| model.id == id))
        .map(|model| tl!(model.label).to_string())
        .unwrap_or_else(|| tl!("Model").into())
}

fn progress_message(app: &PhotocraftApp, stage: &str) -> String {
    let runtime = &app.prefs_rt.remove_setup;
    let label = model_label(app, runtime.active_model.as_deref());
    let step = runtime.completed.saturating_add(1);
    let total = runtime.total.max(step.saturating_add(runtime.queue.len()));
    crate::i18n::fmt(
        tl!("{model} ({step} of {total}): {stage}"),
        &[("model", &label), ("step", &step.to_string()), ("total", &total.to_string()), ("stage", stage)],
    )
}

fn stopped(app: &mut PhotocraftApp, cancelled: bool, error: Option<&str>) {
    let runtime = &mut app.prefs_rt.remove_setup;
    runtime.active = None;
    runtime.active_model = None;
    runtime.queue.clear();
    runtime.error = !cancelled;
    runtime.message = if cancelled {
        tl!("Model download cancelled. Installed models are kept. Open Models to download the remaining models or choose a smaller setup.").into()
    } else if let Some(error) = error {
        crate::i18n::fmt(
            tl!("Model setup stopped: {error}. Installed models are kept. Open Models to try again or choose a smaller setup."),
            &[("error", error)],
        )
    } else {
        tl!("Model setup stopped. Installed models are kept. Open Models to try again or choose a smaller setup.").into()
    };
    app.ui.status = runtime.message.clone();
    app.ui.status_error = runtime.error;
}

fn model_finished(app: &mut PhotocraftApp) {
    let runtime = &mut app.prefs_rt.remove_setup;
    runtime.active = None;
    runtime.active_model = None;
    runtime.completed = runtime.completed.saturating_add(1);
    if !runtime.queue.is_empty() {
        return;
    }
    let selected = if runtime.workflow.is_empty() { app.session.prefs().generative_remove.workflow.clone() } else { runtime.workflow.clone() };
    let ready = status(app).is_ok_and(|status| profile_summary(&status, &selected).ready());
    app.prefs_rt.remove_setup.error = !ready;
    app.prefs_rt.remove_setup.message = if ready {
        tl!("Remove setup is ready. Return to your image to start removing objects.").into()
    } else {
        tl!("Model setup is incomplete. Open Models to download the remaining models or choose another setup.").into()
    };
    app.ui.status = app.prefs_rt.remove_setup.message.clone();
    app.ui.status_error = !ready;
}

fn poll_setup(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(id) = app.prefs_rt.remove_setup.active else {
        return;
    };
    let job = app.session.jobs_with_recent().into_iter().find(|job| job.id == id);
    match job {
        Some(job) if job.state == "running" => {
            let stage = if job.message.is_empty() { tl!("Starting…") } else { &job.message };
            app.prefs_rt.remove_setup.message = progress_message(app, stage);
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        Some(job) if job.state == "done" => model_finished(app),
        Some(job) if job.state == "cancelled" => stopped(app, true, None),
        job => stopped(app, false, job.as_ref().and_then(|job| job.error.as_deref())),
    }
}

fn start_next(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.prefs_rt.remove_setup.active.is_some() {
        return;
    }
    let Some(id) = app.prefs_rt.remove_setup.queue.pop_front() else {
        return;
    };
    app.prefs_rt.remove_setup.active_model = Some(id.clone());
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
                model_finished(app);
                ctx.request_repaint();
            } else {
                app.prefs_rt.remove_setup.message = progress_message(app, tl!("Starting…"));
            }
        }
        Err(error) => stopped(app, false, Some(&error)),
    }
}

fn prompt_once(app: &mut PhotocraftApp) {
    if app.ui.tool != crate::state::Tool::Remove
        || app.prefs_rt.remove_setup.prompted
        || busy(app)
        || super::models::model_folder_pending(app)
        || !app.ui.dialogs.is_empty()
        || app.ui.shell.dialog.is_some()
        || !app.session.removal_service().management_available()
    {
        return;
    }
    // Configured profiles need one metadata check after restart, not a directory scan per frame.
    // Migrated Custom choices stay untouched and never get an unsolicited preset proposal.
    app.prefs_rt.remove_setup.prompted = true;
    let prefs = app.session.prefs().generative_remove.clone();
    if !prefs.setup_complete && !prefs.enabled {
        open(app);
    } else if prefs.setup_complete
        && known_profile(&prefs.workflow)
        && prefs.workflow != "classical"
        && status(app).is_ok_and(|status| !profile_summary(&status, &prefs.workflow).ready())
    {
        open_profile(app, &prefs.workflow);
    }
}

/// Downloads remain authorized and sequential when the dialog closes. Failure/cancellation
/// stops the queue; recovery proposes a choice and never silently restarts a transfer.
pub(crate) fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    poll_setup(app, ctx);
    start_next(app, ctx);
    prompt_once(app);
}

fn validation(app: &PhotocraftApp, f: &Map<String, Value>, status: &Value) -> Result<(), String> {
    if busy(app) || app.session.jobs().iter().any(|job| job.command.starts_with("ai.models.")) {
        return Err(tl!("Wait for model setup to finish or cancel it first.").into());
    }
    let selected = workflow(f);
    if !known_profile(selected) {
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
        let required = entry.get("memoryMib").and_then(Value::as_u64).unwrap_or(0);
        if selected != "recommended" && required > 0 && u64::from(app.session.prefs().performance.memory_usage_mb) <= required {
            return Err(crate::i18n::fmt(
                tl!(
                    "This setup needs more than {memory} of PhotoCraft memory allowance, including room for image buffers. Increase Memory Usage in Preferences > Performance or choose another setup."
                ),
                &[("memory", &memory_size(required))],
            ));
        }
        if flag(entry, "busy") {
            return Err(tl!("A model operation is still finishing. Wait before trying again.").into());
        }
        if !flag(entry, "available") {
            return Err(tl!("This model is unavailable in this build. Choose a smaller setup.").into());
        }
        if flag(entry, "installed") && entry.get("canAttempt").and_then(Value::as_bool) == Some(false) {
            let reason = entry.get("reason").and_then(Value::as_str).unwrap_or(tl!("The installed model needs attention."));
            return Err(crate::i18n::fmt(tl!("{reason}. Close this dialog and open Models to manage model files."), &[("reason", reason)]));
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
    authorize_setup(app, f, &status)
}

fn authorize_setup(app: &PhotocraftApp, f: &Map<String, Value>, status: &Value) -> Result<(), String> {
    validation(app, f, status)?;
    let commands = std::iter::once(("remove.setup", setup_params(f)))
        .chain(missing(status, workflow(f)).into_iter().map(|id| ("ai.models.install", json!({"modelId":id}))));
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
    let summary = profile_summary(&status, selected);
    let result = app.run("remove.setup", setup_params(f))?;
    app.ui.tool_options.remove_method = "auto".into();
    // Choosing models must not expand the area the user asked Remove to repair.
    if selected != "recommended" {
        app.ui.tool_options.remove_effects = false;
    }
    app.prefs_rt.remove_setup.queue = summary.missing.into();
    app.prefs_rt.remove_setup.active_model = None;
    app.prefs_rt.remove_setup.workflow = selected.into();
    app.prefs_rt.remove_setup.total = summary.total;
    app.prefs_rt.remove_setup.completed = summary.installed;
    app.prefs_rt.remove_setup.automated = app.automation_input;
    app.prefs_rt.remove_setup.authorize = app.session.authorize;
    app.prefs_rt.remove_setup.error = false;
    app.prefs_rt.remove_setup.message = if busy(app) {
        tl!("Downloading the missing models. You can keep working while setup finishes. Open Models to check progress or cancel.").into()
    } else {
        tl!("Remove setup is ready. You can change it in Models.").into()
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
    match status(app) {
        Ok(status) => render(app, ui, f, &status),
        Err(error) => {
            ui.label(error);
        }
    }
}

fn download_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.1} GB", bytes as f64 / 1_000_000_000.0) } else { format!("{:.0} MB", bytes as f64 / 1_000_000.0) }
}

fn memory_size(mib: u64) -> String {
    if mib.is_multiple_of(1024) { format!("{} GiB", mib / 1024) } else { format!("{:.1} GiB", mib as f64 / 1024.0) }
}

fn profile_option(ui: &mut egui::Ui, selected: &mut String, id: &str, status: &Value, physical_memory: Option<u64>) -> bool {
    let t = Tokens::get(ui.ctx());
    let summary = profile_summary(status, id);
    let compatible = summary.available && (id != "recommended" || enough_full_memory(physical_memory));
    let changed = ui.add_enabled_ui(compatible, |ui| ui.radio_value(selected, id.into(), workflow_label(id)).changed()).inner;
    ui.indent(id, |ui| {
        let purpose = match id {
            "recommended" => tl!("Best for objects and small marks. LaMa and OSOR are chosen automatically; review every result before applying."),
            "lama" => tl!("Repair backgrounds and small marks with a smaller AI download."),
            "compact" => tl!("The smallest AI download for basic repairs."),
            _ => tl!("Use the built-in removal algorithm. You can install AI models later."),
        };
        ui.label(RichText::new(purpose).color(t.text_dim));
        let readiness = if summary.total == 0 {
            tl!("Ready to use. No download.").to_string()
        } else if summary.missing.is_empty() {
            tl!("Installed. No download needed.").to_string()
        } else {
            crate::i18n::fmt(
                tl!("{installed} of {total} models installed. Download remaining: {size}."),
                &[("installed", &summary.installed.to_string()), ("total", &summary.total.to_string()), ("size", &download_size(summary.remaining_bytes))],
            )
        };
        ui.label(RichText::new(readiness).color(t.text_dim));
        if !summary.available {
            let attention = components(id).iter().find_map(|id| {
                model(status, id)
                    .filter(|entry| flag(entry, "available") && flag(entry, "installed") && entry.get("canAttempt").and_then(Value::as_bool) == Some(false))
            });
            let reason = attention.and_then(|entry| entry.get("reason").and_then(Value::as_str)).unwrap_or(if attention.is_some() {
                tl!("The installed model needs attention. Open Models for details.")
            } else {
                tl!("Not supported by this build.")
            });
            ui.label(RichText::new(reason).color(t.text_dim));
        }
        if id == "recommended" {
            let requirement = match physical_memory {
                Some(bytes) if !enough_full_memory(Some(bytes)) => crate::i18n::fmt(
                    tl!("Requires 48 GiB of computer memory; this computer has {memory}. Choose a smaller setup."),
                    &[("memory", &memory_size(bytes / (1 << 20)))],
                ),
                _ => tl!("Requires at least 48 GiB of computer memory.").into(),
            };
            ui.label(RichText::new(requirement).color(t.text_dim));
        }
    });
    ui.add_space(6.0);
    changed
}

fn full_requirements(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, summary: &ProfileSummary) {
    let t = Tokens::get(ui.ctx());
    ui.separator();
    if summary.missing.iter().any(|id| id == "osor-sdxl") {
        ui.label(tl!("OSOR will be prepared after downloading. Allow about 19.1 GB of free disk space; installed OSOR weights use about 10.6 GB."));
    } else if model(status, "osor-sdxl").is_some_and(|entry| entry.get("canAttempt").and_then(Value::as_bool) == Some(false)) {
        ui.label(tl!("The installed OSOR model needs attention. Open Models for its details before using this setup."));
    } else {
        ui.label(tl!("OSOR is installed (about 10.6 GB). No OSOR download or preparation is needed."));
    }
    let proposed = f.get("memoryUsageMb").and_then(Value::as_u64).unwrap_or(FULL_MEMORY_MIB);
    let current = u64::from(app.session.prefs().performance.memory_usage_mb);
    let message = if proposed == current {
        crate::i18n::fmt(tl!("PhotoCraft memory allowance: {memory} (already set)."), &[("memory", &memory_size(proposed))])
    } else {
        crate::i18n::fmt(
            tl!("Confirming this setup changes PhotoCraft's memory allowance from {current} to {proposed}, saved in Preferences."),
            &[("current", &memory_size(current)), ("proposed", &memory_size(proposed))],
        )
    };
    ui.label(RichText::new(message).color(t.text_dim));
    ui.collapsing(tl!("Advanced memory settings"), |ui| {
        ui.horizontal(|ui| {
            let label = ui.label(tl!("PhotoCraft memory allowance (MiB):"));
            let mut budget = proposed;
            ui.add(egui::DragValue::new(&mut budget).range(FULL_MEMORY_MIB..=1_048_576).speed(1024)).labelled_by(label.id);
            f.insert("memoryUsageMb".into(), json!(budget));
        });
    });
}

fn licenses(ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value, summary: &ProfileSummary) {
    if summary.missing.is_empty() {
        return;
    }
    ui.separator();
    ui.label(tl!("Licenses for the models to download:"));
    for id in &summary.missing {
        if let Some(entry) = model(status, id) {
            let label = entry.get("label").and_then(Value::as_str).unwrap_or(id);
            let url =
                entry.get("licenseUrl").and_then(Value::as_str).filter(|s| !s.is_empty()).or_else(|| entry.get("source").and_then(Value::as_str)).unwrap_or("");
            let license = entry.get("license").and_then(Value::as_str).unwrap_or("");
            ui.hyperlink_to(format!("{label}: {license}"), url);
        }
    }
    if summary.missing.iter().any(|id| id == "osor-sdxl") {
        ui.label(tl!("OSOR weights use CreativeML Open RAIL++-M, which includes use restrictions."));
    }
    let mut accepted = f.get("acceptLicenses").and_then(Value::as_bool).unwrap_or(false);
    crate::widgets::checkbox(ui, &mut accepted, tl!("I have read and accept the model licenses."));
    f.insert("acceptLicenses".into(), json!(accepted));
}

fn render(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value) {
    let t = Tokens::get(ui.ctx());
    ui.label(tl!("Choose local models for Remove. Your images stay on this computer. A brush stroke never downloads a model."));
    let mut selected = workflow(f).to_owned();
    let current = &app.session.prefs().generative_remove;
    let recovering = current.setup_complete && current.workflow == selected && !profile_summary(status, &selected).ready();
    if recovering && !busy(app) {
        ui.add_space(6.0);
        ui.label(tl!("Your selected setup is not ready yet. Download the remaining models, or choose another setup below."));
    }
    if !app.prefs_rt.remove_setup.message.is_empty() {
        let color = if app.prefs_rt.remove_setup.error { t.danger } else { t.text_dim };
        ui.label(RichText::new(&app.prefs_rt.remove_setup.message).color(color));
    }
    ui.add_space(8.0);
    let physical_memory = crate::gpu_canvas::physical_memory();
    for id in ["recommended", "lama", "compact", "classical"] {
        if profile_option(ui, &mut selected, id, status, physical_memory) {
            f.insert("acceptLicenses".into(), json!(false));
        }
    }
    f.insert("workflow".into(), json!(selected));
    let summary = profile_summary(status, &selected);
    if selected == "recommended" {
        full_requirements(app, ui, f, status, &summary);
    }
    licenses(ui, f, status, &summary);
    if let Err(reason) = validation(app, f, status) {
        ui.add_space(6.0);
        ui.label(RichText::new(reason).color(t.text_dim));
    }
    f.insert("__download".into(), json!(!summary.missing.is_empty()));
    f.insert("__resume".into(), json!(!summary.missing.is_empty() && (summary.installed > 0 || recovering)));
}

pub(crate) fn confirm_label(f: &Map<String, Value>) -> &'static str {
    if f.get("__download").and_then(Value::as_bool) != Some(true) {
        tl!("Use this setup")
    } else if f.get("__resume").and_then(Value::as_bool) == Some(true) {
        tl!("Download remaining")
    } else {
        tl!("Download and use")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_ux_review_rejects_small_profile_with_too_little_memory() {
        let mut app = app();
        app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 1024);
        for (workflow, id, memory) in [("lama", "lama", 3072), ("compact", "migan-v1", 1536)] {
            let metadata = json!({"runtimeAvailable":true,"managementAvailable":true,"models":[{
                "id":id,"available":true,"installed":true,"canAttempt":true,"memoryMib":memory
            }]});
            let fields = json!({"workflow":workflow}).as_object().unwrap().clone();
            let error = validation(&app, &fields, &metadata).unwrap_err();
            assert!(error.contains("Preferences > Performance"), "actionable memory guidance: {error}");
            app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = memory as u32);
            assert!(validation(&app, &fields, &metadata).is_err(), "allow image buffer headroom");
            app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 8192);
            assert!(validation(&app, &fields, &metadata).is_ok());
            app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 1024);
        }
    }

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
        assert!(!app.prefs_rt.remove_setup.error, "explicit cancellation is not an error");
        assert!(!app.ui.status_error);
        assert!(app.prefs_rt.remove_setup.message.contains("Installed models are kept"));
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
            open_recommended(&mut app);
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
            open_recommended(&mut app);
            app
        });
        harness.run_steps(4);
        harness.render().unwrap().save(path).unwrap();
    }

    fn ux_status(installed: &[&str]) -> Value {
        let models: Vec<Value> = photocraft_ml::RemovalService::default()
            .catalog()
            .iter()
            .map(|model| {
                let present = installed.contains(&model.id);
                json!({"id":model.id,"label":model.label,"description":model.description,"available":true,"installed":present,
                    "canAttempt":present,"canDownload":true,"downloadBytes":model.download_bytes,"bytes":model.bytes,
                    "license":model.license,"licenseUrl":model.license_url,"source":model.source})
            })
            .collect();
        json!({"runtimeAvailable":true,"managementAvailable":true,"models":models})
    }

    #[test]
    fn model_ux_defaults_to_best_supported_setup_for_memory_and_build() {
        let mut status = ux_status(&[]);
        assert_eq!(best_compatible_workflow(&status, Some(16 << 30)), "lama");
        assert_eq!(best_compatible_workflow(&status, Some(48 << 30)), "recommended");
        assert_eq!(best_compatible_workflow(&status, None), "recommended");
        for model in status["models"].as_array_mut().unwrap() {
            if matches!(model["id"].as_str(), Some("lama" | "osor-sdxl")) {
                model["available"] = json!(false);
            }
        }
        assert_eq!(best_compatible_workflow(&status, Some(96 << 30)), "compact");
        for model in status["models"].as_array_mut().unwrap() {
            model["available"] = json!(false);
        }
        assert_eq!(best_compatible_workflow(&status, Some(96 << 30)), "classical");
    }

    #[test]
    fn model_ux_profile_summary_counts_only_missing_downloads() {
        let fresh = profile_summary(&ux_status(&[]), "recommended");
        assert_eq!((fresh.total, fresh.installed, fresh.remaining_bytes), (2, 0, 8_623_145_822));
        assert_eq!(fresh.missing, ["lama", "osor-sdxl"]);
        assert!(fresh.available, "missing weights must remain downloadable");
        assert!(!fresh.ready());
        let partial = profile_summary(&ux_status(&["lama"]), "recommended");
        assert_eq!((partial.total, partial.installed, partial.remaining_bytes), (2, 1, 8_415_101_006));
        assert_eq!(partial.missing, ["osor-sdxl"]);
        let ready = profile_summary(&ux_status(&["lama", "osor-sdxl"]), "recommended");
        assert_eq!((ready.total, ready.installed, ready.remaining_bytes), (2, 2, 0));
        assert!(ready.ready());
        assert!(profile_summary(&ux_status(&[]), "classical").ready());
        assert!(!profile_summary(&ux_status(&[]), "custom").ready(), "Custom has separate controls");
    }

    #[test]
    fn model_ux_present_but_unusable_model_is_not_ready_or_redownloaded_implicitly() {
        let mut status = ux_status(&["lama"]);
        let lama = status["models"].as_array_mut().unwrap().iter_mut().find(|model| model["id"] == "lama").unwrap();
        lama["canAttempt"] = json!(false);
        lama["reason"] = json!("The installed model has an unexpected size; remove it and install again");
        let summary = profile_summary(&status, "lama");
        assert_eq!(summary.installed, 1);
        assert!(summary.missing.is_empty());
        assert!(!summary.ready());
        let app = app();
        let error = validation(&app, json!({"workflow":"lama"}).as_object().unwrap(), &status).unwrap_err();
        assert!(error.contains("unexpected size"));
    }

    #[test]
    fn model_ux_open_profile_is_only_a_proposal_and_clears_license_consent() {
        let mut app = app();
        app.run("remove.setup", json!({"workflow":"classical"})).unwrap();
        let before = app.session.prefs_value();
        let id = open_profile(&mut app, "recommended");
        app.ui.dialog_mut(id).unwrap().fields.insert("acceptLicenses".into(), json!(true));
        assert_eq!(open_profile(&mut app, "lama"), id);
        let fields = &app.ui.dialog_mut(id).unwrap().fields;
        assert_eq!(fields["workflow"], "lama");
        assert_eq!(fields["acceptLicenses"], false);
        assert_eq!(app.session.prefs_value(), before);
        assert!(!busy(&app));
        assert!(!app.session.has_jobs());
    }

    #[test]
    fn model_ux_reopening_setup_preserves_a_configured_profile() {
        let mut app = app();
        app.run("remove.setup", json!({"workflow":"compact"})).unwrap();
        let id = open(&mut app);
        assert_eq!(app.ui.dialog_mut(id).unwrap().fields["workflow"], "compact");
        assert!(!app.session.has_jobs());
    }

    struct NoModels;
    impl photocraft_ml::RemovalBackend for NoModels {
        fn ready(&self, _: &str) -> bool {
            false
        }
        fn infer(
            &self,
            _: &str,
            _: &photocraft_ml::RemovalInput,
            _: &dyn photocraft_ml::RemovalControl,
        ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
            Err(photocraft_ml::Error::Inference("setup must not infer".into()))
        }
    }

    fn recovery_app() -> PhotocraftApp {
        let mut app = app();
        app.session.set_removal_service(Arc::new(
            photocraft_ml::RemovalService::new(Some(std::env::temp_dir().join("photocraft-model-ux-no-models"))).with_backend(Arc::new(NoModels)),
        ));
        app.ui.tool = crate::state::Tool::Remove;
        app
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_ux_incomplete_saved_profile_recovers_once_without_starting_downloads() {
        let mut app = recovery_app();
        app.run("remove.setup", json!({"workflow":"recommended"})).unwrap();
        let saved = app.session.prefs().generative_remove.clone();
        tick(&mut app, &egui::Context::default());
        let dialog = app.ui.dialogs.iter().find(|dialog| owns(&dialog.fields)).unwrap();
        let id = dialog.id;
        assert_eq!(dialog.fields["workflow"], "recommended");
        assert_eq!(app.session.prefs().generative_remove, saved);
        assert!(!app.session.has_jobs());
        crate::dialogs::cancel(&mut app, id).unwrap();
        for _ in 0..4 {
            tick(&mut app, &egui::Context::default());
        }
        assert!(app.ui.dialogs.is_empty(), "recovery is not a per-frame popup");
        let mut restarted = recovery_app();
        restarted.session.edit_prefs(|prefs| prefs.generative_remove = saved);
        tick(&mut restarted, &egui::Context::default());
        assert!(restarted.ui.dialogs.iter().any(|dialog| owns(&dialog.fields)), "saved incomplete choice is recoverable after restart");
        assert!(!restarted.session.has_jobs());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_ux_migrated_custom_setup_is_not_replaced_by_a_recovery_prompt() {
        let mut app = recovery_app();
        app.session.edit_prefs(|prefs| {
            prefs.generative_remove.workflow = "custom".into();
            prefs.generative_remove.enabled = true;
        });
        let before = app.session.prefs_value();
        tick(&mut app, &egui::Context::default());
        tick(&mut app, &egui::Context::default());
        assert!(app.ui.dialogs.is_empty());
        assert_eq!(app.session.prefs_value(), before);
        assert!(!app.session.has_jobs());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_ux_ready_profile_metadata_is_checked_only_once_in_the_tick_loop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountReady(Arc<AtomicUsize>);
        impl photocraft_ml::RemovalBackend for CountReady {
            fn ready(&self, _: &str) -> bool {
                self.0.fetch_add(1, Ordering::SeqCst);
                true
            }
            fn infer(
                &self,
                _: &str,
                _: &photocraft_ml::RemovalInput,
                _: &dyn photocraft_ml::RemovalControl,
            ) -> photocraft_ml::Result<photocraft_ml::RemovalOutput> {
                Err(photocraft_ml::Error::Inference("setup must not infer".into()))
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut app = recovery_app();
        app.session.set_removal_service(Arc::new(
            photocraft_ml::RemovalService::new(Some(std::env::temp_dir().join("photocraft-model-ux-ready"))).with_backend(Arc::new(CountReady(calls.clone()))),
        ));
        app.run("remove.setup", json!({"workflow":"lama"})).unwrap();
        tick(&mut app, &egui::Context::default());
        let after_first = calls.load(Ordering::SeqCst);
        assert!(after_first > 0);
        for _ in 0..8 {
            tick(&mut app, &egui::Context::default());
        }
        assert_eq!(calls.load(Ordering::SeqCst), after_first, "ready profiles need no per-frame directory checks");
        assert!(app.ui.dialogs.is_empty());
    }

    #[test]
    fn model_ux_missing_download_still_requires_both_license_and_command_authority() {
        let mut app = app();
        let status = ux_status(&[]);
        let mut fields = json!({"__prefsui":"removeSetup","workflow":"lama","acceptLicenses":false}).as_object().unwrap().clone();
        assert!(authorize_setup(&app, &fields, &status).unwrap_err().contains("licenses"));
        fields.insert("acceptLicenses".into(), json!(true));
        app.session.authorize =
            Some(|command, _| if command == "ai.models.install" { Err(photocraft_engine::EngineError::Other("download denied".into())) } else { Ok(()) });
        let before = app.session.prefs_value();
        assert!(authorize_setup(&app, &fields, &status).unwrap_err().contains("download denied"));
        assert_eq!(app.session.prefs_value(), before);
        assert!(!busy(&app));
        assert!(!app.session.has_jobs());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_ux_setup_progress_names_each_component_and_only_final_completion_is_ready() {
        let mut app = app();
        let runtime = &mut app.prefs_rt.remove_setup;
        runtime.workflow = "recommended".into();
        runtime.total = 2;
        runtime.active_model = Some("lama".into());
        runtime.queue.push_back("osor-sdxl".into());
        let first = progress_message(&app, "Downloading");
        assert!(first.contains("LaMa") && first.contains("1 of 2"));
        app.prefs_rt.remove_setup.message = first;
        model_finished(&mut app);
        assert!(!app.prefs_rt.remove_setup.message.contains("setup is ready"));
        app.prefs_rt.remove_setup.queue.pop_front();
        app.prefs_rt.remove_setup.active_model = Some("osor-sdxl".into());
        let second = progress_message(&app, "Preparing");
        assert!(second.contains("OSOR") && second.contains("2 of 2"));
        model_finished(&mut app);
        assert!(app.prefs_rt.remove_setup.message.contains("setup is ready"));
        assert!(!app.prefs_rt.remove_setup.error);
    }

    #[test]
    fn model_ux_ready_setup_copy_omits_download_and_preparation_requirements() {
        use egui_kittest::{Harness, kittest::Queryable};
        let status = ux_status(&["lama", "osor-sdxl"]);
        let fields = json!({"workflow":"recommended","memoryUsageMb":49152}).as_object().unwrap().clone();
        let mut harness = Harness::builder().with_size(egui::vec2(700.0, 900.0)).build_ui_state(
            |ui, state: &mut (PhotocraftApp, Map<String, Value>, Value)| render(&mut state.0, ui, &mut state.1, &state.2),
            (app(), fields, status),
        );
        harness.run_steps(3);
        assert!(harness.query_by_label("OSOR is installed (about 10.6 GB). No OSOR download or preparation is needed.").is_some());
        assert!(
            harness
                .query_by_label("OSOR will be prepared after downloading. Allow about 19.1 GB of free disk space; installed OSOR weights use about 10.6 GB.")
                .is_none()
        );
        assert_eq!(harness.state().1["__download"], false);
        assert_eq!(confirm_label(&harness.state().1), "Use this setup");
    }

    #[test]
    #[ignore = "offscreen UI evidence; set PHOTOCRAFT_SETUP_SNAPSHOT and optionally PHOTOCRAFT_SETUP_SNAPSHOT_STATE=fresh|partial|ready"]
    fn model_ux_setup_state_snapshot() {
        let _gpu = photocraft_testkit::gpu_lock();
        let path = std::env::var("PHOTOCRAFT_SETUP_SNAPSHOT").unwrap();
        let state = std::env::var("PHOTOCRAFT_SETUP_SNAPSHOT_STATE").unwrap_or_else(|_| "fresh".into());
        let installed: &[&str] = match state.as_str() {
            "partial" => &["lama"],
            "ready" => &["lama", "osor-sdxl"],
            _ => &[],
        };
        let fields = json!({"workflow":"recommended","memoryUsageMb":49152}).as_object().unwrap().clone();
        let status = ux_status(installed);
        let size = if std::env::var_os("PHOTOCRAFT_SETUP_SMALL_SNAPSHOT").is_some() { egui::vec2(650.0, 600.0) } else { egui::vec2(800.0, 900.0) };
        let mut harness = egui_kittest::Harness::builder().with_size(size).wgpu().build_ui_state(
            |ui, state: &mut (PhotocraftApp, Map<String, Value>, Value)| {
                ui.set_width(ui.available_width().min(620.0));
                ui.heading("Set Up Generative Remove");
                ui.separator();
                let height = (ui.ctx().content_rect().height() - 160.0).max(160.0);
                egui::ScrollArea::vertical().max_height(height).show(ui, |ui| render(&mut state.0, ui, &mut state.1, &state.2));
                ui.separator();
                ui.horizontal(|ui| {
                    let enabled = validation(&state.0, &state.1, &state.2).is_ok();
                    let _ = ui.add_enabled(enabled, egui::Button::new(confirm_label(&state.1)));
                    let _ = ui.button("Cancel");
                });
            },
            (app(), fields, status),
        );
        PhotocraftApp::setup_context(&harness.ctx, Default::default());
        harness.run_steps(4);
        harness.render().unwrap().save(path).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_ux_recommended_choice_preserves_the_users_effect_scope() {
        for effects in [false, true] {
            let mut app = app();
            app.ui.tool_options.remove_effects = effects;
            let fields = json!({"workflow":"recommended","memoryUsageMb":49152}).as_object().unwrap().clone();
            confirm(&mut app, &fields).unwrap();
            assert_eq!(app.ui.tool_options.remove_effects, effects, "model choice must not expand the edited area");
            assert!(!app.session.has_jobs());
            confirm(&mut app, json!({"workflow":"lama"}).as_object().unwrap()).unwrap();
            assert!(!app.ui.tool_options.remove_effects, "non-OSOR profiles cannot request unsupported effects");
        }
    }
}
