//! Optional local Remove models. Downloads and processing are explicit user choices.

use super::{DialogKind, Map, PhotocraftApp, RichText, Tokens, Value, json};
use crate::file_dialog::{FileDialogAnswer, FileDialogRequest};

/// Open the model manager without installing models or changing processing preferences.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    if let Some(dialog) = app.ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("removeModels")) {
        return dialog.id;
    }
    let fields = json!({"__prefsui": "removeModels", "__label": "Generative Remove Models", "message": ""});
    app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default())
}

fn flag(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn remember_result(f: &mut Map<String, Value>, result: Result<Value, String>) {
    match result {
        Ok(result) => {
            let message = if let Some(job) = result.get("job").and_then(Value::as_u64) {
                f.insert("job".into(), json!(job));
                tl!("The model operation continues while this window is closed.")
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

fn preference(app: &mut PhotocraftApp, path: &str, value: Value) -> Result<Value, String> {
    app.run("prefs.set", json!({"path": path, "value": value}))
}

// Model transfers can take minutes. Start them as jobs even if a host normally runs other
// commands inline; the browser has no model-management capability and never starts one.
fn model_command(app: &mut PhotocraftApp, id: &str, params: Value) -> Result<Value, String> {
    let background = app.background_jobs;
    app.background_jobs = true;
    let result = app.run(id, params);
    app.background_jobs = background;
    result
}

fn enabled_model(app: &mut PhotocraftApp, id: &str, enabled: bool) -> Result<Value, String> {
    let mut ids = app.session.prefs().generative_remove.enabled_models.clone();
    ids.retain(|value| value != id);
    if enabled {
        ids.push(id.to_string());
    }
    preference(app, "generativeRemove.enabledModels", json!(ids))
}

/// Ask for the artifact's path only: hashing, copying and validation run on the engine worker.
fn import_model(app: &mut PhotocraftApp, id: &str) -> Result<Value, String> {
    let id = id.to_string();
    app.ask_file(FileDialogRequest::Open { multiple: false, initial_dir: None, extensions: Some(vec!["onnx".into()]) }, move |app, answer| {
        let result = match answer {
            FileDialogAnswer::Paths(paths) => match paths.first() {
                Some(path) => model_command(app, "ai.models.import", json!({"modelId": id, "path": path})),
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

pub(super) fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let status = match app.session.execute("ai.models.list", json!({})) {
        Ok(status) => status,
        Err(error) => {
            ui.label(RichText::new(error.to_string()).color(t.text_dim));
            return;
        }
    };
    render(app, ui, f, &status);
}

fn render(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, status: &Value) {
    let t = Tokens::get(ui.ctx());
    ui.label(tl!("Choose which local models may process your images. A brush stroke never downloads a model."));
    ui.add_space(6.0);
    let available = flag(status, "runtimeAvailable");
    let management = flag(status, "managementAvailable");
    let prefs = app.session.prefs().generative_remove.clone();
    let mut enabled = prefs.enabled;
    let enabled_ids = prefs.enabled_models;
    let selected_id = prefs.selected_model;
    ui.add_enabled_ui(available, |ui| {
        if crate::widgets::checkbox(ui, &mut enabled, tl!("Enable local AI removal")).changed() {
            let result = preference(app, "generativeRemove.enabled", json!(enabled));
            remember_result(f, result);
        }
    });
    if !available {
        ui.label(RichText::new(tl!("Local AI inference is unavailable in this build. Classical Remove remains available.")).color(t.text_dim));
    }
    if !management {
        ui.label(RichText::new(tl!("Model installation is unavailable in this session.")).color(t.text_dim));
    }
    ui.add_space(6.0);
    let models = status.get("models").and_then(Value::as_array).cloned().unwrap_or_default();
    egui::ScrollArea::vertical().max_height((ui.available_height() - 110.0).clamp(160.0, 420.0)).id_salt("remove-models-list").show(ui, |ui| {
        for model in models {
            let id = text(&model, "id");
            ui.push_id(id, |ui| {
                ui.group(|ui| {
                    ui.set_min_width(ui.available_width());
                    let installed = flag(&model, "installed");
                    let busy = flag(&model, "busy");
                    let mut model_enabled = enabled_ids.iter().any(|value| value == id);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(text(&model, "label")).strong());
                        let bytes = model.get("bytes").and_then(Value::as_u64).unwrap_or(0);
                        ui.label(RichText::new(format!("{:.0} MB", bytes as f64 / 1_000_000.0)).color(t.text_dim));
                        let readiness = if busy {
                            tl!("In use")
                        } else if installed {
                            tl!("Installed")
                        } else {
                            tl!("Not installed")
                        };
                        ui.label(RichText::new(readiness).color(t.text_dim));
                    });
                    ui.label(tl!(text(&model, "description")));
                    ui.add_enabled_ui(available, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if crate::widgets::checkbox(ui, &mut model_enabled, tl!("Enable this process")).changed() {
                                let result = enabled_model(app, id, model_enabled);
                                remember_result(f, result);
                            }
                            ui.add_enabled_ui(model_enabled, |ui| {
                                if ui.radio(selected_id == id, tl!("Selected model")).clicked() {
                                    let result = preference(app, "generativeRemove.selectedModel", json!(id));
                                    remember_result(f, result);
                                }
                            });
                        });
                    });
                    let reason = text(&model, "reason");
                    // Global runtime unavailability and a missing install are already named
                    // above; keep the card compact while retaining other model-specific reasons.
                    let duplicate = reason == "model is not installed"
                        || (!available && matches!(reason, "native AI inference is unavailable in this build" | "AI removal is unavailable in the browser"));
                    if !reason.is_empty() && !duplicate {
                        ui.label(RichText::new(tl!(reason)).color(t.text_dim));
                    }
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(text(&model, "license")).color(t.text_dim));
                        let source = text(&model, "source");
                        if !source.is_empty() {
                            ui.hyperlink_to(tl!("Model source and license"), source);
                        }
                        let artifact = text(&model, "url");
                        if !artifact.is_empty() {
                            ui.hyperlink_to(tl!("Model download source"), artifact);
                        }
                    });
                    // An uninstalled model is intentionally eligible for installation.
                    ui.add_enabled_ui(available && management && !busy, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if !installed && ui.button(tl!("Download")).clicked() {
                                let result = model_command(app, "ai.models.install", json!({"modelId": id}));
                                remember_result(f, result);
                            }
                            if ui.add_enabled(app.services.file_dialog.is_some(), egui::Button::new(tl!("Install from file…"))).clicked() {
                                let result = import_model(app, id);
                                remember_result(f, result);
                            }
                            if installed && ui.button(tl!("Remove model")).clicked() {
                                let result = model_command(app, "ai.models.remove", json!({"modelId": id}));
                                remember_result(f, result);
                            }
                        });
                    });
                });
                ui.add_space(6.0);
            });
        }
    });
    for job in app.session.jobs().into_iter().filter(|job| job.command.starts_with("ai.models.")) {
        ui.horizontal_wrapped(|ui| {
            ui.label(tl!(&job.label));
            ui.add(egui::ProgressBar::new(job.progress).desired_width(160.0).show_percentage());
            if ui.button(tl!("Cancel model operation")).clicked() {
                crate::jobs_ui::cancel(app, job.id);
            }
        });
        if !job.message.is_empty() {
            ui.label(RichText::new(tl!(&job.message)).color(t.text_dim));
        }
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
    if let Some(id) = f.get("job").and_then(Value::as_u64)
        && let Some(job) = app.session.jobs_with_recent().into_iter().find(|job| job.id.0 == id)
    {
        let message = match job.state {
            "failed" => job.error,
            "cancelled" => Some(tl!("Model installation cancelled").to_string()),
            "done" if job.command == "ai.models.remove" => Some(tl!("Model removed").to_string()),
            "done" => Some(tl!("Model installed and verified").to_string()),
            _ => None,
        };
        if let Some(message) = message {
            f.insert("message".into(), json!(message));
            f.insert("error".into(), json!(job.state == "failed"));
            f.remove("job");
        }
    }
    if let Some(message) = f.get("message").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        let color = if f.get("error").and_then(Value::as_bool) == Some(true) { t.danger } else { t.text_dim };
        ui.label(RichText::new(message).color(color));
    } else {
        ui.label(RichText::new(tl!("Settings take effect immediately. Downloads continue while this window is closed.")).color(t.text_dim));
    }
}

#[cfg(test)]
mod tests {
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
                "installed":false, "available":false, "busy":false
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
        assert!(!h.get_by_label("Download").accesskit_node().is_disabled(), "an absent model can be installed");
        h.get_by_label("Enable this process").click();
        h.run_steps(3);
        assert_eq!(h.state().app.session.prefs_value().pointer("/generativeRemove/enabledModels"), Some(&json!(["lama", "migan-v1"])));
        h.get_by_label("Selected model").click();
        h.run_steps(3);
        assert_eq!(h.state().app.session.prefs_value().pointer("/generativeRemove/selectedModel"), Some(&json!("migan-v1")));
        assert!(!h.state().app.session.has_jobs(), "selecting a model does not download it");
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
        assert_eq!(h.state().fields["message"], "Model installation cancelled");
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
}
