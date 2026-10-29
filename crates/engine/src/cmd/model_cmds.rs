//! Local removal model discovery and explicit, authorized administration.

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};
use serde_json::{Value, json};
use std::path::PathBuf;

fn bad(cmd: &str, message: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: message.into() }
}

pub(crate) fn model_id<'a>(s: &Session, p: &'a Value, cmd: &str) -> Result<&'a str> {
    let id =
        p.get("modelId").and_then(Value::as_str).filter(|id| !id.is_empty() && id.len() <= 64).ok_or_else(|| bad(cmd, "`modelId` must be a known model id"))?;
    if !s.removal_service().catalog().iter().any(|model| model.id == id) {
        return Err(bad(cmd, format!("unknown removal model `{id}`")));
    }
    Ok(id)
}

/// Model administration always requires an explicit direct user invocation.
pub fn is_management(id: &str) -> bool {
    matches!(id, "ai.models.install" | "ai.models.import" | "ai.models.remove" | "remove.setup")
}

pub(crate) fn refuse_playback(s: &Session, id: &str) -> Result<()> {
    if is_management(id) && (s.actions.playing > 0 || s.actions.script_depth > 0) {
        return Err(bad(id, "model management is unavailable during action or script playback"));
    }
    Ok(())
}

fn management_allowed(s: &Session) -> bool {
    let p = json!({"modelId":"lama", "path":"model.onnx"});
    s.actions.playing == 0
        && s.actions.script_depth == 0
        && ["ai.models.install", "ai.models.import", "ai.models.remove"]
            .iter()
            .all(|id| s.authorize.is_none_or(|gate| gate(id, &p).is_ok()) && crate::jobs::authorize_render_command(id, &p).is_ok())
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let service = s.removal_service();
    let runtime = service.runtime_available();
    let preferences = &s.prefs().generative_remove;
    let statuses = service.status();
    let models: Vec<Value> = service
        .catalog()
        .iter()
        .map(|spec| {
            let status = statuses.iter().find(|status| status.id == spec.id);
            json!({
                "id":spec.id, "label":spec.label, "description":spec.description, "bytes":spec.bytes,
                "sha256":spec.sha256, "url":spec.url, "license":spec.license, "source":spec.source,
                "backend":spec.backend, "inputSize":spec.input_size, "memoryMib":spec.memory_mib,
                "downloadBytes":spec.download_bytes, "installBytes":spec.install_bytes,
                "preparationWorkspaceBytes":spec.preparation_workspace_bytes, "preparationMemoryMib":spec.preparation_memory_mib,
                "licenseUrl":spec.license_url, "artifacts":spec.artifacts,
                // `available` means this build supports the adapter, so an uninstalled model can be
                // installed. A size-matching artifact can be attempted but still needs verification.
                "available":service.model_runtime_available(spec.id), "canAttempt":status.is_some_and(|v| v.available), "autoQualified":false,
                "artifactKind":if spec.id == "osor-sdxl" { "bundle" } else { "file" }, "canDownload":!spec.url.is_empty() || !spec.artifacts.is_empty(),
                "installed":status.is_some_and(|v| v.installed), "verified":status.is_some_and(|v| v.verified),
                "busy":status.is_some_and(|v| v.busy), "reason":status.and_then(|v| v.reason.as_deref()),
                "enabled":preferences.enabled_models.iter().any(|id| id == spec.id),
                "selected":preferences.selected_model == spec.id,
            })
        })
        .collect();
    Ok(json!({"runtimeAvailable":runtime, "managementAvailable":service.management_available() && management_allowed(s),
        "modelDirectory":service.model_directory().map(|path| path.to_string_lossy()), "enabled":preferences.enabled, "workflow":preferences.workflow, "setupComplete":preferences.setup_complete,
        "componentIds":workflow_models(&preferences.workflow, &preferences.selected_model), "models":models, "lastExecution":service.last_execution()}))
}

pub(crate) fn workflow_models<'a>(workflow: &str, custom: &'a str) -> Vec<&'a str> {
    match workflow {
        "recommended" => vec!["lama", "osor-sdxl"],
        "lama" => vec!["lama"],
        "compact" => vec!["migan-v1"],
        "classical" => Vec::new(),
        _ => vec![custom],
    }
}

fn setup(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "remove.setup";
    refuse_playback(s, CMD)?;
    let object = p.as_object().ok_or_else(|| bad(CMD, "params must be an object"))?;
    if object.keys().any(|key| !["workflow", "memoryUsageMb"].contains(&key.as_str())) {
        return Err(bad(CMD, "only workflow and optional memoryUsageMb are accepted; downloads require ai.models.install"));
    }
    let workflow = p
        .get("workflow")
        .and_then(Value::as_str)
        .filter(|id| matches!(*id, "recommended" | "lama" | "compact" | "classical" | "custom"))
        .ok_or_else(|| bad(CMD, "unknown Remove workflow"))?
        .to_string();
    let budget = p
        .get("memoryUsageMb")
        .map(|value| value.as_u64().filter(|n| (256..=1_048_576).contains(n)).ok_or_else(|| bad(CMD, "memoryUsageMb must be within 256..1048576")))
        .transpose()?;
    let models = workflow_models(&workflow, &s.prefs().generative_remove.selected_model).into_iter().map(str::to_owned).collect::<Vec<_>>();
    s.edit_prefs(|prefs| {
        prefs.generative_remove.workflow = workflow.clone();
        prefs.generative_remove.setup_complete = true;
        if workflow != "custom" {
            prefs.generative_remove.enabled = workflow != "classical";
            prefs.generative_remove.enabled_models = models;
            if let Some(model) = prefs.generative_remove.enabled_models.first() {
                prefs.generative_remove.selected_model = model.clone();
            }
        }
        if let Some(budget) = budget {
            prefs.performance.memory_usage_mb = budget as u32;
        }
    });
    capabilities(s, &json!({}))
}

fn capabilities(s: &mut Session, p: &Value) -> Result<Value> {
    if p.get("modelId").is_some() {
        model_id(s, p, "remove.capabilities")?;
    }
    let mut result = list(s, p)?;
    let target_reason = crate::retouch_cmds::removal_context_reason(s, p);
    let prefs = &s.prefs().generative_remove;
    let service = s.removal_service();
    let components = if let Some(id) = p.get("modelId").and_then(Value::as_str) { vec![id] } else { workflow_models(&prefs.workflow, &prefs.selected_model) };
    let statuses = service.status();
    let reason = if components.is_empty() {
        None
    } else if !prefs.enabled {
        Some("local AI processing is disabled".to_string())
    } else if let Some(reason) = target_reason.clone() {
        Some(reason)
    } else {
        components.iter().find_map(|id| {
            if !prefs.enabled_models.iter().any(|model| model == id) {
                Some(format!("removal model `{id}` is disabled"))
            } else if !service.model_runtime_available(id) {
                Some(format!("local inference for `{id}` is unavailable in this build"))
            } else {
                match statuses.iter().find(|status| status.id == *id) {
                    Some(status) if status.busy => Some("local AI is busy".into()),
                    Some(status) if status.available => None,
                    Some(status) => Some(format!("{id}: {}", status.reason.as_deref().unwrap_or("model is not installed"))),
                    None => Some(format!("unknown removal model `{id}`")),
                }
            }
        })
    };
    if let Some(object) = result.as_object_mut() {
        object.insert(
            "workflows".into(),
            json!([
                {"id":"recommended","label":"Recommended — LaMa + OSOR","componentIds":["lama","osor-sdxl"],"previewRequired":true},
                {"id":"lama","label":"LaMa only","componentIds":["lama"]},
                {"id":"compact","label":"Compact","componentIds":["migan-v1"]},
                {"id":"classical","label":"Classical","componentIds":[]},
                {"id":"custom","label":"Custom","componentIds":[prefs.selected_model]}
            ]),
        );
        object.insert("routingHeuristic".into(), json!("Auto routes elongated footprints (at least 3:1, minor extent at most 64 pixels) to LaMa; compact footprints to OSOR. Detail/Object overrides always win."));
        object.insert("targetCompatible".into(), json!(target_reason.is_none()));
        object.insert("compatible".into(), json!(reason.is_none()));
        object.insert("compatibilityReason".into(), json!(reason));
        object.insert("contextRequirements".into(), json!("The sampled crop must be opaque and valid sRGB."));
    }
    Ok(result)
}

fn ml_error(error: photocraft_ml::Error) -> EngineError {
    if matches!(error, photocraft_ml::Error::Cancelled) { EngineError::Cancelled } else { EngineError::Other(error.to_string()) }
}

fn manage(s: &mut Session, p: &Value, cmd: &'static str) -> Result<Value> {
    refuse_playback(s, cmd)?;
    if let Some(params) = p.as_object() {
        let allowed: &[&str] = if cmd == "ai.models.import" { &["modelId", "path"] } else { &["modelId"] };
        if let Some(key) = params.keys().find(|key| !allowed.contains(&key.as_str())) {
            let shown: String = key.chars().take(64).collect();
            return Err(bad(cmd, format!("unknown model option `{shown}`")));
        }
    }
    let id = model_id(s, p, cmd)?.to_string();
    let path = if cmd == "ai.models.import" {
        let path = p
            .get("path")
            .and_then(Value::as_str)
            .filter(|path| !path.is_empty() && path.len() <= 4096)
            .ok_or_else(|| bad(cmd, "`path` must be a nonempty model file or bundle directory path"))?;
        Some(PathBuf::from(path))
    } else {
        None
    };
    let service = s.removal_service();
    let name = service.catalog().iter().find(|model| model.id == id).map_or(id.as_str(), |model| model.label);
    let label = match cmd {
        "ai.models.install" if id == "osor-sdxl" => format!("Download and prepare {name}"),
        "ai.models.install" => format!("Download {name}"),
        "ai.models.import" => format!("Import {name}"),
        _ => format!("Remove {name}"),
    };
    let budget_bytes = u64::from(s.prefs().performance.memory_usage_mb).saturating_mul(1024 * 1024);
    let result_id = id.clone();
    crate::jobs::run(
        s,
        &label,
        false,
        move |ctx| {
            ctx.check()?;
            match cmd {
                "ai.models.install" => service.install_with_memory_budget(&id, budget_bytes, ctx).map_err(ml_error)?,
                "ai.models.import" => service.import(&id, path.as_deref().ok_or_else(|| bad(cmd, "missing path"))?, ctx).map_err(ml_error)?,
                "ai.models.remove" => {
                    // Once unlink begins, do not claim cancellation when the file was removed.
                    ctx.begin_irreversible()?;
                    service.remove(&id).map_err(ml_error)?;
                }
                _ => return Err(bad(cmd, "unknown model operation")),
            }
            Ok(())
        },
        move |_, ()| Ok(json!({"modelId":result_id, "operation":cmd})),
    )
}

pub fn specs() -> Vec<CommandSpec> {
    let spec = |id, label, params, run| CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: |_| Ok(()), run, journal: false };
    vec![
        spec("ai.models.list", "Local AI Models", r#"{} → {runtimeAvailable,managementAvailable,models}"#, list),
        spec(
            "remove.setup",
            "Set Up Remove",
            r#"{"workflow":"recommended|lama|compact|classical|custom","memoryUsageMb":u32?} → persist explicit choice; no downloads"#,
            setup,
        ),
        spec("remove.capabilities", "Remove Capabilities", r#"{"modelId":catalog id?} → availability and target requirements"#, capabilities),
        spec("ai.models.install", "Download Local Model", r#"{"modelId":catalog id}"#, |s, p| manage(s, p, "ai.models.install")),
        spec("ai.models.import", "Import Local Model", r#"{"modelId":catalog id,"path":local model file or bundle directory path}"#, |s, p| {
            manage(s, p, "ai.models.import")
        }),
        spec("ai.models.remove", "Remove Local Model", r#"{"modelId":catalog id}"#, |s, p| manage(s, p, "ai.models.remove")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(unix)]
    #[test]
    fn model_folder_status_handles_non_unicode_paths_without_panicking() {
        use std::os::unix::ffi::OsStringExt;
        let folder = std::env::temp_dir().join(std::ffi::OsString::from_vec(b"photocraft-models-\xff".to_vec()));
        let mut session = Session::new();
        session.set_removal_service(Arc::new(photocraft_ml::RemovalService::new(Some(folder.clone()))));
        let status = session.execute("ai.models.list", json!({})).unwrap();
        assert_eq!(status["modelDirectory"], folder.to_string_lossy().as_ref());
        assert!(!session.has_jobs());
    }

    struct CountingDownloader(AtomicUsize);
    impl photocraft_ml::ModelDownloader for CountingDownloader {
        fn open(&self, _: &photocraft_ml::ModelSpec) -> photocraft_ml::Result<Box<dyn std::io::Read + Send>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(std::io::empty()))
        }
    }

    #[test]
    fn model_queries_preserve_fade_history_and_report_authority() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new();
        s.set_removal_service(Arc::new(photocraft_ml::RemovalService::new(Some(dir.path().into()))));
        s.execute("file.new", json!({"width":32,"height":32})).unwrap();
        s.execute("edit.fill", json!({})).unwrap();
        assert!(s.is_enabled("edit.fade"));
        let past = s.active().unwrap().history.past_len();
        let journal = s.journal.clone();
        let list = s.execute("ai.models.list", json!({})).unwrap();
        assert_eq!(list["managementAvailable"], true);
        assert!(list["models"].as_array().unwrap().iter().any(|m| m["id"] == "lama" && m["installed"] == false));
        assert!(s.execute("remove.capabilities", json!({})).is_ok());
        assert!(s.is_enabled("edit.fade"));
        assert_eq!(s.active().unwrap().history.past_len(), past);
        assert_eq!(s.journal, journal);
        s.authorize = Some(|id, _| if is_management(id) { Err(EngineError::Other("denied".into())) } else { Ok(()) });
        let untrusted = s.execute("ai.models.list", json!({})).unwrap();
        assert_eq!(untrusted["managementAvailable"], false);
        assert!(s.execute("ai.models.remove", json!({"modelId":"lama"})).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn model_download_job_names_its_model_for_reopened_progress() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new();
        s.set_removal_service(Arc::new(
            photocraft_ml::RemovalService::new(Some(dir.path().into())).with_downloader(Arc::new(CountingDownloader(AtomicUsize::new(0)))),
        ));
        let crate::jobs::Started::Job(job) = s.start("ai.models.install", json!({"modelId":"lama"})).unwrap() else {
            panic!("model transfer should start a background job");
        };
        assert_eq!(s.job(job).unwrap().label, "Download LaMa");
        assert!(s.wait_job(job).is_err(), "the empty fixture stream cannot publish a model");
    }

    #[test]
    fn model_management_is_refused_in_playback_before_files_or_network() {
        let dir = tempfile::tempdir().unwrap();
        let downloader = Arc::new(CountingDownloader(AtomicUsize::new(0)));
        let mut s = Session::new();
        s.set_removal_service(Arc::new(photocraft_ml::RemovalService::new(Some(dir.path().into())).with_downloader(downloader.clone())));
        for script in [false, true] {
            s.actions.playing = u8::from(!script);
            s.actions.script_depth = u8::from(script);
            for id in ["ai.models.install", "ai.models.import", "ai.models.remove"] {
                let error = s.execute(id, json!({"modelId":"lama", "path":"/not-a-model"})).unwrap_err().to_string();
                assert!(error.contains("playback"), "{id}: {error}");
            }
        }
        s.actions.playing = 0;
        s.actions.script_depth = 0;
        assert!(s.execute("ai.models.install", json!({"modelId":"lama", "url":"https://example.invalid/"})).is_err());
        assert_eq!(downloader.0.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    #[test]
    fn batch_refuses_model_management_before_reading_inputs_or_writing_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("output");
        let mut s = Session::new();
        let error = s
            .execute(
                "file.automate.batch",
                json!({"steps":[["ai.models.install",{"modelId":"lama"}]],
            "input":dir.path().join("missing"),"output":output}),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("model management"), "{error}");
        assert!(!output.exists());
    }
}
