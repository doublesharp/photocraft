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
    matches!(id, "ai.models.install" | "ai.models.import" | "ai.models.remove")
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
                // `available` means this build supports the adapter, so an uninstalled model can be
                // installed. A size-matching artifact can be attempted but still needs verification.
                "available":runtime, "canAttempt":status.is_some_and(|v| v.available), "autoQualified":false,
                "installed":status.is_some_and(|v| v.installed), "verified":status.is_some_and(|v| v.verified),
                "busy":status.is_some_and(|v| v.busy), "reason":status.and_then(|v| v.reason.as_deref()),
                "enabled":preferences.enabled_models.iter().any(|id| id == spec.id),
                "selected":preferences.selected_model == spec.id,
            })
        })
        .collect();
    Ok(json!({"runtimeAvailable":runtime, "managementAvailable":service.management_available() && management_allowed(s),
        "enabled":preferences.enabled, "models":models}))
}

fn capabilities(s: &mut Session, p: &Value) -> Result<Value> {
    if p.get("modelId").is_some() {
        model_id(s, p, "remove.capabilities")?;
    }
    let mut result = list(s, p)?;
    let target_reason = crate::retouch_cmds::removal_context_reason(s, p);
    let prefs = &s.prefs().generative_remove;
    let id = p.get("modelId").and_then(Value::as_str).unwrap_or(&prefs.selected_model);
    let service = s.removal_service();
    let reason = if !prefs.enabled {
        Some("local AI processing is disabled".to_string())
    } else if !prefs.enabled_models.iter().any(|model| model == id) {
        Some("the selected removal model is disabled".to_string())
    } else if !service.runtime_available() {
        Some("local AI inference is unavailable in this build".to_string())
    } else if let Some(reason) = target_reason.clone() {
        Some(reason)
    } else if let Some(status) = service.status().into_iter().find(|status| status.id == id) {
        if status.busy {
            Some("local AI is busy".into())
        } else if !status.available {
            Some(status.reason.unwrap_or_else(|| "model is not installed".into()))
        } else {
            None
        }
    } else {
        Some("the selected removal model is unknown".into())
    };
    if let Some(object) = result.as_object_mut() {
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
            .ok_or_else(|| bad(cmd, "`path` must be a nonempty model file path"))?;
        Some(PathBuf::from(path))
    } else {
        None
    };
    let service = s.removal_service();
    let label = match cmd {
        "ai.models.install" => "Download model",
        "ai.models.import" => "Import model",
        _ => "Remove model",
    };
    let result_id = id.clone();
    crate::jobs::run(
        s,
        label,
        false,
        move |ctx| {
            ctx.check()?;
            match cmd {
                "ai.models.install" => service.install(&id, ctx).map_err(ml_error)?,
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
        spec("remove.capabilities", "Remove Capabilities", r#"{"modelId":catalog id?} → availability and target requirements"#, capabilities),
        spec("ai.models.install", "Download Local Model", r#"{"modelId":catalog id}"#, |s, p| manage(s, p, "ai.models.install")),
        spec("ai.models.import", "Import Local Model", r#"{"modelId":catalog id,"path":local file path}"#, |s, p| manage(s, p, "ai.models.import")),
        spec("ai.models.remove", "Remove Local Model", r#"{"modelId":catalog id}"#, |s, p| manage(s, p, "ai.models.remove")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
