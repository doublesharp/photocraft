//! Restore the user's authorized local Remove choices for explicit CLI processing only.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[path = "../../photocraft/src/app_dirs/locations.rs"]
mod locations;

/// Use the desktop's authoritative directory, including its portable-mode fallback.
fn preferences(env: impl Fn(&str) -> Option<OsString>, exe: Option<&Path>) -> Option<PathBuf> {
    let resolved = locations::resolve(env, exe.and_then(Path::parent));
    if resolved.mode == locations::Mode::Platform
        && let Some(warning) = &resolved.warning
    {
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr().lock(), "photocraft: {warning}");
    }
    resolved.dir.map(|config| config.join("preferences.json"))
}

/// Read only the user's local processing choices; remote automation never calls this path.
fn load_processing_preferences(session: &mut photocraft_engine::Session, reader: impl std::io::Read) -> Result<(), String> {
    use std::io::Read as _;
    const MAX_PREFERENCES_BYTES: u64 = 16 * 1024 * 1024;
    let mut bytes = Vec::new();
    reader.take(MAX_PREFERENCES_BYTES + 1).read_to_end(&mut bytes).map_err(|error| format!("reading local processing preferences: {error}"))?;
    if bytes.len() as u64 > MAX_PREFERENCES_BYTES {
        return Err("local processing preferences exceed the 16 MiB limit".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| format!("local processing preferences: {error}"))?;
    if !value.is_object() {
        return Err("local processing preferences must be a JSON object".into());
    }
    if value.get("performance").is_some_and(|performance| !performance.is_object()) {
        return Err("local processing preferences: performance must be a JSON object".into());
    }
    // Validate only this explicit subset. Script hooks, paths, history logging and the
    // remaining desktop preferences must not acquire authority in a headless batch.
    let mut validated = photocraft_engine::Session::new();
    for (path, value) in [("generativeRemove", value.get("generativeRemove")), ("performance.memoryUsageMb", value.pointer("/performance/memoryUsageMb"))] {
        if let Some(value) = value {
            validated.execute("prefs.set", serde_json::json!({"path":path,"value":value})).map_err(|error| error.to_string())?;
        }
    }
    let removal = validated.prefs().generative_remove.clone();
    let memory = validated.prefs().performance.memory_usage_mb;
    session.edit_prefs(|prefs| {
        prefs.generative_remove = removal;
        prefs.performance.memory_usage_mb = memory;
    });
    Ok(())
}

fn load_local_processing_preferences(session: &mut photocraft_engine::Session) -> Result<(), String> {
    let executable = std::env::current_exe().ok();
    let Some(path) = preferences(|key| std::env::var_os(key), executable.as_deref()) else { return Ok(()) };
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("local processing preferences {}: {error}", path.display())),
    };
    if !metadata.is_file() {
        return Err("local processing preferences must be a regular file".into());
    }
    let file = std::fs::File::open(&path).map_err(|error| format!("local processing preferences {}: {error}", path.display()))?;
    load_processing_preferences(session, file)
}

/// Explicit local CLI entry points inherit the user's processing choices and saved model folder.
/// Remote automation uses model_services::configure without reading these preferences.
pub(crate) fn configure(session: &mut photocraft_engine::Session) -> Result<(), String> {
    load_local_processing_preferences(session)?;
    crate::model_services::configure(session);
    Ok(())
}

#[cfg(test)]
mod preference_tests {
    use super::*;
    #[test]
    fn trusted_cli_processing_preferences_restore_the_saved_model_folder() {
        let folder = std::env::temp_dir().join("photocraft-existing-cli-models");
        let text = serde_json::json!({"generativeRemove":{"modelDirectory":folder,"enabled":true}}).to_string();
        let mut session = photocraft_engine::Session::new();
        load_processing_preferences(&mut session, text.as_bytes()).unwrap();
        assert_eq!(session.prefs().generative_remove.model_directory, folder.to_string_lossy());
        crate::model_services::configure(&mut session);
        let expected = std::env::var_os("PHOTOCRAFT_MODEL_DIR").filter(|path| !path.is_empty()).map(PathBuf::from).unwrap_or(folder);
        assert_eq!(session.removal_service().model_directory(), Some(expected.as_path()));
        assert!(!session.has_jobs());
    }
    #[test]
    fn review_cli_restores_enabled_models_and_memory_without_other_authority() {
        let mut session = photocraft_engine::Session::new();
        let text = serde_json::json!({"version":2,"generativeRemove":{"enabled":true,"enabledModels":["migan-v1"],"selectedModel":"migan-v1"},
            "performance":{"memoryUsageMb":256},"scriptEvents":{"enabled":true},"historyLog":{"enabled":true}})
        .to_string();
        load_processing_preferences(&mut session, text.as_bytes()).unwrap();
        assert!(session.prefs().generative_remove.enabled, "CLI lost the user's processing opt-in");
        assert_eq!(session.prefs().generative_remove.selected_model, "migan-v1");
        assert_eq!(session.prefs().generative_remove.enabled_models, vec!["migan-v1"]);
        assert_eq!(session.prefs().performance.memory_usage_mb, 256);
        assert!(!session.prefs().script_events.enabled, "loading model preferences granted script authority");
    }
}

#[cfg(test)]
mod malformed_preferences_tests {
    use super::*;

    #[test]
    fn malformed_performance_container_cannot_widen_the_configured_budget() {
        for performance in [serde_json::json!(false), serde_json::json!([])] {
            let mut session = photocraft_engine::Session::new();
            session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 256);
            let value = serde_json::json!({"generativeRemove":{"enabled":true},"performance":performance}).to_string();
            let result = load_processing_preferences(&mut session, value.as_bytes());
            assert!(result.is_err(), "a malformed budget container was silently replaced with the default");
            assert_eq!(session.prefs().performance.memory_usage_mb, 256);
            assert!(!session.prefs().generative_remove.enabled, "invalid settings partially enabled processing");
        }
    }
}

#[cfg(test)]
mod portable_preference_tests {
    use super::*;

    #[test]
    fn review2_unwritable_portable_preferences_follow_the_desktop_fallback() {
        let root = std::env::temp_dir().join(format!("photocraft-cli-portable-blocked-{}", std::process::id()));
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("portable.txt"), b"").unwrap();
        std::fs::write(bin.join("PhotoCraftData"), b"not a directory").unwrap();
        let env = |key: &str| match key {
            "HOME" => Some(root.join("home").into_os_string()),
            "APPDATA" => Some(root.join("roaming").into_os_string()),
            "XDG_CONFIG_HOME" => Some(root.join("config").into_os_string()),
            _ => None,
        };
        let expected = preferences(env, None);
        let actual = preferences(env, Some(&bin.join("photocraft")));
        let _ = std::fs::remove_dir_all(root);
        assert_eq!(actual, expected, "CLI selected an unavailable portable configuration instead of the desktop fallback");
    }

    #[cfg(unix)]
    #[test]
    fn review2_readable_stale_portable_preferences_do_not_override_the_fallback() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("photocraft-cli-portable-stale-{}", std::process::id()));
        let bin = root.join("bin");
        let data = bin.join("PhotoCraftData");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(bin.join("portable.txt"), b"").unwrap();
        std::fs::write(data.join("preferences.json"), br#"{"generativeRemove":{"enabled":true},"performance":{"memoryUsageMb":8192}}"#).unwrap();
        let env = |key: &str| match key {
            "HOME" => Some(root.join("home").into_os_string()),
            "APPDATA" => Some(root.join("roaming").into_os_string()),
            "XDG_CONFIG_HOME" => Some(root.join("config").into_os_string()),
            _ => None,
        };
        let expected = preferences(env, None);
        let original = std::fs::metadata(&data).unwrap().permissions();
        std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o555)).unwrap();
        let blocked = std::fs::write(data.join("probe"), b"x").is_err();
        let actual = preferences(env, Some(&bin.join("photocraft")));
        std::fs::set_permissions(&data, original).unwrap();
        let _ = std::fs::remove_dir_all(root);
        assert!(blocked, "fixture must actually reject writes");
        assert_eq!(actual, expected, "CLI restored readable stale portable choices");
    }
}
