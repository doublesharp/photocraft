//! Native model transport and storage used by the desktop and headless CLI.
//! No work happens at startup: resolving a directory and creating an HTTP agent download
//! nothing. Only explicit engine model-management commands open a connection.

use photocraft_ml::{Error, ModelArtifact, ModelDownloader, ModelSpec, RemovalService};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[path = "model_services/directories.rs"]
mod directories;

/// Optional model data is local to this machine, separate from roaming preferences.
/// PHOTOCRAFT_MODEL_DIR explicitly moves it (including to Scratch for development).
pub fn model_dir(saved: &str) -> Option<PathBuf> {
    model_dir_at(saved, shared_store_path().as_deref())
}

/// Profiles and portable copies isolate shared AI settings unless CRAFT_AI_DIR explicitly
/// selects another store. The ordinary desktop profile uses the suite's standard location.
pub fn shared_store_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    directories::shared_store_path(|key| std::env::var_os(key), exe.as_deref())
}

fn model_dir_at(saved: &str, store: Option<&Path>) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let shared = match store {
        Some(path) => RemovalService::shared_model_directory_at(path),
        None => RemovalService::shared_model_directory(),
    }
    .ok()
    .flatten();
    directories::resolve_with_shared(
        |key| std::env::var_os(key),
        exe.as_deref(),
        directories::Platform::current(),
        (!saved.is_empty()).then(|| Path::new(saved)),
        shared.as_deref(),
    )
}

pub fn configure(session: &mut photocraft_engine::Session) {
    let store = shared_store_path();
    let models = model_dir(&session.prefs().generative_remove.model_directory);
    configure_paths(session, models, store);
}

fn configure_paths(session: &mut photocraft_engine::Session, models: Option<PathBuf>, store: Option<PathBuf>) {
    let service = RemovalService::new(models).with_downloader(Arc::new(HttpsModels::new()));
    let service = match store {
        Some(path) => service.with_shared_ai_at(path),
        None => service.with_shared_ai(),
    };
    #[cfg(all(feature = "osor", target_os = "macos"))]
    let service = service.with_osor_backend(photocraft_ml::OsorBackend::Metal { ordinal: 0 });
    session.set_removal_service(Arc::new(service));
}

struct HttpsModels {
    agent: ureq::Agent,
}

impl HttpsModels {
    fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .https_only(true)
            .max_redirects(5)
            .user_agent("Photocraft-dev")
            .timeout_global(Some(Duration::from_secs(3600)))
            .timeout_resolve(Some(Duration::from_secs(10)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_per_read(Some(Duration::from_secs(10)))
            .build();
        Self { agent: config.into() }
    }
}

impl ModelDownloader for HttpsModels {
    fn open(&self, model: &ModelSpec) -> photocraft_ml::Result<Box<dyn std::io::Read + Send>> {
        // The public trait cannot turn an arbitrary URL into ambient network authority.
        // The compiled catalog pins the initial source; redirects remain HTTPS, and the
        // store checks the exact length and SHA-256 even across CDN redirects.
        let catalog = RemovalService::default();
        if !catalog.catalog().iter().any(|entry| entry.id == model.id && entry.url == model.url && entry.sha256 == model.sha256 && entry.bytes == model.bytes)
            || !model.url.starts_with("https://huggingface.co/")
        {
            return Err(Error::Download("the model source is not in the verified catalog".into()));
        }
        self.open_pinned(model.url, model.bytes)
    }

    fn open_artifact(&self, artifact: &ModelArtifact) -> photocraft_ml::Result<Box<dyn std::io::Read + Send>> {
        let pinned = photocraft_ml::model_artifact(artifact.id)?;
        if artifact.file_name != pinned.file_name
            || artifact.bytes != pinned.bytes
            || artifact.sha256 != pinned.sha256
            || artifact.url != pinned.url
            || artifact.license != pinned.license
            || artifact.license_url != pinned.license_url
            || artifact.source != pinned.source
            || !artifact.url.starts_with("https://huggingface.co/")
        {
            return Err(Error::Download("the model artifact is not in the verified catalog".into()));
        }
        self.open_pinned(pinned.url, pinned.bytes)
    }
}

impl HttpsModels {
    fn open_pinned(&self, url: &str, expected_bytes: u64) -> photocraft_ml::Result<Box<dyn std::io::Read + Send>> {
        let response = self.agent.get(url).header("Accept-Encoding", "identity").call().map_err(|e| Error::Download(e.to_string()))?;
        if let Some(length) = response.headers().get("content-length") {
            let bytes = length.to_str().ok().and_then(|value| value.parse::<u64>().ok());
            if bytes != Some(expected_bytes) {
                return Err(Error::Download("the server returned an unexpected model size".into()));
            }
        }
        let (_, body) = response.into_parts();
        Ok(Box::new(body.into_reader()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_service_reuses_the_saved_folder_after_preferences_are_loaded() {
        let folder = std::env::temp_dir().join("photocraft-existing-model-store");
        let text = serde_json::json!({"generativeRemove":{"modelDirectory":folder}}).to_string();
        let mut session = photocraft_engine::Session::new();
        assert!(session.load_prefs_json(&text).unwrap().is_empty());
        configure(&mut session);
        let expected = std::env::var_os("PHOTOCRAFT_MODEL_DIR").filter(|path| !path.is_empty()).map(PathBuf::from).unwrap_or(folder);
        assert_eq!(session.removal_service().model_directory(), Some(expected.as_path()));
        let status = session.execute("ai.models.list", serde_json::json!({})).unwrap();
        assert_eq!(status["modelDirectory"], serde_json::json!(expected));
        assert!(!session.has_jobs());
    }
    #[test]
    fn configuring_default_models_does_not_create_or_rewrite_shared_settings() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        struct TestDirectory(PathBuf);
        impl Drop for TestDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = TestDirectory(std::env::temp_dir().join(format!("photocraft-model-configure-{}-{id}", std::process::id())));
        std::fs::create_dir(&root.0).unwrap();
        let store = root.0.join("ai/connections.json");
        let models = root.0.join("models");
        let mut session = photocraft_engine::Session::new();
        configure_paths(&mut session, Some(models.clone()), Some(store.clone()));
        assert!(!store.exists());
        assert!(!models.exists());
        std::fs::create_dir_all(store.parent().unwrap()).unwrap();
        let original = br#"{ "schema_version": 1, "connections": [], "defaults": [], "apps": {} }"#;
        std::fs::write(&store, original).unwrap();
        configure_paths(&mut session, Some(models.clone()), Some(store.clone()));
        session.execute("ai.models.list", serde_json::json!({})).unwrap();
        assert_eq!(std::fs::read(&store).unwrap(), original);
        assert!(!models.exists());
        assert!(!session.has_jobs());
    }
    #[test]
    fn model_transport_is_https_only_with_bounded_timeouts() {
        let client = HttpsModels::new();
        let config = client.agent.config();
        assert!(config.https_only());
        assert_eq!(config.max_redirects(), 5);
        assert_eq!(config.timeout_per_read(), Some(Duration::from_secs(10)));
    }
    #[test]
    fn model_transport_refuses_uncatalogued_artifact_before_network() {
        let mut artifact = photocraft_ml::model_artifact("osor-sdxl-fixed-prompt").unwrap().clone();
        artifact.url = "https://example.invalid/substituted.safetensors";
        let error = HttpsModels::new().open_artifact(&artifact).err().unwrap().to_string();
        assert!(error.contains("verified catalog"), "{error}");
    }
    #[test]
    fn model_transport_refuses_substituted_artifact_identity_before_network() {
        let client = HttpsModels::new();
        let pinned = photocraft_ml::model_artifact("osor-sdxl-fixed-prompt").unwrap();
        for variant in 0..4 {
            let mut artifact = pinned.clone();
            match variant {
                0 => artifact.bytes += 1,
                1 => artifact.sha256 = "00",
                2 => artifact.file_name = "other.safetensors",
                _ => artifact.license_url = "https://example.invalid/terms",
            }
            let error = client.open_artifact(&artifact).err().unwrap().to_string();
            assert!(error.contains("verified catalog"), "{error}");
        }
    }
    #[test]
    #[ignore = "explicit HTTPS qualification; requires PHOTOCRAFT_TEST_PROMPT_PATH with the independently hash-verified catalog prompt"]
    fn real_https_model_artifact_matches_verified_local_prompt() {
        use std::io::Read;
        let expected = std::fs::read(std::env::var_os("PHOTOCRAFT_TEST_PROMPT_PATH").expect("set the verified prompt path")).unwrap();
        let artifact = photocraft_ml::model_artifact("osor-sdxl-fixed-prompt").unwrap();
        assert_eq!(expected.len() as u64, artifact.bytes);
        let stream = HttpsModels::new().open_artifact(artifact).unwrap();
        let mut actual = Vec::new();
        stream.take(artifact.bytes + 1).read_to_end(&mut actual).unwrap();
        assert_eq!(actual, expected, "HTTPS transport must return the independently verified pinned bytes");
    }
    #[test]
    fn model_transport_refuses_uncatalogued_source_before_network() {
        let mut model = RemovalService::default().catalog()[0].clone();
        model.url = "https://example.invalid/model.onnx";
        let error = HttpsModels::new().open(&model).err().unwrap().to_string();
        assert!(error.contains("verified catalog"), "{error}");
    }
}
