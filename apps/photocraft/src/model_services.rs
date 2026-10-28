//! Native model transport and storage used by the desktop and headless CLI.
//! No work happens at startup: resolving a directory and creating an HTTP agent download
//! nothing. Only explicit engine model-management commands open a connection.

use photocraft_ml::{Error, ModelDownloader, ModelSpec, RemovalService};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[path = "model_services/directories.rs"]
mod directories;

/// Optional model data is local to this machine, separate from roaming preferences.
/// PHOTOCRAFT_MODEL_DIR explicitly moves it (including to Scratch for development).
pub fn model_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    directories::resolve(|k| std::env::var_os(k), exe.as_deref(), directories::Platform::current())
}

pub fn configure(session: &mut photocraft_engine::Session) {
    let service = RemovalService::new(model_dir()).with_downloader(Arc::new(HttpsModels::new()));
    #[cfg(all(feature = "osor-metal", target_os = "macos"))]
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
            .timeout_global(Some(Duration::from_secs(600)))
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
        let response = self.agent.get(model.url).header("Accept-Encoding", "identity").call().map_err(|e| Error::Download(e.to_string()))?;
        if let Some(length) = response.headers().get("content-length") {
            let bytes = length.to_str().ok().and_then(|value| value.parse::<u64>().ok());
            if bytes != Some(model.bytes) {
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
    fn model_transport_is_https_only_with_bounded_timeouts() {
        let client = HttpsModels::new();
        let config = client.agent.config();
        assert!(config.https_only());
        assert_eq!(config.max_redirects(), 5);
        assert_eq!(config.timeout_per_read(), Some(Duration::from_secs(10)));
    }
    #[test]
    fn model_transport_refuses_uncatalogued_source_before_network() {
        let mut model = RemovalService::default().catalog()[0].clone();
        model.url = "https://example.invalid/model.onnx";
        let error = HttpsModels::new().open(&model).err().unwrap().to_string();
        assert!(error.contains("verified catalog"), "{error}");
    }
}
