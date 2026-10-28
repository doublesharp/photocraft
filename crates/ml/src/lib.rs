//! Bounded local image repair and optional, verified model data.
//!
//! Model data is never embedded or fetched implicitly. The application supplies a managed
//! directory and optional HTTPS transport. RGB values are encoded sRGB floats, not linear light.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod catalog;
#[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
mod osor_runtime;
#[cfg(any(test, all(not(target_arch = "wasm32"), any(feature = "native-inference", feature = "osor"))))]
mod resample;
#[cfg(all(not(target_arch = "wasm32"), feature = "native-inference"))]
mod runtime;
#[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
pub use osor::Backend as OsorBackend;
#[cfg(not(target_arch = "wasm32"))]
mod store;

pub use catalog::{ModelArtifact, ModelSpec, ModelStatus, OSOR_PREPARATION_BYTES, model_artifact};
pub const MAX_INPUT_PIXELS: u64 = 16_777_216;
/// Weight license shipped with explicitly downloaded OSOR data, separate from the code license.
pub const OSOR_LICENSE_TEXT: &str = include_str!("../licenses/LICENSE-osor-sdxl.txt");

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("local AI inference is unavailable in this build")]
    Unavailable,
    #[error("unknown local removal model `{0}`")]
    UnknownModel(String),
    #[error("model `{0}` is not installed; install it in AI Models")]
    NotInstalled(String),
    #[error("local AI is busy; wait for its current operation to finish")]
    Busy,
    #[error("operation cancelled")]
    Cancelled,
    #[error("{0}")]
    InvalidInput(String),
    #[error("model verification failed: {0}")]
    Integrity(String),
    #[error("{0}")]
    Storage(String),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("local AI inference failed: {0}")]
    Inference(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Cooperative boundaries; a noninterruptible forward retains the service's permit until return.
pub trait RemovalControl: Send + Sync {
    fn cancelled(&self) -> bool;
    fn progress(&self, fraction: f32, label: &str);
    /// Close the cancellation window before an irreversible external side effect.
    /// Hosts with asynchronous cancellation should override this with an atomic commit gate.
    fn begin_irreversible(&self) -> Result<()> {
        if self.cancelled() { Err(Error::Cancelled) } else { Ok(()) }
    }
}

/// Compiled adapter injection for deterministic application tests and future qualified runtimes.
/// Loading arbitrary model code from disk is deliberately not part of this interface.
pub trait RemovalBackend: Send + Sync {
    fn ready(&self, model_id: &str) -> bool;
    fn infer(&self, model_id: &str, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput>;
}

pub struct NoopControl;
impl RemovalControl for NoopControl {
    fn cancelled(&self) -> bool {
        false
    }
    fn progress(&self, _fraction: f32, _label: &str) {}
}

/// Explicit app-layer transport. Implementations must use bounded HTTPS requests and a generic
/// User-Agent. The store independently verifies all bytes before publishing them.
pub trait ModelDownloader: Send + Sync {
    fn open(&self, model: &ModelSpec) -> Result<Box<dyn Read + Send>>;
    fn open_artifact(&self, _artifact: &ModelArtifact) -> Result<Box<dyn Read + Send>> {
        Err(Error::Download("multi-file model downloading is unavailable in this application".into()))
    }
}

/// A context crop, in encoded sRGB, with white (positive) mask coverage denoting the hole.
#[derive(Clone, Debug)]
pub struct RemovalInput {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
    pub mask: Vec<f32>,
}

#[derive(Debug)]
pub struct RemovalOutput {
    pub rgb: Vec<f32>,
    /// Learned source-sized edit support. OSOR RGB is already latent blended; do not
    /// multiply this alpha into RGB a second time. None for adapters without learned support.
    pub edit_alpha: Option<Vec<f32>>,
    /// Model pixels per input pixel; values below 1 disclose downsampling.
    pub inference_scale: f32,
    pub load_ms: f64,
    pub forward_ms: f64,
}

/// An injected service, shared by the engine and its workers. Admission and the model cache are process-wide.
pub struct RemovalService {
    #[cfg(not(target_arch = "wasm32"))]
    backend: Option<Arc<dyn RemovalBackend>>,
    #[cfg(not(target_arch = "wasm32"))]
    native: store::NativeService,
}

/// Process-wide admission held until preprocessing and the real worker have finished.
pub struct RemovalPermit<'a> {
    service: &'a RemovalService,
    model: &'static ModelSpec,
    transient_bytes: u64,
    available_bytes: u64,
    used: bool,
    #[cfg(not(target_arch = "wasm32"))]
    _lease: store::Lease,
}

/// One frozen resource admission for at most 32 sequential regional forwards.
/// The caller budgets retained region outputs in `transient_bytes`.
pub struct RemovalBatchPermit<'a> {
    service: &'a RemovalService,
    models: Vec<&'static ModelSpec>,
    transient_bytes: u64,
    available_bytes: u64,
    calls: u32,
    failed: bool,
    #[cfg(not(target_arch = "wasm32"))]
    _lease: store::Lease,
}

impl RemovalBatchPermit<'_> {
    pub fn into_reservation(self) -> RemovalReservation {
        RemovalReservation {
            #[cfg(not(target_arch = "wasm32"))]
            _lease: self._lease,
        }
    }

    pub fn model_sha256(&self, model_id: &str) -> Result<String> {
        self.model(model_id)?;
        self.service.model_sha256(model_id)
    }

    fn model(&self, model_id: &str) -> Result<&'static ModelSpec> {
        self.models
            .iter()
            .copied()
            .find(|model| model.id == model_id)
            .ok_or_else(|| Error::InvalidInput("model was not included in the batch admission".into()))
    }

    pub fn infer(&mut self, model_id: &str, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
        let model = self.model(model_id)?;
        if self.failed || self.calls >= 32 {
            return Err(Error::InvalidInput("removal batch has failed or reached its 32-forward limit".into()));
        }
        input.validate()?;
        if input.minimum_buffer_bytes_for(model_id)? > self.transient_bytes {
            return Err(Error::InvalidInput("removal input and output exceed the admitted transient buffer bound".into()));
        }
        check(ctl)?;
        self.calls += 1;
        let result = self.service.infer_admitted(model, input, ctl, self.available_bytes);
        self.failed = result.is_err();
        result
    }
}

/// Owns the process-wide reservation while admitted output remains live.
/// Dropping the value releases admission, including when a result is discarded.
pub struct RemovalReservation {
    #[cfg(not(target_arch = "wasm32"))]
    _lease: store::Lease,
}

impl RemovalPermit<'_> {
    /// Transfer admission to the caller's output, to hold until it is applied or discarded.
    pub fn into_reservation(self) -> RemovalReservation {
        RemovalReservation {
            #[cfg(not(target_arch = "wasm32"))]
            _lease: self._lease,
        }
    }

    /// Run the admitted model. The exclusive borrow prevents overlapping forwards on one permit.
    pub fn infer(&mut self, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
        input.validate()?;
        let minimum = input.minimum_buffer_bytes_for(self.model.id)?;
        if minimum > self.transient_bytes {
            return Err(Error::InvalidInput("removal input and output exceed the admitted transient buffer bound".into()));
        }
        check(ctl)?;
        if self.used {
            return Err(Error::InvalidInput("a removal permit can run only one inference".into()));
        }
        self.used = true;
        self.service.infer_admitted(self.model, input, ctl, self.available_bytes)
    }
}

impl RemovalService {
    fn infer_admitted(&self, model: &'static ModelSpec, input: &RemovalInput, ctl: &dyn RemovalControl, available_bytes: u64) -> Result<RemovalOutput> {
        #[cfg(not(target_arch = "wasm32"))]
        self.native.prepare_model(model.id, self.backend.is_some(), available_bytes);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(backend) = &self.backend {
            return self.native.infer_backend(backend.as_ref(), model.id, input, ctl);
        }
        #[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
        if model.id == "osor-sdxl" {
            return self.native.infer_osor(input, ctl);
        }
        #[cfg(all(not(target_arch = "wasm32"), feature = "native-inference"))]
        {
            self.native.infer(model, input, ctl)
        }
        #[cfg(not(all(not(target_arch = "wasm32"), feature = "native-inference")))]
        {
            let _ = (model, input, ctl, available_bytes);
            Err(Error::Unavailable)
        }
    }
}

impl Default for RemovalService {
    fn default() -> Self {
        Self::new(None)
    }
}

impl RemovalService {
    pub fn admit_models(&self, model_ids: &[&str], transient_bytes: u64, budget_bytes: u64) -> Result<RemovalBatchPermit<'_>> {
        if model_ids.is_empty() || model_ids.len() > 32 {
            return Err(Error::InvalidInput("removal batch must admit between 1 and 32 model identifiers".into()));
        }
        let mut models = Vec::new();
        let mut working = 0u64;
        for id in model_ids {
            let model = catalog::find(id)?;
            working = working.max(u64::from(model.memory_mib) * 1024 * 1024);
            if !self.model_runtime_available(model.id) {
                return Err(Error::Unavailable);
            }
            if !models.iter().any(|admitted: &&ModelSpec| admitted.id == model.id) {
                models.push(model);
            }
        }
        let required = working.checked_add(transient_bytes).ok_or_else(|| Error::InvalidInput("removal memory estimate overflow".into()))?;
        if required > budget_bytes {
            return Err(Error::InvalidInput(format!("local removal needs an estimated {required} bytes, exceeding the {budget_bytes}-byte budget")));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let lease = self.native.begin()?;
            let available_bytes = budget_bytes.saturating_sub(transient_bytes);
            crate::cache::admit(self.backend.is_some(), available_bytes);
            Ok(RemovalBatchPermit { service: self, models, transient_bytes, available_bytes, calls: 0, failed: false, _lease: lease })
        }
        #[cfg(target_arch = "wasm32")]
        Err(Error::Unavailable)
    }

    /// Admit model memory plus caller-owned transient buffers against a frozen byte budget.
    pub fn admit(&self, model_id: &str, transient_bytes: u64, budget_bytes: u64) -> Result<RemovalPermit<'_>> {
        let model = catalog::find(model_id)?;
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (model, transient_bytes, budget_bytes);
            return Err(Error::Unavailable);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let required = u64::from(model.memory_mib)
            .checked_mul(1024 * 1024)
            .and_then(|working| working.checked_add(transient_bytes))
            .ok_or_else(|| Error::InvalidInput("removal memory estimate overflow".into()))?;
        #[cfg(not(target_arch = "wasm32"))]
        if required > budget_bytes {
            return Err(Error::InvalidInput(format!("local removal needs an estimated {required} bytes, exceeding the {budget_bytes}-byte budget")));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if !self.model_runtime_available(model.id) {
                return Err(Error::Unavailable);
            }
            let lease = self.native.begin()?;
            let available_bytes = budget_bytes.saturating_sub(transient_bytes);
            self.native.prepare_model(model.id, self.backend.is_some(), available_bytes);
            Ok(RemovalPermit { service: self, model, transient_bytes, available_bytes, used: false, _lease: lease })
        }
    }

    pub fn new(model_dir: Option<PathBuf>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = model_dir;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            backend: None,
            #[cfg(not(target_arch = "wasm32"))]
            native: store::NativeService::new(model_dir),
        }
    }

    pub fn with_downloader(self, downloader: Arc<dyn ModelDownloader>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        self.native.set_downloader(downloader);
        #[cfg(target_arch = "wasm32")]
        let _ = downloader;
        self
    }

    pub fn with_backend(self, backend: Arc<dyn RemovalBackend>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut service = self;
            service.backend = Some(backend);
            service
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = backend;
            self
        }
    }

    /// Select OSOR compute explicitly; CPU is the default. No process environment is read.
    #[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
    pub fn with_osor_backend(self, backend: OsorBackend) -> Self {
        self.native.set_osor_backend(backend);
        self
    }

    /// Whether this build supports the selected adapter (separate from installation).
    pub fn model_runtime_available(&self, id: &str) -> bool {
        if catalog::find(id).is_err() {
            return false;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(backend) = &self.backend {
            return backend.ready(id);
        }
        if id == "osor-sdxl" {
            #[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
            let available = self.native.osor_runtime_available();
            #[cfg(not(all(not(target_arch = "wasm32"), feature = "osor")))]
            let available = false;
            return available;
        }
        cfg!(all(not(target_arch = "wasm32"), feature = "native-inference"))
    }

    pub fn runtime_available(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        if self.backend.is_some() {
            return true;
        }
        cfg!(all(not(target_arch = "wasm32"), any(feature = "native-inference", feature = "osor")))
    }

    pub fn management_available(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.management_available()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Immutable catalog artifact hash, or the explicitly imported OSOR manifest hash.
    /// Injected adapters retain the catalog identity supplied by their test contract.
    pub fn model_sha256(&self, model_id: &str) -> Result<String> {
        let model = catalog::find(model_id)?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.backend.is_some() {
            return Ok(model.sha256.into());
        }
        if model.id == "osor-sdxl" {
            #[cfg(all(not(target_arch = "wasm32"), feature = "osor"))]
            return self.native.model_sha256();
            #[cfg(not(all(not(target_arch = "wasm32"), feature = "osor")))]
            return Err(Error::Unavailable);
        }
        Ok(model.sha256.into())
    }

    pub fn catalog(&self) -> &'static [ModelSpec] {
        catalog::CATALOG
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut statuses = self.native.status(|id| self.model_runtime_available(id));
            if let Some(backend) = &self.backend {
                for status in &mut statuses {
                    if backend.ready(&status.id) {
                        status.installed = true;
                        status.available = true;
                        status.verified = true;
                        status.reason = None;
                    }
                }
            }
            statuses
        }
        #[cfg(target_arch = "wasm32")]
        {
            catalog::CATALOG.iter().map(|m| ModelStatus::unavailable(m, "AI removal is unavailable in the browser")).collect()
        }
    }

    pub fn infer(&self, model_id: &str, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
        catalog::find(model_id)?;
        input.validate()?;
        check(ctl)?;
        let mut permit = self.admit(model_id, input.minimum_buffer_bytes_for(model_id)?, u64::MAX)?;
        permit.infer(input, ctl)
    }

    pub fn import(&self, model_id: &str, path: &Path, ctl: &dyn RemovalControl) -> Result<()> {
        let model = catalog::find(model_id)?;
        check(ctl)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.import(model, path, ctl)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (model, path);
            Err(Error::Unavailable)
        }
    }

    pub fn install(&self, model_id: &str, ctl: &dyn RemovalControl) -> Result<()> {
        self.install_with_memory_budget(model_id, 0, ctl)
    }

    /// Explicit preparation with a host-frozen memory budget. This never runs from inference.
    /// OSOR conversion requires a 40 GiB admission estimate; ONNX installs stream to disk.
    pub fn install_with_memory_budget(&self, model_id: &str, budget_bytes: u64, ctl: &dyn RemovalControl) -> Result<()> {
        let model = catalog::find(model_id)?;
        check(ctl)?;
        let required = u64::from(model.preparation_memory_mib) * 1024 * 1024;
        if budget_bytes < required {
            return Err(Error::InvalidInput(format!(
                "{} preparation needs an estimated {required} bytes, exceeding the {budget_bytes}-byte memory budget",
                model.label
            )));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.install(model, ctl)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (model, budget_bytes);
            Err(Error::Unavailable)
        }
    }

    pub fn remove(&self, model_id: &str) -> Result<()> {
        let model = catalog::find(model_id)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.remove(model)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = model;
            Err(Error::Unavailable)
        }
    }
}

impl RemovalInput {
    fn minimum_buffer_bytes_for(&self, model_id: &str) -> Result<u64> {
        u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|n| n.checked_mul(if model_id == "osor-sdxl" { 32 } else { 28 }))
            .ok_or_else(|| Error::InvalidInput("removal buffer estimate overflow".into()))
    }

    pub fn validate(&self) -> Result<()> {
        let pixels = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .filter(|&n| n > 0 && n <= MAX_INPUT_PIXELS)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::InvalidInput("invalid or oversized removal crop".into()))?;
        let rgb_len = pixels.checked_mul(3).ok_or_else(|| Error::InvalidInput("removal crop is too large".into()))?;
        if self.rgb.len() != rgb_len || self.mask.len() != pixels {
            return Err(Error::InvalidInput("removal crop buffer dimensions do not match".into()));
        }
        if self.rgb.iter().chain(&self.mask).any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err(Error::InvalidInput("removal crop samples must be finite values between 0 and 1".into()));
        }
        if !self.mask.iter().any(|&v| v > 0.0) || !self.mask.contains(&0.0) {
            return Err(Error::InvalidInput("removal crop must contain both a hole and known context".into()));
        }
        Ok(())
    }
}

fn check(ctl: &dyn RemovalControl) -> Result<()> {
    if ctl.cancelled() { Err(Error::Cancelled) } else { Ok(()) }
}

#[cfg(test)]
mod tests;
