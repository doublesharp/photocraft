//! Bounded local image repair and optional, verified model data.
//!
//! Model data is never embedded or fetched implicitly. The application supplies a managed
//! directory and optional HTTPS transport. RGB values are encoded sRGB floats, not linear light.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod catalog;
#[cfg(any(test, all(not(target_arch = "wasm32"), feature = "native-inference")))]
mod resample;
#[cfg(all(not(target_arch = "wasm32"), feature = "native-inference"))]
mod runtime;
#[cfg(not(target_arch = "wasm32"))]
mod store;

pub use catalog::{ModelSpec, ModelStatus};
pub const MAX_INPUT_PIXELS: u64 = 16_777_216;

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
    used: bool,
    #[cfg(not(target_arch = "wasm32"))]
    _lease: store::Lease,
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
        let minimum = input.minimum_buffer_bytes()?;
        if minimum > self.transient_bytes {
            return Err(Error::InvalidInput("removal input and output exceed the admitted transient buffer bound".into()));
        }
        check(ctl)?;
        if self.used {
            return Err(Error::InvalidInput("a removal permit can run only one inference".into()));
        }
        self.used = true;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(backend) = &self.service.backend {
            return self.service.native.infer_backend(backend.as_ref(), self.model.id, input, ctl);
        }
        #[cfg(all(not(target_arch = "wasm32"), feature = "native-inference"))]
        {
            self.service.native.infer(self.model, input, ctl)
        }
        #[cfg(not(all(not(target_arch = "wasm32"), feature = "native-inference")))]
        {
            let _ = (&self.service, self.model);
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
            if !self.runtime_available() {
                return Err(Error::Unavailable);
            }
            let lease = self.native.begin()?;
            self.native.prepare_model(model.id, self.backend.is_some());
            Ok(RemovalPermit { service: self, model, transient_bytes, used: false, _lease: lease })
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

    pub fn runtime_available(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        if self.backend.is_some() {
            return true;
        }
        cfg!(all(not(target_arch = "wasm32"), feature = "native-inference"))
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

    pub fn catalog(&self) -> &'static [ModelSpec] {
        catalog::CATALOG
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut statuses = self.native.status(self.runtime_available());
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
        let mut permit = self.admit(model_id, input.minimum_buffer_bytes()?, u64::MAX)?;
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
        let model = catalog::find(model_id)?;
        check(ctl)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.install(model, ctl)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = model;
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
    fn minimum_buffer_bytes(&self) -> Result<u64> {
        u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|n| n.checked_mul(28))
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
