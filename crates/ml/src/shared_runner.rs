//! Native runners for the shared Craft AI client. Images stay in private in-memory tickets;
//! the ticket guard fences cancellation until the actual native worker has drained.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use craft_ai_core::{AiError, CapabilityId, CapabilitySpec, ErrorKind, InputRole, ModelId, ModelInfo, Output, Request};
use craft_ai_engine::{LoadedModel, ModelRunner, RunContext};

use crate::{Error, ModelSpec, RemovalControl, RemovalInput, RemovalOutput, Result, store};

#[path = "shared_worker.rs"]
mod worker;
pub(super) use worker::Bridge;

pub(super) const AUTOMATIC_MODEL: &str = "photocraft-repair-auto";
const SOURCE_MEDIA: &str = "image/x-photocraft-rgb-f32";
const MASK_MEDIA: &str = "image/x-photocraft-mask-f32";
const OUTPUT_MEDIA: &str = "image/x-photocraft-removal-f32";

pub(super) fn engine_id(id: &str) -> Result<&'static str> {
    match id {
        "lama" => Ok("photocraft-lama"),
        "migan-v1" => Ok("photocraft-migan-v1"),
        "osor-sdxl" => Ok("photocraft-osor-sdxl"),
        AUTOMATIC_MODEL => Ok(AUTOMATIC_MODEL),
        _ => Err(Error::UnknownModel(id.into())),
    }
}

pub(super) struct Runner {
    native: Arc<store::NativeService>,
    main: &'static ModelSpec,
    info: ModelInfo,
    engine: &'static str,
    automatic: bool,
    bridge: Arc<Bridge>,
}

impl Runner {
    pub(super) fn new(native: Arc<store::NativeService>, model: &'static ModelSpec, bridge: Arc<Bridge>) -> Result<Self> {
        Self::build(native, model, false, bridge)
    }

    pub(super) fn automatic(native: Arc<store::NativeService>, bridge: Arc<Bridge>) -> Result<Self> {
        Self::build(native, crate::catalog::find("lama")?, true, bridge)
    }

    fn build(native: Arc<store::NativeService>, main: &'static ModelSpec, automatic: bool, bridge: Arc<Bridge>) -> Result<Self> {
        let id = if automatic { AUTOMATIC_MODEL } else { main.id };
        let capability = CapabilityId::parse("image.inpaint").map_err(invalid)?;
        let spec = CapabilitySpec::new(capability)
            .accepting(InputRole::Source, 1)
            .accepting(InputRole::Mask, 1)
            .requiring(InputRole::Source)
            .requiring(InputRole::Mask);
        let label = if automatic { "Recommended: LaMa + OSOR" } else { main.label };
        let info = ModelInfo::new(ModelId::new(id).map_err(invalid)?, label).with(spec);
        Ok(Self { native, main, info, engine: engine_id(id)?, automatic, bridge })
    }
}

impl ModelRunner for Runner {
    fn engine(&self) -> &str {
        self.engine
    }

    fn model(&self) -> ModelInfo {
        self.info.clone()
    }

    fn load(&self, weights: &Path) -> std::result::Result<Box<dyn LoadedModel>, AiError> {
        let expected = self.native.shared_weights_path(self.main).map_err(ai_error)?;
        if weights != expected.as_path() {
            return Err(AiError::unsupported("the selected weights do not match this verified removal model; open AI Models to set it up again"));
        }
        // This wrapper opens no model and allocates no inference buffers. All heavyweight work
        // happens under Flight inside run; NativeService owns verification and cache eviction.
        Ok(Box::new(Loaded { native: self.native.clone(), main: self.main, automatic: self.automatic, weights: expected, bridge: self.bridge.clone() }))
    }
}

struct Loaded {
    native: Arc<store::NativeService>,
    main: &'static ModelSpec,
    automatic: bool,
    weights: PathBuf,
    bridge: Arc<Bridge>,
}

impl LoadedModel for Loaded {
    fn run(&mut self, request: &Request, ctx: &RunContext) -> std::result::Result<Vec<Output>, AiError> {
        ctx.check_cancelled()?;
        let flight = self.bridge.enter(request).map_err(ai_error)?;
        let component = flight.ticket.component;
        if (self.automatic && !matches!(component, "lama" | "osor-sdxl")) || (!self.automatic && component != self.main.id) {
            return Err(AiError::unsupported("the selected removal setup does not include that model"));
        }
        if self.native.shared_weights_path(self.main).map_err(ai_error)? != self.weights {
            return Err(AiError::unsupported("the selected model files changed; open AI Models to select the current installation"));
        }
        let model = crate::catalog::find(component).map_err(ai_error)?;
        let control = Control { ctx, ticket: &flight.ticket };
        let output = run_native(&self.native, model, &flight.ticket.input, &control, flight.ticket.available).map_err(ai_error)?;
        ctx.check_cancelled()?;
        let output = flight.publish(output).map_err(ai_error)?;
        // Flight drops after the native forward, device synchronization and result publication.
        Ok(vec![output])
    }
}

fn run_native(
    native: &store::NativeService,
    model: &'static ModelSpec,
    input: &RemovalInput,
    ctl: &dyn RemovalControl,
    available: u64,
) -> Result<RemovalOutput> {
    #[cfg(feature = "osor")]
    if model.id == "osor-sdxl" {
        return native.infer_osor(input, ctl);
    }
    #[cfg(feature = "native-inference")]
    {
        native.infer(model, input, ctl, available)
    }
    #[cfg(not(feature = "native-inference"))]
    {
        let _ = (native, model, input, ctl, available);
        Err(Error::Unavailable)
    }
}

struct Control<'a> {
    ctx: &'a RunContext,
    ticket: &'a worker::Ticket,
}
impl RemovalControl for Control<'_> {
    fn cancelled(&self) -> bool {
        self.ctx.is_cancelled() || self.ticket.cancelled()
    }
    fn progress(&self, fraction: f32, _label: &str) {
        if fraction.is_finite() {
            self.ctx.progress((fraction.clamp(0.0, 1.0) * 99.0) as u8);
        }
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::InvalidInput(error.to_string())
}
fn ai_error(error: Error) -> AiError {
    let kind = match error {
        Error::Cancelled => ErrorKind::Cancelled,
        Error::Unavailable | Error::NotInstalled(_) | Error::UnknownModel(_) => ErrorKind::Unsupported,
        Error::InvalidInput(_) => ErrorKind::InvalidRequest,
        _ => ErrorKind::Provider,
    };
    AiError::new(kind, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_refuses_components_outside_the_selected_setup() {
        let native = Arc::new(store::NativeService::new(None));
        let bridge = Arc::new(Bridge::default());
        let mut loaded = Loaded { native, main: crate::catalog::find("lama").unwrap(), automatic: false, weights: PathBuf::new(), bridge: bridge.clone() };
        let input = RemovalInput { width: 2, height: 1, rgb: vec![0.25; 6], mask: vec![0.0, 1.0] };
        let ticket = bridge.request(&input, 100, "osor-sdxl").unwrap();
        let error = loaded.run(&ticket.request(), &RunContext::detached()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Unsupported);
        loaded.automatic = true;
        let ticket = bridge.request(&input, 100, "migan-v1").unwrap();
        assert_eq!(loaded.run(&ticket.request(), &RunContext::detached()).unwrap_err().kind, ErrorKind::Unsupported);
    }

    #[test]
    fn shared_terminal_cancellation_keeps_native_admission_until_runner_returns() {
        use craft_ai_core::{JobState, Provider};
        use craft_ai_engine::EngineProvider;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Condvar, Mutex, mpsc};
        use std::time::Duration;

        struct Blocked {
            info: ModelInfo,
            bridge: Arc<Bridge>,
            started: mpsc::Sender<()>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }
        impl ModelRunner for Blocked {
            fn engine(&self) -> &str {
                "photocraft-lama"
            }
            fn model(&self) -> ModelInfo {
                self.info.clone()
            }
            fn load(&self, _: &Path) -> std::result::Result<Box<dyn LoadedModel>, AiError> {
                Ok(Box::new(BlockedRun { bridge: self.bridge.clone(), started: self.started.clone(), release: self.release.clone() }))
            }
        }
        struct BlockedRun {
            bridge: Arc<Bridge>,
            started: mpsc::Sender<()>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }
        impl LoadedModel for BlockedRun {
            fn run(&mut self, request: &Request, ctx: &RunContext) -> std::result::Result<Vec<Output>, AiError> {
                let _flight = self.bridge.enter(request).map_err(ai_error)?;
                self.started.send(()).unwrap();
                let (lock, condvar) = &*self.release;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = condvar.wait(released).unwrap();
                }
                ctx.check_cancelled()?;
                Ok(Vec::new())
            }
        }
        let bridge = Arc::new(Bridge::default());
        let input = RemovalInput { width: 2, height: 1, rgb: vec![0.25; 6], mask: vec![0.0, 1.0] };
        let ticket = bridge.request(&input, 100, "lama").unwrap();
        let info = Runner::new(Arc::new(store::NativeService::new(None)), crate::catalog::find("lama").unwrap(), bridge.clone()).unwrap().model();
        let model = info.id.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let runner: Arc<dyn ModelRunner> = Arc::new(Blocked { info, bridge, started: started_tx, release: release.clone() });
        let weights = tempfile::NamedTempFile::new().unwrap();
        let provider = EngineProvider::new("photocraft-lama", [(runner, weights.path().to_owned())]);
        let job = provider.submit(&model, &ticket.request()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        ticket.cancel();
        provider.cancel(&job).unwrap();
        assert_eq!(provider.poll(&job).unwrap().state, JobState::Cancelled);
        let admission = Arc::new(AtomicBool::new(true));
        let admitted = admission.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let draining = std::thread::spawn(move || {
            drop(ticket);
            admitted.store(false, Ordering::Release);
            done_tx.send(()).unwrap();
        });
        let returned_early = done_rx.recv_timeout(Duration::from_millis(50)).is_ok();
        let held_until_release = admission.load(Ordering::Acquire);
        let (lock, condvar) = &*release;
        *lock.lock().unwrap() = true;
        condvar.notify_all();
        if !returned_early {
            done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        draining.join().unwrap();
        assert!(!returned_early, "a terminal Cancelled snapshot must not release native admission");
        assert!(held_until_release);
        assert!(!admission.load(Ordering::Acquire));
    }
}
