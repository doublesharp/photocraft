//! Shared Craft AI configuration. Construction and status are read-only; only the
//! explicit setup command registers a folder, native connections and a tool binding.
use crate::{
    Error, RemovalControl, RemovalInput, RemovalOutput, Result, catalog,
    shared_runner::{self, AUTOMATIC_MODEL, Bridge, Runner},
    store::NativeService,
};
use craft_ai_client::{AiClient, ProviderFactory, RoleStatus};
use craft_ai_core::{
    AppId, Auth, Billing, Binding, CapabilityId, Connection, ConnectionId, ConsentLedger, Event, InputRole, JobState, Locality, ModelId, Role, RoleId,
};
use craft_ai_engine::{EngineRegistry, ModelRunner};
use craft_ai_native::{secrets::MemoryStore, shared::SharedStore};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(super) struct SharedAi {
    pub(super) store: SharedStore,
    client: AiClient,
    native: Arc<NativeService>,
    bridge: Arc<Bridge>,
}
fn error(e: impl std::fmt::Display) -> Error {
    Error::Storage(format!("Shared AI: {e}"))
}
fn connection_id(model: &str) -> Result<ConnectionId> {
    ConnectionId::new(format!("photocraft-native-{model}")).map_err(error)
}
fn app() -> Result<AppId> {
    AppId::new("photocraft").map_err(error)
}
fn role() -> Result<Role> {
    craft_ai_features::builtin()
        .into_iter()
        .find(|f| f.id == "remove-object")
        .and_then(|f| f.role())
        .ok_or_else(|| error("Remove tool contract is unavailable"))
}
fn fill_role() -> Result<Role> {
    Ok(Role {
        id: RoleId::new("background-repair").map_err(error)?,
        label: "Reconstruct background".into(),
        capability: CapabilityId::parse("image.inpaint").map_err(error)?,
        inputs: vec![InputRole::Source, InputRole::Mask],
    })
}
impl SharedAi {
    pub(super) fn new(native: Arc<NativeService>, store: SharedStore) -> Result<Self> {
        let bridge = Arc::new(Bridge::default());
        let mut registry = EngineRegistry::new();
        let mut runners: Vec<Arc<dyn ModelRunner>> = Vec::new();
        for model in catalog::CATALOG {
            runners.push(Arc::new(Runner::new(native.clone(), model, bridge.clone())?));
        }
        runners.push(Arc::new(Runner::automatic(native.clone(), bridge.clone())?));
        registry.extend(runners);
        // This adapter is intentionally local-only. No provider keys are needed or read.
        let mut factory = ProviderFactory::standard();
        factory.register_engines(registry);
        let client = AiClient::new(app()?, store.clone(), Arc::new(MemoryStore::new())).with_factory(factory);
        Ok(Self { store, client, native, bridge })
    }
    pub(super) fn setup(&self, workflow: &str, selected: &str) -> Result<()> {
        let id = match workflow {
            "classical" => return Ok(()),
            "recommended" => AUTOMATIC_MODEL,
            "lama" => "lama",
            "compact" => "migan-v1",
            "custom" => catalog::find(selected)?.id,
            _ => return Err(error("unknown Remove setup")),
        };
        let directory = self.native.directory.as_ref().filter(|d| d.is_absolute()).ok_or_else(|| error("choose an absolute model folder"))?;
        let folder_path = directory.to_str().ok_or_else(|| error("the model folder is not valid text"))?.to_owned();
        let mut connections = Vec::new();
        for model in catalog::CATALOG {
            let runner = Runner::new(self.native.clone(), model, self.bridge.clone())?;
            connections.push(self.connection(&runner, model)?);
        }
        connections.push(self.connection(&Runner::automatic(self.native.clone(), self.bridge.clone())?, catalog::find("lama")?)?);
        let role = role()?;
        let fill_role = fill_role()?;
        let fill_binding = Binding { role: fill_role.id.clone(), connection: connection_id("lama")?, model: ModelId::new("lama").map_err(error)? };
        let binding = Binding { role: role.id.clone(), connection: connection_id(id)?, model: ModelId::new(id).map_err(error)? };
        let app = app()?;
        self.store
            .update(|store| {
                store.check_tool(&role).map_err(error)?;
                store.check_tool(&fill_role).map_err(error)?;
                if !store.custom_tools.iter().any(|tool| tool.id == fill_role.id) {
                    store.custom_tools.push(fill_role.clone());
                }
                let has_fill_choice = store.defaults.iter().any(|b| b.role == fill_role.id)
                    || store.apps.get(&app).is_some_and(|settings| settings.overrides.iter().any(|b| b.role == fill_role.id));
                if !store.defaults.iter().any(|b| b.role == fill_role.id) {
                    store.defaults.push(fill_binding.clone());
                }
                // Only our native connection IDs are updated; hosted routes, keys and the suite's
                // existing choice survive. Selecting this setup is an explicit app override.
                for connection in connections {
                    store.connections.retain(|old| old.id != connection.id);
                    store.connections.push(connection);
                }
                if !store.model_folders.iter().any(|f| f.path == folder_path) {
                    let mut id = "photocraft-models".to_owned();
                    let mut n = 2u32;
                    while store.model_folders.iter().any(|f| f.id == id) {
                        id = format!("photocraft-models-{n}");
                        n = n.saturating_add(1);
                    }
                    store.model_folders.push(craft_ai_core::model_folders::ModelFolder {
                        id,
                        label: "PhotoCraft models".into(),
                        path: folder_path,
                        default: !store.model_folders.iter().any(|f| f.default),
                        origin: craft_ai_core::model_folders::FolderOrigin::Chosen,
                        kinds: std::collections::BTreeMap::from([("background_removal".into(), vec![String::new()])]),
                    });
                }
                if !store.defaults.iter().any(|old| old.role == role.id) {
                    store.defaults.push(binding.clone());
                }
                let settings = store.apps.entry(app).or_default();
                settings.overrides.retain(|old| old.role != role.id);
                settings.overrides.push(binding);
                if !has_fill_choice || matches!(workflow, "recommended" | "lama") {
                    settings.overrides.retain(|b| b.role != fill_role.id);
                    settings.overrides.push(fill_binding);
                }
                Ok::<(), Error>(())
            })
            .map_err(error)??;
        Ok(())
    }
    fn connection(&self, runner: &Runner, main: &'static crate::ModelSpec) -> Result<Connection> {
        let mut model = runner.model();
        model.local_file = Some(self.native.shared_weights_path(main)?.to_str().ok_or_else(|| error("the model path is not valid text"))?.to_owned());
        Ok(Connection {
            id: connection_id(model.id.as_str())?,
            label: model.label.clone(),
            provider: runner.engine().into(),
            endpoint: None,
            locality: Locality::Local,
            auth: Auth::None,
            billing: Billing::Free,
            models: vec![model],
            enabled: true,
            key_header: None,
        })
    }
    pub(super) fn choice_label(&self) -> Option<String> {
        match self.client.status(&role().ok()?) {
            RoleStatus::Ready(ready) => Some(ready.model_label),
            _ => None,
        }
    }
    pub(super) fn fill_reason(&self, component: &str) -> Option<String> {
        self.reason_for(component, fill_role())
    }

    pub(super) fn reason(&self, component: &str) -> Option<String> {
        self.reason_for(component, role())
    }

    fn reason_for(&self, component: &str, role: Result<Role>) -> Option<String> {
        let role = match role {
            Ok(role) => role,
            Err(e) => return Some(e.to_string()),
        };
        match self.client.status(&role) {
            RoleStatus::Unavailable { reason } => {
                // The pinned client's unbound diagnostic names an internal role ID. The
                // application displays a human action, rather than that storage identifier.
                if reason.starts_with("no model is chosen for '") {
                    Some("Confirm an AI setup in Models to reuse your installed files.".into())
                } else {
                    Some(format!("{reason} Open Models to review this AI setup."))
                }
            }
            RoleStatus::Ready(ready) => {
                if ready.locality != Locality::Local {
                    return Some("The shared choice runs online. Choose a local Remove setup in Models.".into());
                }
                if ready.model.as_str() == component || (ready.model.as_str() == AUTOMATIC_MODEL && matches!(component, "lama" | "osor-sdxl")) {
                    self.check_selected_files(&role, &ready.model).err().map(|e| e.to_string())
                } else {
                    Some(format!("The shared choice is {}. Choose a setup containing {component} in Models.", ready.model_label))
                }
            }
        }
    }
    fn check_selected_files(&self, role: &Role, model: &ModelId) -> Result<()> {
        let main = catalog::find(if model.as_str() == AUTOMATIC_MODEL { "lama" } else { model.as_str() })?;
        let expected = self.native.shared_weights_path(main)?;
        let route = self
            .client
            .local_route_for(role)
            .map_err(|_| error("The shared AI choice changed. Review this setup in Models."))?
            .ok_or_else(|| error("Confirm an AI setup in Models."))?;
        if route.model != *model || route.provider != shared_runner::engine_id(model.as_str())? || route.endpoint.is_some() {
            return Err(error("This shared choice is not a supported in-process removal model. Choose a local setup in Models."));
        }
        if route.file.as_deref().map(Path::new) != Some(expected.as_path()) {
            return Err(error("The selected files do not match this model folder. Choose their folder and restart, or confirm the current setup in Models."));
        }
        Ok(())
    }

    pub(super) fn infer(&self, component: &str, input: &RemovalInput, ctl: &dyn RemovalControl, available: u64, for_fill: bool) -> Result<RemovalOutput> {
        let selected_role = if for_fill { fill_role()? } else { role()? };
        if let Some(reason) = self.reason_for(component, Ok(selected_role.clone())) {
            return Err(error(reason));
        }
        let ticket = self.bridge.request(input, available, component)?;
        let plan = self.client.plan(&selected_role, ticket.request()).map_err(error)?;
        // Remote and loopback routes are never sent our private float payload format.
        if plan.disclosure().locality != Locality::Local || plan.disclosure().destination != shared_runner::engine_id(plan.model().as_str())? {
            return Err(error("the shared model choice is not a supported in-process Remove adapter"));
        }
        let ledger = ConsentLedger::default();
        if plan.needs_consent(&ledger) {
            return Err(error("this removal requires consent; choose a local Remove setup in Models"));
        }
        let job = self.client.start(&plan, &ledger).map_err(error)?;
        let mut events = Vec::new();
        let mut cancelled = false;
        // TicketGuard drains the actual worker on every exit, including poll/cancel errors,
        // before the outer process-wide permit releases. Shared Cancelled is only a UI state.
        loop {
            if ctl.cancelled() && !cancelled {
                ticket.cancel();
                job.cancel().map_err(error)?;
                cancelled = true;
            }
            let snap = job.poll().map_err(error)?;
            for event in snap.events {
                if let Event::Notice { message } = &event {
                    ctl.progress(0.0, message);
                }
                events.push(event);
            }
            match snap.state {
                JobState::Succeeded => return if cancelled { Err(Error::Cancelled) } else { ticket.take_output(&events) },
                JobState::Failed { error: failure } => return if cancelled { Err(Error::Cancelled) } else { Err(error(failure)) },
                JobState::Cancelled => return Err(Error::Cancelled),
                JobState::Running { percent: Some(p) } => ctl.progress(f32::from(p) / 100.0, "Running local AI removal"),
                _ => {}
            }
            std::thread::park_timeout(std::time::Duration::from_millis(5));
        }
    }
}

pub(super) fn chosen_directory() -> Result<Option<PathBuf>> {
    let shared = SharedStore::at_default_location().map_err(error)?;
    chosen_directory_in(&shared)
}
pub(super) fn chosen_directory_in(shared: &SharedStore) -> Result<Option<PathBuf>> {
    let store = shared.load().map_err(error)?;
    // A selected native route is more specific than the download-folder default.
    if let Ok(resolved) = store.resolve(&app()?, &role()?)
        && shared_runner::engine_id(resolved.model.id.as_str()).is_ok()
        && let Some(file) = &resolved.model.local_file
        && let Some(parent) = Path::new(file).parent()
        && store.model_folders.iter().any(|f| {
            Path::new(&f.path) == parent
                && f.origin == craft_ai_core::model_folders::FolderOrigin::Chosen
                && (f.default || resolved.connection.id.as_str().starts_with("photocraft-native-"))
        })
    {
        return Ok(Some(parent.to_path_buf()));
    }
    Ok(store
        .model_folders
        .iter()
        .find(|folder| folder.default && folder.origin == craft_ai_core::model_folders::FolderOrigin::Chosen)
        .map(|folder| PathBuf::from(&folder.path)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_setup_registers_a_shared_choice_without_creating_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let store = SharedStore::new(tmp.path().join("ai/connections.json"));
        let ai = SharedAi::new(Arc::new(NativeService::new(Some(models.clone()))), store.clone()).unwrap();
        assert!(!store.path().exists(), "construction must never change shared settings");
        let message = ai.reason("lama").unwrap();
        assert!(message.contains("Confirm an AI setup"));
        assert!(!message.contains("remove-object"), "a storage role ID is not a user-facing instruction");
        ai.setup("recommended", "lama").unwrap();
        assert!(!models.exists(), "setup must not download or fabricate any weight");
        let saved = store.load().unwrap();
        assert_eq!(saved.resolve(&app().unwrap(), &role().unwrap()).unwrap().model.id.as_str(), AUTOMATIC_MODEL);
        assert!(ai.reason("lama").is_some(), "a binding alone cannot claim missing weights are ready");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(ai.native.shared_weights_path(catalog::find("lama").unwrap()).unwrap(), []).unwrap();
        assert!(ai.reason("lama").is_none());
        assert!(ai.reason("osor-sdxl").is_none());
        assert!(ai.reason("migan-v1").is_some());

        let restarted = SharedAi::new(Arc::new(NativeService::new(Some(models))), store).unwrap();
        assert!(restarted.reason("lama").is_none());
    }
    #[test]
    fn selecting_osor_for_remove_keeps_lama_background_fill_ready() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        let native = Arc::new(NativeService::new(Some(models)));
        for id in ["lama", "osor-sdxl"] {
            std::fs::write(native.shared_weights_path(catalog::find(id).unwrap()).unwrap(), []).unwrap();
        }
        let ai = SharedAi::new(native, SharedStore::new(tmp.path().join("ai/connections.json"))).unwrap();
        ai.setup("recommended", "lama").unwrap();
        ai.setup("custom", "osor-sdxl").unwrap();
        assert!(ai.reason("osor-sdxl").is_none());
        assert!(ai.fill_reason("lama").is_none(), "choosing OSOR Remove must preserve LaMa Fill");
    }
    #[test]
    fn changed_shared_weight_location_is_not_reported_ready_in_the_active_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        let native = Arc::new(NativeService::new(Some(models)));
        let expected = native.shared_weights_path(catalog::find("lama").unwrap()).unwrap();
        std::fs::write(&expected, []).unwrap();
        let store = SharedStore::new(tmp.path().join("ai/connections.json"));
        let ai = SharedAi::new(native, store.clone()).unwrap();
        ai.setup("lama", "lama").unwrap();
        assert!(ai.reason("lama").is_none());
        let other = tmp.path().join("other-models");
        std::fs::create_dir_all(&other).unwrap();
        let selected = other.join(expected.file_name().unwrap());
        std::fs::write(&selected, []).unwrap();
        store
            .update(|store| {
                let connection = store.connections.iter_mut().find(|c| c.id == connection_id("lama").unwrap()).unwrap();
                connection.models[0].local_file = Some(selected.to_string_lossy().into_owned());
            })
            .unwrap();
        let before = std::fs::read(store.path()).unwrap();
        let reason = ai.reason("lama").expect("a different selected file needs explicit recovery");
        assert!(reason.contains("model folder"), "{reason}");
        assert_eq!(std::fs::read(store.path()).unwrap(), before, "readiness cannot rewrite the shared choice");
    }

    #[cfg(all(feature = "native-inference", feature = "osor"))]
    #[test]
    #[ignore = "explicit qualification with existing PHOTOCRAFT_TEST_MODEL_DIR LaMa and OSOR weights"]
    fn real_shared_native_models_keep_warm_results_and_independent_fill() {
        let directory = PathBuf::from(std::env::var_os("PHOTOCRAFT_TEST_MODEL_DIR").expect("set existing verified models"));
        let private = tempfile::tempdir().unwrap();
        let mut service = crate::RemovalService::new(Some(directory));
        #[cfg(target_os = "macos")]
        {
            service = service.with_osor_backend(crate::OsorBackend::Metal { ordinal: 0 });
        }
        let ai = SharedAi::new(service.native.clone(), SharedStore::new(private.path().join("connections.json"))).unwrap();
        ai.setup("recommended", "lama").unwrap();
        service.shared = Some(Ok(ai));
        let (width, height) = (256u32, 256u32);
        let mut rgb = Vec::new();
        let mut mask = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgb.extend([x as f32 / width as f32, y as f32 / height as f32, 0.25]);
                mask.push(if (100..140).contains(&x) && (100..140).contains(&y) { 1.0 } else { 0.0 });
            }
        }
        let input = RemovalInput { width, height, rgb, mask };
        let transient = u64::from(width) * u64::from(height) * 128;
        let budget = 48u64 * 1024 * 1024 * 1024;
        let mut permit = service.admit_models(&["lama", "osor-sdxl"], transient, budget).unwrap();
        let cold = permit.infer("osor-sdxl", &input, &crate::NoopControl).unwrap();
        let start = std::time::Instant::now();
        let warm = permit.infer("osor-sdxl", &input, &crate::NoopControl).unwrap();
        println!(
            "shared OSOR: cold load {:.3} ms / forward {:.3} ms; warm total {:.3} ms / load {:.3} ms / forward {:.3} ms",
            cold.load_ms,
            cold.forward_ms,
            start.elapsed().as_secs_f64() * 1000.0,
            warm.load_ms,
            warm.forward_ms
        );
        assert_eq!(cold.rgb, warm.rgb);
        assert_eq!(warm.load_ms, 0.0);
        let lama = permit.infer("lama", &input, &crate::NoopControl).unwrap();
        assert_eq!(lama.rgb.len(), input.rgb.len());
        let retained = permit.infer("osor-sdxl", &input, &crate::NoopControl).unwrap();
        assert_eq!(retained.load_ms, 0.0, "switching to LaMa must preserve an admitted OSOR cache");
        drop(permit);
        service.configure_shared_setup("custom", "osor-sdxl").unwrap();
        let mut fill = service.admit_fill(transient, budget).unwrap();
        let filled = fill.infer(&input, &crate::NoopControl).unwrap();
        assert_eq!(filled.load_ms, 0.0, "Fill should reuse the same verified LaMa cache");
        assert_eq!(filled.rgb, lama.rgb);
        assert!(service.status_for_fill().iter().any(|s| s.id == "lama" && s.available));
    }

    #[test]
    fn changed_shared_choice_is_authoritative_and_a_broken_store_is_preserved() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SharedStore::new(tmp.path().join("connections.json"));
        let ai = SharedAi::new(Arc::new(NativeService::new(Some(tmp.path().join("models")))), store.clone()).unwrap();
        ai.setup("recommended", "lama").unwrap();
        ai.setup("compact", "migan-v1").unwrap();
        assert!(ai.reason("lama").is_some());
        std::fs::create_dir_all(tmp.path().join("models")).unwrap();
        std::fs::write(ai.native.shared_weights_path(catalog::find("migan-v1").unwrap()).unwrap(), []).unwrap();
        assert!(ai.reason("migan-v1").is_none());
        std::fs::write(store.path(), b"{broken}").unwrap();
        assert!(ai.setup("lama", "lama").is_err());
        assert_eq!(std::fs::read(store.path()).unwrap(), b"{broken}");
    }
}
