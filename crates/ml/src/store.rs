//! Managed, content-pinned model files. One service admits one resource owner at a time.
mod capability;
#[cfg(feature = "osor")]
mod osor_install;
#[cfg(feature = "osor")]
pub(crate) mod osor_reference;

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use crate::{Error, ModelDownloader, ModelSpec, ModelStatus, RemovalBackend, RemovalControl, RemovalInput, RemovalOutput, Result, check};
use capability::{Partial, Quarantine, StoreDir, same_file};
use sha2::{Digest, Sha256};

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fingerprint {
    bytes: u64,
    modified: Option<SystemTime>,
    created: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}
impl Fingerprint {
    fn of(meta: &fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            bytes: meta.len(),
            modified: meta.modified().ok(),
            created: meta.created().ok(),
            #[cfg(unix)]
            device: meta.dev(),
            #[cfg(unix)]
            inode: meta.ino(),
        }
    }
}

#[derive(Default)]
struct State {
    verified: HashMap<&'static str, Fingerprint>,
}

pub(crate) struct NativeService {
    pub(super) directory: Option<PathBuf>,
    onnx_backend: Mutex<crate::OnnxBackend>,
    execution: Mutex<Option<crate::ExecutionInfo>>,
    downloader: Mutex<Option<Arc<dyn ModelDownloader>>>,
    state: Mutex<State>,
    #[cfg(feature = "osor")]
    osor_backend: Mutex<crate::OsorBackend>,
    #[cfg(feature = "osor")]
    osor_verified: Mutex<Option<osor_reference::Verified>>,
    #[cfg(feature = "osor")]
    osor_invalid: Mutex<Option<String>>,
}

// Services can be constructed for separate sessions. They must not each admit a separate
// model working set or keep their own warm family in the same process.
pub(crate) static BUSY: AtomicBool = AtomicBool::new(false);

/// Process-wide ownership that can travel with an unconsumed worker result.
pub(crate) struct Lease;
impl Drop for Lease {
    fn drop(&mut self) {
        crate::cache::idle();
    }
}

impl NativeService {
    pub fn new(directory: Option<PathBuf>) -> Self {
        Self {
            directory,
            onnx_backend: Mutex::new(crate::OnnxBackend::Auto),
            execution: Mutex::new(None),
            downloader: Mutex::new(None),
            state: Mutex::new(State::default()),
            #[cfg(feature = "osor")]
            osor_backend: Mutex::new(crate::OsorBackend::Cpu),
            #[cfg(feature = "osor")]
            osor_verified: Mutex::new(None),
            #[cfg(feature = "osor")]
            osor_invalid: Mutex::new(None),
        }
    }

    pub fn set_onnx_backend(&self, backend: crate::OnnxBackend) {
        *self.onnx_backend.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = backend;
    }

    pub fn clear_execution(&self) {
        *self.execution.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    pub fn last_execution(&self) -> Option<crate::ExecutionInfo> {
        self.execution.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn begin(&self) -> Result<Lease> {
        BUSY.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).map(|_| Lease).map_err(|_| Error::Busy)
    }

    pub(crate) fn prepare_model(&self, id: &str, injected: bool, available_bytes: u64) {
        crate::cache::prepare(id, injected, available_bytes);
    }

    #[cfg(feature = "osor")]
    pub fn osor_runtime_available(&self) -> bool {
        let backend = *self.osor_backend.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(target_os = "macos")]
        {
            matches!(backend, crate::OsorBackend::Cpu | crate::OsorBackend::Metal { .. })
        }
        #[cfg(not(target_os = "macos"))]
        {
            matches!(backend, crate::OsorBackend::Cpu)
        }
    }

    #[cfg(feature = "osor")]
    pub fn model_sha256(&self) -> Result<String> {
        osor_reference::identity(self)
    }

    #[cfg(feature = "osor")]
    pub fn set_osor_backend(&self, backend: crate::OsorBackend) {
        *self.osor_backend.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = backend;
    }

    #[cfg(feature = "osor")]
    pub fn infer_osor(&self, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
        let reference = match osor_reference::verify(self, ctl) {
            Ok(reference) => reference,
            Err(error) => {
                if !matches!(error, Error::Cancelled) {
                    *self.osor_invalid.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
                }
                return Err(error);
            }
        };
        let backend = *self.osor_backend.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (out, info) = crate::osor_runtime::infer(&reference.reference, backend, input, ctl)?;
        check(ctl)?;
        validate_output(input, &out)?;
        *self.execution.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(info);
        Ok(out)
    }

    pub fn set_downloader(&self, downloader: Arc<dyn ModelDownloader>) {
        *self.downloader.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(downloader);
    }

    pub fn management_available(&self) -> bool {
        self.directory.is_some()
    }

    fn directory(&self, create: bool) -> Result<StoreDir> {
        let directory =
            self.directory.as_deref().ok_or_else(|| Error::Storage("model storage is unavailable; choose a writable local model directory".into()))?;
        StoreDir::acquire(directory, create)
    }

    fn file_name(model: &ModelSpec) -> String {
        if model.id == "osor-sdxl" { "osor-sdxl-reference.json".into() } else { format!("{}-{}.onnx", model.id, model.sha256) }
    }

    // Path exposure is retained only for fixture assertions; production I/O uses StoreDir.
    #[cfg(test)]
    fn path(&self, model: &ModelSpec, create: bool) -> Result<PathBuf> {
        self.directory(create)?;
        self.directory.as_deref().map(|d| d.join(Self::file_name(model))).ok_or_else(|| Error::Storage("model storage is unavailable".into()))
    }

    pub fn status(&self, runtime: impl Fn(&str) -> bool) -> Vec<ModelStatus> {
        let busy = BUSY.load(Ordering::Acquire);
        crate::catalog::CATALOG
            .iter()
            .map(|model| {
                let runtime = runtime(model.id);
                if model.id == "osor-sdxl" {
                    #[cfg(feature = "osor")]
                    return osor_reference::status(self, runtime, busy);
                    #[cfg(not(feature = "osor"))]
                    return ModelStatus {
                        id: model.id.into(),
                        installed: self.directory(false).ok().is_some_and(|dir| dir.open_regular(OsStr::new(&Self::file_name(model))).is_ok()),
                        available: false,
                        busy,
                        verified: false,
                        reason: Some("OSOR inference and model preparation require the native osor feature".into()),
                    };
                }
                let fingerprint = self.directory(false).ok().and_then(|dir| {
                    let file = dir.open_regular(OsStr::new(&Self::file_name(model))).ok()?;
                    dir.ensure_attached().ok()?;
                    let metadata = file.metadata().ok()?;
                    Some(Fingerprint::of(&metadata))
                });
                // Presence is separate from readiness: even a truncated owned file must be
                // visible to the manager so the user can remove it and install again.
                let installed = fingerprint.is_some();
                let correct_size = fingerprint.as_ref().is_some_and(|f| f.bytes == model.bytes);
                let verified = fingerprint.as_ref().is_some_and(|f| self.state().verified.get(model.id) == Some(f));
                let reason = if !runtime {
                    Some("native AI inference is unavailable in this build".into())
                } else if self.directory.is_none() {
                    Some("model storage is unavailable".into())
                } else if !installed {
                    Some("model is not installed".into())
                } else if !correct_size {
                    Some("The installed model has an unexpected size; remove it and install again".into())
                } else {
                    None
                };
                ModelStatus { id: model.id.into(), installed, available: runtime && correct_size, busy, verified, reason }
            })
            .collect()
    }

    #[cfg(any(test, feature = "native-inference"))]
    fn verify_open(&self, model: &'static ModelSpec, ctl: &dyn RemovalControl) -> Result<(File, Fingerprint)> {
        self.state().verified.remove(model.id);
        let dir = self.directory(false).map_err(|e| match e {
            Error::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound => Error::NotInstalled(model.id.into()),
            _ => e,
        })?;
        self.verify_in(&dir, model, ctl)
    }

    fn verify_in(&self, dir: &StoreDir, model: &'static ModelSpec, ctl: &dyn RemovalControl) -> Result<(File, Fingerprint)> {
        check(ctl)?;
        self.state().verified.remove(model.id);
        let name = Self::file_name(model);
        let mut file = dir.open_regular(OsStr::new(&name)).map_err(|e| match e {
            Error::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound => Error::NotInstalled(model.id.into()),
            _ => e,
        })?;
        let fingerprint = Fingerprint::of(&file.metadata()?);
        if fingerprint.bytes != model.bytes {
            return Err(Error::Integrity(format!("{} has an unexpected size; remove it and install again", model.label)));
        }
        ctl.progress(0.02, "Verifying model");
        verify_reader(&mut file, model, ctl, None)?;
        if Fingerprint::of(&file.metadata()?) != fingerprint {
            return Err(Error::Integrity("model data changed during verification".into()));
        }
        dir.ensure_attached()?;
        let current = dir.open_regular(OsStr::new(&name))?;
        if !same_file(&file, &current)? {
            return Err(Error::Storage("model file was replaced during verification".into()));
        }
        self.state().verified.insert(model.id, fingerprint.clone());
        file.seek(SeekFrom::Start(0))?;
        Ok((file, fingerprint))
    }

    pub fn import(&self, model: &'static ModelSpec, path: &Path, ctl: &dyn RemovalControl) -> Result<()> {
        let _lease = self.begin()?;
        check(ctl)?;
        if model.id == "osor-sdxl" {
            #[cfg(feature = "osor")]
            return osor_reference::import(self, path, ctl);
            #[cfg(not(feature = "osor"))]
            return Err(Error::Unavailable);
        }
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let name = path.file_name().ok_or_else(|| Error::Storage("select a regular model file".into()))?;
        let source_dir = StoreDir::acquire(parent, false)?;
        let mut source = source_dir.open_regular(name)?;
        source_dir.ensure_attached()?;
        if source.metadata()?.len() != model.bytes {
            return Err(Error::Integrity(format!("expected {} bytes for {}", model.bytes, model.label)));
        }
        self.publish(model, &mut source, ctl)
    }

    pub fn install(&self, model: &'static ModelSpec, ctl: &dyn RemovalControl) -> Result<()> {
        let _lease = self.begin()?;
        check(ctl)?;
        if model.id == "osor-sdxl" {
            #[cfg(feature = "osor")]
            {
                self.prepare_model("", true, 0);
                return osor_install::install(self, ctl);
            }
            #[cfg(not(feature = "osor"))]
            return Err(Error::Unavailable);
        }
        let dir = self.directory(true)?;
        match self.verify_in(&dir, model, ctl) {
            Ok(_) => return Ok(()),
            Err(Error::NotInstalled(_)) => {}
            Err(e) => return Err(e),
        }
        let downloader = self
            .downloader
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| Error::Download("model downloading is unavailable; use Install from file".into()))?;
        let mut source = downloader.open(model)?;
        self.publish_in(&dir, model, &mut source, ctl)
    }

    fn publish(&self, model: &'static ModelSpec, reader: &mut dyn Read, ctl: &dyn RemovalControl) -> Result<()> {
        let dir = self.directory(true)?;
        self.publish_in(&dir, model, reader, ctl)
    }

    fn publish_in(&self, dir: &StoreDir, model: &'static ModelSpec, reader: &mut dyn Read, ctl: &dyn RemovalControl) -> Result<()> {
        self.publish_in_with(dir, model, reader, ctl, || {})
    }

    fn publish_in_with(
        &self,
        dir: &StoreDir,
        model: &'static ModelSpec,
        reader: &mut dyn Read,
        ctl: &dyn RemovalControl,
        before_link: impl FnOnce(),
    ) -> Result<()> {
        check(ctl)?;
        let mut partial = Partial::create(dir)?;
        verify_reader(reader, model, ctl, Some(&mut partial.file))?;
        partial.file.sync_all()?;
        check(ctl)?;
        // Verify the file that will be linked, rather than trusting only the incoming stream.
        partial.file.seek(SeekFrom::Start(0))?;
        verify_reader(&mut partial.file, model, ctl, None)?;
        dir.ensure_attached()?;
        partial.ensure_owned()?;
        check(ctl)?;
        before_link();
        let name = Self::file_name(model);
        ctl.begin_irreversible()?;
        match dir.dir.hard_link(&partial.name, &dir.dir, &name) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                self.verify_in(dir, model, &CommittedControl(ctl))?;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }
        let published = dir.open_regular(OsStr::new(&name));
        let valid = published.as_ref().is_ok_and(|file| same_file(file, &partial.file).unwrap_or(false));
        if !valid || dir.ensure_attached().is_err() {
            // Check the observed entry before cleanup. Portable unlink is still name-based,
            // so a writer with access to this directory can race the check itself.
            match published {
                Ok(file) => {
                    if dir.open_regular(OsStr::new(&name)).is_ok_and(|current| same_file(&file, &current).unwrap_or(false)) {
                        let _ = dir.dir.remove_file(&name);
                    }
                }
                Err(_) => {
                    if dir.dir.symlink_metadata(&name).is_ok_and(|metadata| !metadata.is_file()) {
                        // Removing the directory entry itself never follows a substituted symlink.
                        let _ = dir.dir.remove_file(&name);
                    }
                }
            }
            return Err(Error::Storage("model directory or staged file changed during publication".into()));
        }
        let published = published?;
        ctl.progress(1., "Model installed");
        let still_valid = dir.ensure_attached();
        let current = dir.open_regular(OsStr::new(&name));
        if still_valid.is_err() || !current.as_ref().is_ok_and(|file| same_file(&published, file).unwrap_or(false)) {
            if current.as_ref().is_ok_and(|file| same_file(&published, file).unwrap_or(false)) {
                let _ = dir.dir.remove_file(&name);
            }
            return Err(still_valid.err().unwrap_or_else(|| Error::Storage("model directory or file changed during publication".into())));
        }
        self.state().verified.insert(model.id, Fingerprint::of(&published.metadata()?));
        Ok(())
    }

    pub fn remove(&self, model: &'static ModelSpec) -> Result<()> {
        let _lease = self.begin()?;
        crate::cache::clear_model(model.id);
        #[cfg(feature = "osor")]
        if model.id == "osor-sdxl" {
            return osor_reference::remove(self);
        }
        self.state().verified.remove(model.id);
        let dir = match self.directory(false) {
            Ok(dir) => dir,
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        self.remove_in(&dir, model, || {})
    }

    fn remove_in(&self, dir: &StoreDir, model: &'static ModelSpec, before_unlink: impl FnOnce()) -> Result<()> {
        let name = Self::file_name(model);
        let file = match dir.open_regular(OsStr::new(&name)) {
            Ok(file) => file,
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        dir.ensure_attached()?;
        let current = dir.open_regular(OsStr::new(&name))?;
        if !same_file(&file, &current)? {
            return Err(Error::Storage("model file was replaced during removal".into()));
        }
        let quarantine = Quarantine::create(dir)?;
        before_unlink();
        quarantine.move_entry(&name)?;
        let validated = dir
            .ensure_attached()
            .and_then(|_| quarantine.ensure_attached())
            .and_then(|_| if quarantine.matches(&file)? { Ok(()) } else { Err(Error::Storage("model file was replaced during removal".into())) });
        if let Err(error) = validated {
            return match quarantine.restore(&name) {
                Ok(()) => Err(error),
                Err(_) => Err(quarantine.recovery_error(error)),
            };
        }
        if let Err(error) = quarantine.remove_entry() {
            return match quarantine.restore(&name) {
                Ok(()) => Err(error),
                Err(_) => Err(quarantine.recovery_error(error)),
            };
        }
        self.state().verified.remove(model.id);
        crate::cache::clear_model(model.id);
        Ok(())
    }

    pub fn infer_backend(&self, backend: &dyn RemovalBackend, id: &str, input: &RemovalInput, ctl: &dyn RemovalControl) -> Result<RemovalOutput> {
        check(ctl)?;
        if !backend.ready(id) {
            return Err(Error::NotInstalled(id.into()));
        }
        let out = backend.infer(id, input, ctl)?;
        check(ctl)?;
        validate_output(input, &out)?;
        // Injected adapters do not expose a verified device contract. Leave execution
        // unknown rather than classifying an arbitrary backend as CPU or GPU.
        Ok(out)
    }

    #[cfg(feature = "native-inference")]
    pub fn infer(&self, model: &'static ModelSpec, input: &RemovalInput, ctl: &dyn RemovalControl, available_bytes: u64) -> Result<RemovalOutput> {
        check(ctl)?;
        let backend = *self.onnx_backend.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (file, fingerprint) = self.verify_open(model, ctl)?;
        let old = crate::cache::take_onnx();
        let request = crate::runtime::Request { backend, available_bytes, fallback: None };
        // The lease outlives preparation, forward, synchronization and retained output.
        let (mut cached, mut load_ms) = crate::runtime::load(model, file, fingerprint, old, request, ctl)?;
        crate::cache::prepare_required(model.id, cached.working_bytes, available_bytes);
        let mut forward = crate::runtime::forward(model, &cached, input, ctl);
        if cached.info.uses_gpu
            && backend == crate::OnnxBackend::Auto
            && forward.drained
            && let Err(error) = &forward.result
            && !matches!(error, Error::Cancelled)
        {
            let reason = format!("Metal forward failed; using the same model on CPU: {error}");
            drop(cached);
            check(ctl)?;
            let (file, fingerprint) = self.verify_open(model, ctl)?;
            let request = crate::runtime::Request { backend, available_bytes, fallback: Some(reason) };
            let (cpu, cpu_load_ms) = crate::runtime::load(model, file, fingerprint, None, request, ctl)?;
            cached = cpu;
            load_ms += cpu_load_ms;
            crate::cache::prepare_required(model.id, cached.working_bytes, available_bytes);
            forward = crate::runtime::forward(model, &cached, input, ctl);
        }
        let info = cached.info.clone();
        // A failed synchronization is never retained as a usable warm plan.
        if forward.drained {
            crate::cache::put_onnx(cached);
        }
        let mut out = forward.result?;
        out.load_ms = load_ms;
        check(ctl)?;
        validate_output(input, &out)?;
        *self.execution.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(info);
        Ok(out)
    }
}

// Once an irreversible commit starts, even the idempotent existing-file path must not
// return Cancelled. The host's gate serializes cancellation with publication.
struct CommittedControl<'a>(&'a dyn RemovalControl);
impl RemovalControl for CommittedControl<'_> {
    fn cancelled(&self) -> bool {
        false
    }
    fn progress(&self, fraction: f32, label: &str) {
        self.0.progress(fraction, label);
    }
}

fn validate_output(input: &RemovalInput, out: &RemovalOutput) -> Result<()> {
    if out.edit_alpha.as_ref().is_some_and(|alpha| alpha.len() != input.mask.len() || alpha.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
        || out.rgb.len() != input.rgb.len()
        || out.rgb.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || !out.inference_scale.is_finite()
        || !(0.0..=1.0).contains(&out.inference_scale)
        || out.inference_scale == 0.0
        || !out.load_ms.is_finite()
        || out.load_ms < 0.0
        || !out.forward_ms.is_finite()
        || out.forward_ms < 0.0
    {
        return Err(Error::Inference("adapter returned malformed image data or diagnostics".into()));
    }
    Ok(())
}

#[cfg(any(test, feature = "native-inference"))]
pub(crate) fn verified_snapshot(file: &mut File, model: &ModelSpec, ctl: &dyn RemovalControl) -> Result<Vec<u8>> {
    let size = usize::try_from(model.bytes).map_err(|_| Error::Inference("model is too large for this platform".into()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| Error::Inference("not enough memory to load the verified model".into()))?;
    verify_reader(file, model, ctl, Some(&mut bytes))?;
    Ok(bytes)
}

fn verify_reader(reader: &mut dyn Read, model: &ModelSpec, ctl: &dyn RemovalControl, write: Option<&mut dyn Write>) -> Result<()> {
    verify_pinned_reader(reader, model.bytes, model.sha256, ctl, write)
}

fn verify_pinned_reader(
    reader: &mut dyn Read,
    expected_bytes: u64,
    expected_sha256: &str,
    ctl: &dyn RemovalControl,
    mut write: Option<&mut dyn Write>,
) -> Result<()> {
    let mut sha = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65_536];
    loop {
        check(ctl)?;
        let limit = (expected_bytes.saturating_sub(bytes).saturating_add(1)).min(buffer.len() as u64) as usize;
        let slice = buffer.get_mut(..limit).ok_or_else(|| Error::Integrity("invalid model byte bound".into()))?;
        let n = reader.read(slice)?;
        if n == 0 {
            break;
        }
        bytes = bytes.saturating_add(n as u64);
        if bytes > expected_bytes {
            return Err(Error::Integrity("download exceeds the pinned model size".into()));
        }
        let part = slice.get(..n).ok_or_else(|| Error::Integrity("invalid model stream".into()))?;
        sha.update(part);
        if let Some(file) = write.as_deref_mut() {
            file.write_all(part)?;
        }
        ctl.progress((bytes as f32 / expected_bytes.max(1) as f32) * 0.98, "Verifying model data");
    }
    if bytes != expected_bytes {
        return Err(Error::Integrity(format!("expected {expected_bytes} model bytes, received {bytes}")));
    }
    let digest = hex_digest(sha.finalize());
    if digest != expected_sha256 {
        return Err(Error::Integrity("SHA-256 does not match the pinned model; the previous install is unchanged".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
