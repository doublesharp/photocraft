//! Small, managed receipts pin user-supplied local bundles without copying their weights.
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{
    Fingerprint, NativeService,
    capability::{Partial, StoreDir, same_file},
};
use crate::{Error, ModelStatus, RemovalControl, Result, check};
use osor_rs::artifact::{MANIFEST_FILE, Manifest, PROMPT_FILE, UNET_FILE, VAE_FILE};
use serde::{Deserialize, Serialize};

pub(super) const RECEIPT: &str = "osor-sdxl-reference.json";
const FILES: [&str; 4] = [MANIFEST_FILE, UNET_FILE, VAE_FILE, PROMPT_FILE];
const MAX_MANIFEST: u64 = 1024 * 1024;
const MAX_RECEIPT: u64 = 2 * MAX_MANIFEST;
const MAX_BUNDLE: u64 = 16 * 1024 * 1024 * 1024;
pub(crate) const MAX_DECODED_WEIGHTS: u64 = 12 * 1024 * 1024 * 1024;

/// Count F32 resident bytes from the exact header bytes included in the hash stream.
pub(super) fn decoded_header_bytes(header: &[u8], limit: u64) -> Result<u64> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| invalid(format!("invalid OSOR tensor header: {e}")))?;
    let mut count = 0usize;
    let mut decoded = 0u64;
    for (name, tensor) in header {
        if name == "__metadata__" {
            continue;
        }
        count = count.saturating_add(1);
        if count > 30_000 || name.len() > 512 || !matches!(tensor.get("dtype").and_then(|v| v.as_str()), Some("F16" | "BF16" | "F32")) {
            return Err(invalid("OSOR tensor header has an unsupported dtype, name or tensor count"));
        }
        let shape = tensor
            .get("shape")
            .and_then(|v| v.as_array())
            .filter(|shape| !shape.is_empty() && shape.len() <= 8)
            .ok_or_else(|| invalid("OSOR tensor header has an invalid shape"))?;
        let elements = shape.iter().try_fold(1u64, |count, value| {
            value
                .as_u64()
                .filter(|value| *value > 0 && *value <= 1_000_000)
                .and_then(|value| count.checked_mul(value))
                .ok_or_else(|| invalid("OSOR tensor shape is invalid or overflows"))
        })?;
        decoded = elements
            .checked_mul(4)
            .and_then(|bytes| decoded.checked_add(bytes))
            .filter(|bytes| *bytes <= limit)
            .ok_or_else(|| invalid("OSOR decoded F32 weights exceed the 12 GiB admission policy"))?;
    }
    if count == 0 {
        return Err(invalid("OSOR tensor file is empty"));
    }
    Ok(decoded)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Reference {
    schema: u32,
    pub path: PathBuf,
    pub manifest: Manifest,
    files: Vec<PinnedFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedFile {
    name: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Verified {
    path: PathBuf,
    receipt: Fingerprint,
    files: Vec<Fingerprint>,
}

/// Capabilities and open identities remain alive through model loading and inference.
pub(crate) struct HeldReference {
    pub reference: Reference,
    _source: StoreDir,
    _files: Vec<File>,
}

impl Reference {
    pub(crate) fn manifest_sha256(&self) -> Result<&str> {
        self.files.first().map(|file| file.sha256.as_str()).ok_or_else(|| invalid("missing pinned OSOR manifest digest"))
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Integrity(message.into())
}

fn bounded_read(file: &mut File, limit: u64) -> Result<Vec<u8>> {
    if file.metadata()?.len() > limit {
        return Err(invalid("OSOR receipt or manifest is oversized"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("OSOR receipt or manifest is oversized"));
    }
    Ok(bytes)
}

fn hash(file: &mut File, ctl: &dyn RemovalControl, decoded_limit: Option<u64>) -> Result<(u64, String, u64)> {
    let before = Fingerprint::of(&file.metadata()?);
    if before.bytes == 0 || before.bytes > MAX_BUNDLE {
        return Err(invalid("OSOR bundle file has an invalid or oversized length"));
    }
    let mut sha = Sha256::new();
    let mut bytes = 0u64;
    let mut decoded = 0u64;
    if let Some(limit) = decoded_limit {
        check(ctl)?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let length = u64::from_le_bytes(prefix);
        if length == 0 || length > 16 * 1024 * 1024 || length.checked_add(8).is_none_or(|length| length > before.bytes) {
            return Err(invalid("OSOR tensor header is invalid or oversized"));
        }
        let length = usize::try_from(length).map_err(|_| invalid("OSOR tensor header is too large for this host"))?;
        let mut header = vec![0u8; length];
        file.read_exact(&mut header)?;
        decoded = decoded_header_bytes(&header, limit)?;
        sha.update(prefix);
        sha.update(&header);
        bytes = 8 + length as u64;
    }
    let mut buffer = [0u8; 65_536];
    loop {
        check(ctl)?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes = bytes.checked_add(n as u64).filter(|n| *n <= before.bytes).ok_or_else(|| invalid("OSOR bundle changed while hashing"))?;
        sha.update(buffer.get(..n).ok_or_else(|| invalid("invalid OSOR stream"))?);
    }
    if bytes != before.bytes || Fingerprint::of(&file.metadata()?) != before {
        return Err(invalid("OSOR bundle changed while hashing"));
    }
    file.seek(SeekFrom::Start(0))?;
    Ok((bytes, format!("{:x}", sha.finalize()), decoded))
}

fn open_files(source: &StoreDir) -> Result<Vec<File>> {
    let files = FILES.iter().map(|name| source.open_regular(OsStr::new(name))).collect::<Result<Vec<_>>>()?;
    source.ensure_attached()?;
    Ok(files)
}

fn fingerprints(files: &[File]) -> Result<Vec<Fingerprint>> {
    files.iter().map(|file| Ok(Fingerprint::of(&file.metadata()?))).collect()
}

fn check_attached(source: &StoreDir, files: &[File]) -> Result<()> {
    source.ensure_attached()?;
    for (name, file) in FILES.iter().zip(files) {
        if !same_file(file, &source.open_regular(OsStr::new(name))?)? {
            return Err(invalid("OSOR bundle file was replaced during verification"));
        }
    }
    Ok(())
}

fn validate_reference(reference: &Reference) -> Result<()> {
    reference.manifest.validate().map_err(|e| invalid(e.to_string()))?;
    if reference.schema != 1 || !reference.path.is_absolute() || reference.files.len() != FILES.len() {
        return Err(invalid("unsupported OSOR reference receipt"));
    }
    let expected = [None, Some(&reference.manifest.sha256.unet), Some(&reference.manifest.sha256.vae), Some(&reference.manifest.sha256.prompt)];
    let mut total = 0u64;
    for ((file, name), digest) in reference.files.iter().zip(FILES).zip(expected) {
        total = total.checked_add(file.bytes).filter(|bytes| *bytes <= MAX_BUNDLE).ok_or_else(|| invalid("OSOR bundle exceeds its size limit"))?;
        if file.name != name
            || file.bytes == 0
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || digest.is_some_and(|digest| *digest != file.sha256)
        {
            return Err(invalid("OSOR receipt does not match its pinned manifest"));
        }
    }
    Ok(())
}

fn read_reference(service: &NativeService) -> Result<(StoreDir, File, Reference)> {
    let store = service.directory(false).map_err(|e| match e {
        Error::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound => Error::NotInstalled("osor-sdxl".into()),
        _ => e,
    })?;
    let mut file = store.open_regular(OsStr::new(RECEIPT)).map_err(|e| match e {
        Error::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound => Error::NotInstalled("osor-sdxl".into()),
        _ => e,
    })?;
    let bytes = bounded_read(&mut file, MAX_RECEIPT)?;
    let reference: Reference = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("invalid OSOR receipt: {e}")))?;
    validate_reference(&reference)?;
    store.ensure_attached()?;
    if !same_file(&file, &store.open_regular(OsStr::new(RECEIPT))?)? {
        return Err(invalid("OSOR receipt changed while opening"));
    }
    Ok((store, file, reference))
}

pub(super) fn import(service: &NativeService, path: &Path, ctl: &dyn RemovalControl) -> Result<()> {
    let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    let source = StoreDir::acquire(&path, false)?;
    let mut files = open_files(&source)?;
    let manifest_file = files.first_mut().ok_or_else(|| invalid("missing OSOR manifest"))?;
    let bytes = bounded_read(manifest_file, MAX_MANIFEST)?;
    manifest_file.seek(SeekFrom::Start(0))?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("invalid OSOR manifest: {e}")))?;
    manifest.validate().map_err(|e| invalid(e.to_string()))?;
    let mut pinned = Vec::new();
    let mut decoded_remaining = MAX_DECODED_WEIGHTS;
    for (name, file) in FILES.iter().zip(&mut files) {
        ctl.progress(0.05, "Verifying local OSOR bundle");
        let (bytes, sha256, decoded) = hash(file, ctl, (*name != MANIFEST_FILE).then_some(decoded_remaining))?;
        decoded_remaining = decoded_remaining.checked_sub(decoded).ok_or_else(|| invalid("OSOR decoded weight budget overflows"))?;
        pinned.push(PinnedFile { name: (*name).into(), bytes, sha256 });
    }
    if pinned.first().is_none_or(|pin| pin.sha256 != format!("{:x}", Sha256::digest(&bytes))) {
        return Err(invalid("OSOR manifest changed while importing"));
    }
    let reference = Reference { schema: 1, path, manifest, files: pinned };
    validate_reference(&reference)?;
    check_attached(&source, &files)?;
    let store = service.directory(true)?;
    let bytes = serde_json::to_vec(&reference).map_err(|e| invalid(e.to_string()))?;
    let mut partial = Partial::create(&store)?;
    partial.file.write_all(&bytes)?;
    partial.file.sync_all()?;
    partial.ensure_owned()?;
    store.ensure_attached()?;
    check(ctl)?;
    ctl.begin_irreversible()?;
    match store.dir.hard_link(&partial.name, &store.dir, RECEIPT) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let mut existing = store.open_regular(OsStr::new(RECEIPT))?;
            if bounded_read(&mut existing, MAX_RECEIPT)? != bytes {
                return Err(Error::Storage("a different OSOR bundle is already imported; remove its reference before importing another".into()));
            }
        }
        Err(e) => return Err(e.into()),
    }
    store.ensure_attached()?;
    let receipt = store.open_regular(OsStr::new(RECEIPT))?;
    let verified = Verified { path: reference.path.clone(), receipt: Fingerprint::of(&receipt.metadata()?), files: fingerprints(&files)? };
    *service.osor_verified.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(verified);
    *service.osor_invalid.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    ctl.progress(1., "Local OSOR bundle reference imported");
    Ok(())
}

pub(super) fn verify(service: &NativeService, ctl: &dyn RemovalControl) -> Result<HeldReference> {
    *service.osor_verified.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    let (store, receipt, reference) = read_reference(service)?;
    let source = StoreDir::acquire(&reference.path, false)?;
    let mut files = open_files(&source)?;
    let mut decoded_remaining = MAX_DECODED_WEIGHTS;
    for (file, pin) in files.iter_mut().zip(&reference.files) {
        ctl.progress(0.02, "Reverifying local OSOR bundle");
        let (bytes, digest, decoded) = hash(file, ctl, (pin.name != MANIFEST_FILE).then_some(decoded_remaining))?;
        decoded_remaining = decoded_remaining.checked_sub(decoded).ok_or_else(|| invalid("OSOR decoded weight budget overflows"))?;
        if bytes != pin.bytes || digest != pin.sha256 {
            return Err(invalid(format!("OSOR {} no longer matches its imported SHA-256; import the intended bundle again", pin.name)));
        }
    }
    check_attached(&source, &files)?;
    store.ensure_attached()?;
    if !same_file(&receipt, &store.open_regular(OsStr::new(RECEIPT))?)? {
        return Err(invalid("OSOR reference changed during verification"));
    }
    let verified = Verified { path: reference.path.clone(), receipt: Fingerprint::of(&receipt.metadata()?), files: fingerprints(&files)? };
    *service.osor_verified.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(verified);
    Ok(HeldReference { reference, _source: source, _files: files })
}

pub(super) fn status(service: &NativeService, runtime: bool, busy: bool) -> ModelStatus {
    let mut status = ModelStatus { id: "osor-sdxl".into(), installed: false, available: false, busy, verified: false, reason: None };
    // Receipt presence remains visible even if its source bundle was moved or corrupted.
    status.installed = service.directory(false).ok().is_some_and(|store| store.open_regular(OsStr::new(RECEIPT)).is_ok());
    let inspect = || -> Result<Verified> {
        let (_store, receipt, reference) = read_reference(service)?;
        let source = StoreDir::acquire(&reference.path, false)?;
        let files = open_files(&source)?;
        let current = fingerprints(&files)?;
        if current.iter().zip(&reference.files).any(|(actual, pin)| actual.bytes != pin.bytes) {
            return Err(invalid("OSOR bundle file size changed; import the intended bundle again"));
        }
        Ok(Verified { path: reference.path, receipt: Fingerprint::of(&receipt.metadata()?), files: current })
    };
    match inspect() {
        Ok(current) => {
            status.verified = service.osor_verified.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() == Some(&current);
            status.available = runtime;
            if !runtime {
                status.reason = Some("OSOR runtime is unavailable in this build".into());
            } else if !status.verified {
                status.reason = Some("Local bundle reference found; bytes will be reverified before inference".into());
            }
        }
        Err(error) => status.reason = Some(error.to_string()),
    }
    if let Some(reason) = service.osor_invalid.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() {
        status.available = false;
        status.verified = false;
        status.reason = Some(reason.clone());
    }
    status
}

pub(super) fn identity(service: &NativeService) -> Result<String> {
    let (_store, _file, reference) = read_reference(service)?;
    Ok(reference.manifest_sha256()?.into())
}

pub(super) fn remove(service: &NativeService) -> Result<()> {
    crate::osor_runtime::clear_cache();
    *service.osor_invalid.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    *service.osor_verified.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    let store = match service.directory(false) {
        Ok(store) => store,
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    service.remove_in(&store, crate::catalog::find("osor-sdxl")?, || {})
}
