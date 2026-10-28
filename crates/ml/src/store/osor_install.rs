//! Explicit pinned downloads, owned preparation, and receipt-last publication.
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use super::{NativeService, capability::StagedDirectory, osor_reference, verify_pinned_reader};
use crate::{Error, ModelArtifact, RemovalControl, Result, check};
use osor_rs::{
    artifact::{ArtifactLimits, InputDigests, Manifest, SourceProvenance},
    conversion::{ConversionEvent, ConversionInput, ConversionOptions, ConversionOutput, ConversionReaders, ConversionSink, convert_with_io},
};

pub(super) fn install(service: &NativeService, ctl: &dyn RemovalControl) -> Result<()> {
    install_recipe(service, crate::catalog::OSOR_ARTIFACTS, ctl, convert)
}

fn native_error(error: Error) -> osor_rs::Error {
    match error {
        Error::Cancelled => osor_rs::Error::Cancelled,
        Error::Io(error) => osor_rs::Error::Io(error),
        other => osor_rs::Error::InvalidArtifact(other.to_string()),
    }
}

struct Sink<'a, 'b>(&'a mut StagedDirectory<'b>);
impl ConversionSink for Sink<'_, '_> {
    type Writer = File;
    fn create_new(&mut self, output: ConversionOutput) -> osor_rs::Result<File> {
        self.0.create_file(output.file_name()).map_err(native_error)
    }
    fn finish(&mut self, _: ConversionOutput, file: File) -> osor_rs::Result<()> {
        file.sync_all()?;
        self.0.ensure_owned().map_err(native_error)
    }
    fn publish_manifest(&mut self, bytes: &[u8]) -> osor_rs::Result<()> {
        self.0.publish_manifest(bytes).map_err(native_error)
    }
}

fn convert(stage: &mut StagedDirectory<'_>, files: Vec<File>, ctl: &dyn RemovalControl) -> Result<Manifest> {
    let mut files: [File; 4] = files.try_into().map_err(|_| Error::InvalidInput("OSOR conversion requires exactly four held inputs".into()))?;
    let [unet, vae, checkpoint, prompt] = &mut files;
    let [unet_pin, vae_pin, checkpoint_pin, prompt_pin] = crate::catalog::OSOR_ARTIFACTS else {
        return Err(Error::InvalidInput("OSOR compiled recipe requires exactly four pins".into()));
    };
    let base = SourceProvenance {
        repository: "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1".into(),
        revision: "115134f363124c53c7d878647567d04daf26e41e".into(),
        license: "CreativeML Open RAIL++-M".into(),
    };
    let mirror = SourceProvenance {
        repository: "https://huggingface.co/eerie-road/osor-sdxl".into(),
        revision: "a14effb4aabab39cbcbefd7517d2af147ff4d283".into(),
        license: "CreativeML Open RAIL++-M".into(),
    };
    let options = ConversionOptions {
        base_model_source: base.clone(),
        vae_source: base.clone(),
        checkpoint_source: mirror.clone(),
        prompt_source: mirror,
        config_source: Some(base),
        expected_input_sha256: InputDigests {
            base_unet: unet_pin.sha256.into(),
            base_vae: vae_pin.sha256.into(),
            checkpoint: checkpoint_pin.sha256.into(),
            prompt: prompt_pin.sha256.into(),
        },
        scheduler: osor_rs::scheduler::SchedulerConfig::default(),
        limits: ArtifactLimits { max_file_bytes: 16 * 1024 * 1024 * 1024, max_total_tensor_bytes: 24 * 1024 * 1024 * 1024, ..ArtifactLimits::default() },
    };
    let mut progress = 0.72_f32;
    convert_with_io(&options, ConversionReaders { base_unet: unet, base_vae: vae, checkpoint, prompt }, &mut Sink(stage), &mut |event| {
        check(ctl).map_err(native_error)?;
        let input_index = |input| match input {
            ConversionInput::BaseUnet => 0.,
            ConversionInput::BaseVae => 1.,
            ConversionInput::Checkpoint => 2.,
            ConversionInput::Prompt => 3.,
        };
        let (fraction, label) = match event {
            ConversionEvent::VerifyingInput { input, bytes_read, total_bytes } => {
                (0.72 + 0.03 * (input_index(input) + bytes_read as f32 / total_bytes.max(1) as f32) / 4., "Verifying OSOR conversion inputs")
            }
            ConversionEvent::ReadingInput { input, bytes_read, total_bytes } => {
                (0.75 + 0.06 * (input_index(input) + bytes_read as f32 / total_bytes.max(1) as f32) / 4., "Reading OSOR model tensors")
            }
            ConversionEvent::MergingAdapters { completed, total } => (0.81 + 0.08 * completed as f32 / total.max(1) as f32, "Merging OSOR model adapters"),
            ConversionEvent::ValidatingModel => (0.89, "Validating complete OSOR model"),
            ConversionEvent::ModelValidated => (0.9, "OSOR model validated"),
            ConversionEvent::WritingOutput { output, bytes_written, total_bytes } => {
                let index = match output {
                    ConversionOutput::Unet => 0.,
                    ConversionOutput::Vae => 1.,
                    ConversionOutput::Prompt => 2.,
                };
                (0.9 + 0.08 * (index + bytes_written as f32 / total_bytes.max(1) as f32) / 3., "Writing prepared OSOR model")
            }
            ConversionEvent::PublishingManifest => (0.985, "Publishing prepared OSOR manifest"),
        };
        progress = progress.max(fraction.clamp(0., 0.985));
        ctl.progress(progress, label);
        check(ctl).map_err(native_error)
    })
    .map_err(|error| match error {
        osor_rs::Error::Cancelled => Error::Cancelled,
        other => Error::Inference(format!("OSOR preparation: {other}")),
    })
}

/// The converter sees held input files and can create only named output files in
/// the owned workspace. Synthetic converters exercise transaction boundaries only.
pub(super) fn install_recipe(
    service: &NativeService,
    artifacts: &[ModelArtifact],
    ctl: &dyn RemovalControl,
    convert: impl FnOnce(&mut StagedDirectory<'_>, Vec<File>, &dyn RemovalControl) -> Result<Manifest>,
) -> Result<()> {
    if artifacts.is_empty() || artifacts.len() > 4 {
        return Err(Error::InvalidInput("OSOR preparation requires one bounded compiled artifact recipe".into()));
    }
    let store = service.directory(true)?;
    match store.open_regular(OsStr::new(osor_reference::RECEIPT)) {
        Ok(_) => {
            osor_reference::verify(service, ctl)?;
            return Ok(());
        }
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let downloader = service
        .downloader
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .ok_or_else(|| Error::Download("model downloading is unavailable; use Install from file".into()))?;
    let mut stage = StagedDirectory::create(&store)?;
    let mut inputs = Vec::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        check(ctl)?;
        let phase = Phase { ctl, start: index as f32 * 0.7 / artifacts.len() as f32, span: 0.7 / artifacts.len() as f32 };
        phase.progress(0., "Downloading pinned OSOR model data");
        let mut source = downloader.open_artifact(artifact)?;
        check(ctl)?;
        let mut file = stage.create_file(artifact.file_name)?;
        verify_pinned_reader(source.as_mut(), artifact.bytes, artifact.sha256, &phase, Some(&mut file))?;
        file.sync_all()?;
        file.seek(SeekFrom::Start(0))?;
        verify_pinned_reader(&mut file, artifact.bytes, artifact.sha256, &phase, None)?;
        file.seek(SeekFrom::Start(0))?;
        inputs.push(file);
        stage.ensure_owned()?;
    }
    check(ctl)?;
    stage.ensure_owned()?;
    ctl.progress(0.72, "Preparing OSOR model locally; cancellation waits for the current conversion stage");
    let manifest = convert(&mut stage, inputs, ctl)?;
    check(ctl)?;
    stage.create_file(osor_reference::LICENSE_FILE)?.write_all(crate::OSOR_LICENSE_TEXT.as_bytes())?;
    stage.create_file(osor_reference::NOTICE_FILE)?.write_all(b"PhotoCraft prepared this local OSOR-SDXL Phase II bundle from separately downloaded model data. The UNet files were modified by merging the rank-256 OSOR adapters and trained output head into the SDXL Inpainting base; tensor files were serialized as F32 safetensors. Source repositories, immutable revisions, input hashes, output hashes and numerical configuration are recorded in manifest.json. Credit for OSOR belongs to its authors: https://github.com/Zhouqm-Git/osor . The safe input mirror is https://huggingface.co/eerie-road/osor-sdxl . The base model is https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 . These weights remain subject to the accompanying CreativeML Open RAIL++-M license and its use restrictions.\n")?;
    stage.sync()?;
    for artifact in artifacts {
        stage.remove_file(artifact.file_name)?;
    }
    stage.ensure_owned()?;
    let base = service.directory.as_ref().ok_or_else(|| Error::Storage("model storage is unavailable".into()))?;
    let path = if base.is_absolute() { base.join(&stage.name) } else { std::env::current_dir()?.join(base).join(&stage.name) };
    osor_reference::import_managed(service, &path, &stage.name, &stage.store, &manifest, ctl)?;
    stage.retain();
    Ok(())
}

struct Phase<'a> {
    ctl: &'a dyn RemovalControl,
    start: f32,
    span: f32,
}
impl RemovalControl for Phase<'_> {
    fn cancelled(&self) -> bool {
        self.ctl.cancelled()
    }
    fn progress(&self, fraction: f32, label: &str) {
        self.ctl.progress(self.start + self.span * fraction.clamp(0., 1.), label);
    }
}
