use crate::{Error, Result};
use serde::Serialize;

/// A pinned, inert model-data download. IDs and destinations are never accepted from the network.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelArtifact {
    pub id: &'static str,
    pub file_name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    pub url: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub source: &'static str,
}

/// Exact runtime artifact identity and explicit preparation resource estimates.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    pub url: &'static str,
    pub license: &'static str,
    pub source: &'static str,
    pub backend: &'static str,
    pub input_size: u32,
    pub memory_mib: u32,
    pub download_bytes: u64,
    pub install_bytes: u64,
    pub preparation_workspace_bytes: u64,
    pub preparation_memory_mib: u32,
    pub license_url: &'static str,
    pub artifacts: &'static [ModelArtifact],
}

pub const OSOR_PREPARATION_BYTES: u64 = 40 * 1024 * 1024 * 1024;

pub static OSOR_ARTIFACTS: &[ModelArtifact] = &[
    ModelArtifact {
        id: "osor-sdxl-base-unet",
        file_name: "base-unet.safetensors",
        bytes: 5_135_178_560,
        sha256: "6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f",
        url: "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/115134f363124c53c7d878647567d04daf26e41e/unet/diffusion_pytorch_model.fp16.safetensors",
        license: "CreativeML Open RAIL++-M",
        license_url: "https://huggingface.co/eerie-road/osor-sdxl/raw/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md",
        source: "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e",
    },
    ModelArtifact {
        id: "osor-sdxl-base-vae",
        file_name: "base-vae.safetensors",
        bytes: 167_335_338,
        sha256: "4ad62825e5c8b31eefb77355ba6693785619bf7d669d9b1a6fd9f19dec6d65b3",
        url: "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/115134f363124c53c7d878647567d04daf26e41e/vae/diffusion_pytorch_model.fp16.safetensors",
        license: "CreativeML Open RAIL++-M",
        license_url: "https://huggingface.co/eerie-road/osor-sdxl/raw/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md",
        source: "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e",
    },
    ModelArtifact {
        id: "osor-sdxl-phase2",
        file_name: "phase2.safetensors",
        bytes: 3_111_950_972,
        sha256: "fff6bea76360666f1233dbf74840f45a438140f2c0f093365ef017ebd7a457ff",
        url: "https://huggingface.co/eerie-road/osor-sdxl/resolve/a14effb4aabab39cbcbefd7517d2af147ff4d283/sdxlinpainting_phase2.safetensors",
        license: "CreativeML Open RAIL++-M",
        license_url: "https://huggingface.co/eerie-road/osor-sdxl/raw/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md",
        source: "https://huggingface.co/eerie-road/osor-sdxl/tree/a14effb4aabab39cbcbefd7517d2af147ff4d283",
    },
    ModelArtifact {
        id: "osor-sdxl-fixed-prompt",
        file_name: "fixed-prompt.safetensors",
        bytes: 636_136,
        sha256: "93bc3ef2fe9e1616c164450aa58f9f34ba23569215722158d83a3fb5ad157e50",
        url: "https://huggingface.co/eerie-road/osor-sdxl/resolve/a14effb4aabab39cbcbefd7517d2af147ff4d283/fixed_prompt_embeds.safetensors",
        license: "CreativeML Open RAIL++-M",
        license_url: "https://huggingface.co/eerie-road/osor-sdxl/raw/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md",
        source: "https://huggingface.co/eerie-road/osor-sdxl/tree/a14effb4aabab39cbcbefd7517d2af147ff4d283",
    },
];

/// Resolve only a compiled, approved data artifact; app transports must reject arbitrary URLs.
pub fn model_artifact(id: &str) -> Result<&'static ModelArtifact> {
    OSOR_ARTIFACTS.iter().find(|artifact| artifact.id == id).ok_or_else(|| Error::UnknownModel(id.chars().take(128).collect()))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub installed: bool,
    pub available: bool,
    pub busy: bool,
    pub verified: bool,
    pub reason: Option<String>,
}

impl ModelStatus {
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn unavailable(model: &ModelSpec, reason: &str) -> Self {
        Self { id: model.id.into(), installed: false, available: false, busy: false, verified: false, reason: Some(reason.into()) }
    }
}

pub(crate) const CATALOG: &[ModelSpec] = &[
    ModelSpec {
        id: "osor-sdxl",
        label: "OSOR-SDXL Phase II",
        description: "Local object and effect removal; explicit download and preparation or local bundle import, up to 512 px context",
        bytes: 0,
        sha256: "",
        url: "",
        license: "CreativeML Open RAIL++-M (weights; use restrictions)",
        source: "https://github.com/Zhouqm-Git/osor",
        backend: "osor CPU / optional Metal",
        input_size: 512,
        memory_mib: 40960,
        download_bytes: 8_415_101_006,
        install_bytes: 10_606_000_000,
        preparation_workspace_bytes: 19_100_000_000,
        preparation_memory_mib: 40960,
        license_url: "https://huggingface.co/eerie-road/osor-sdxl/raw/a14effb4aabab39cbcbefd7517d2af147ff4d283/LICENSE.md",
        artifacts: OSOR_ARTIFACTS,
    },
    ModelSpec {
        id: "lama",
        label: "LaMa",
        description: "Local background reconstruction; 512 px context, CPU processing",
        bytes: 208_044_816,
        sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx",
        license: "Apache-2.0",
        source: "https://github.com/advimman/lama",
        backend: "tract CPU",
        input_size: 512,
        memory_mib: 3072,
        download_bytes: 208_044_816,
        install_bytes: 208_044_816,
        preparation_workspace_bytes: 208_044_816,
        preparation_memory_mib: 0,
        license_url: "https://github.com/advimman/lama/blob/main/LICENSE",
        artifacts: &[],
    },
    ModelSpec {
        id: "migan-v1",
        label: "MI-GAN v1",
        description: "Compact local reconstruction; exact older MI-GAN pipeline, 512 px CPU context",
        bytes: 29_546_882,
        sha256: "593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae",
        url: "https://huggingface.co/andraniksargsyan/migan/resolve/406830d0fa60666da0071c342ad2fbc8f30c5c64/migan.onnx",
        license: "MIT",
        source: "https://github.com/Picsart-AI-Research/MI-GAN",
        backend: "tract CPU",
        input_size: 512,
        memory_mib: 1536,
        download_bytes: 29_546_882,
        install_bytes: 29_546_882,
        preparation_workspace_bytes: 29_546_882,
        preparation_memory_mib: 0,
        license_url: "https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE",
        artifacts: &[],
    },
];

pub(crate) fn find(id: &str) -> Result<&'static ModelSpec> {
    if id.len() > 128 {
        return Err(Error::InvalidInput("model identifier is too long".into()));
    }
    CATALOG.iter().find(|m| m.id == id).ok_or_else(|| Error::UnknownModel(id.into()))
}
