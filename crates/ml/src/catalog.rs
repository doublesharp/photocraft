use crate::{Error, Result};
use serde::Serialize;

/// Exact runtime artifact identity. OSOR uses a locally pinned reference receipt rather than
/// a redistributed artifact, so its catalog bytes/hash/URL are empty until explicit import.
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
        description: "Local object and effect removal; explicit converted bundle import, up to 512 px context",
        bytes: 0,
        sha256: "",
        url: "",
        license: "CreativeML Open RAIL++-M (weights; use restrictions)",
        source: "https://github.com/Zhouqm-Git/osor",
        backend: "osor-rs CPU / optional Metal",
        input_size: 512,
        memory_mib: 40960,
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
    },
];

pub(crate) fn find(id: &str) -> Result<&'static ModelSpec> {
    if id.len() > 128 {
        return Err(Error::InvalidInput("model identifier is too long".into()));
    }
    CATALOG.iter().find(|m| m.id == id).ok_or_else(|| Error::UnknownModel(id.into()))
}
