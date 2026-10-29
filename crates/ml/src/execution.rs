//! Completed inference diagnostics, independent of compiled capabilities.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OnnxBackend {
    #[default]
    Auto,
    Cpu,
    /// Require Metal for qualification; unsupported graphs return an error.
    Metal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionInfo {
    pub model_id: String,
    pub runtime: String,
    pub device: String,
    pub uses_gpu: bool,
    pub graph: Option<GraphPlacement>,
    pub fallback_reason: Option<String>,
}

/// Counts describe placement, not percentages of work or execution time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphPlacement {
    /// Live computational nodes, excluding constants, inputs and device transfers.
    pub gpu_nodes: usize,
    pub cpu_nodes: usize,
    pub transfer_nodes: usize,
    pub gpu_arena_bytes: u64,
}
