//! The pinned tract runtime owns lowering and arena installation; no custom kernels.
use super::{Arc, Error, ModelSpec, Result, Runnable, SymbolValues, TypedModel, model_error, runtime_for_name};
use tract_gpu::fact::DeviceTypedFactExt;

pub(super) fn prepare(graph: TypedModel, model: &ModelSpec) -> Result<(Arc<dyn Runnable>, crate::ExecutionInfo)> {
    // Preflight is fallible. Upstream's process-global initializer can still panic,
    // so the caller also contains unwinding and drains before selecting CPU.
    match tract_gpu::device::get_context() {
        Ok(context) if !context.is::<tract_metal::MetalContext>() => return Err(Error::Inference("another tract GPU context is already active".into())),
        Ok(_) => {}
        Err(_) => {
            tract_metal::MetalContext::new().map_err(model_error)?;
        }
    }
    let runtime = runtime_for_name("metal").map_err(model_error)?.ok_or_else(|| Error::Inference("tract Metal runtime is unavailable".into()))?;
    let plan: Arc<dyn Runnable> = Arc::from(runtime.prepare(graph).map_err(model_error)?);
    let graph = plan.typed_model().ok_or_else(|| Error::Inference("Metal runtime did not expose its prepared graph".into()))?;
    let mut placement = crate::GraphPlacement { gpu_nodes: 0, cpu_nodes: 0, transfer_nodes: 0, gpu_arena_bytes: 0 };
    let mut info = crate::ExecutionInfo {
        model_id: model.id.into(),
        runtime: "tract Metal".into(),
        device: "Metal system GPU".into(),
        uses_gpu: true,
        graph: None,
        fallback_reason: None,
    };
    for id in graph.eval_order().map_err(model_error)? {
        let node = graph.nodes().get(id).ok_or_else(|| Error::Inference("invalid prepared graph node".into()))?;
        if node.inputs.is_empty() {
            continue;
        }
        if node.op_as::<tract_gpu::sync::DeviceSync>().is_some() {
            placement.transfer_nodes += 1;
        } else if node.outputs.iter().any(|out| out.fact.as_device_fact().is_some_and(|fact| fact.is_from_device())) {
            placement.gpu_nodes += 1;
        } else {
            placement.cpu_nodes += 1;
        }
    }
    if placement.gpu_nodes == 0 {
        return Err(Error::Inference("this graph has no supported Metal operations".into()));
    }
    let typed_plan = plan.typed_plan().ok_or_else(|| Error::Inference("Metal runtime did not expose its memory plan".into()))?;
    let handler = tract_gpu::turn_handler::DeviceTurnHandler::for_plan(typed_plan, None)
        .map_err(model_error)?
        .ok_or_else(|| Error::Inference("Metal graph has no bounded memory arena".into()))?;
    placement.gpu_arena_bytes = u64::try_from(handler.mem_schema.eval_memory_size(&SymbolValues::default()).map_err(model_error)?)
        .map_err(|_| Error::Inference("invalid Metal memory arena size".into()))?;
    let context = tract_gpu::device::get_context().map_err(model_error)?;
    if !context.is::<tract_metal::MetalContext>() {
        return Err(Error::Inference("prepared graph has a different GPU context".into()));
    }
    info.graph = Some(placement);
    Ok((plan, info))
}

pub(super) fn drain() -> Result<()> {
    tract_metal::with_metal_stream(|stream| stream.wait_until_completed()).map_err(model_error)
}
