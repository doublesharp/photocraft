//! Opt-in checks against retained independent oracle bytes; never downloads weights.
use super::*;
use crate::OnnxBackend;

fn frozen_input(oracle: &Path) -> RemovalInput {
    let image = fs::read(oracle.join("image.rgb")).unwrap();
    let mask = fs::read(oracle.join("mask.bin")).unwrap();
    assert_eq!(image.len(), 512 * 512 * 3);
    assert_eq!(mask.len(), 512 * 512);
    RemovalInput {
        width: 512,
        height: 512,
        rgb: image.iter().map(|v| f32::from(*v) / 255.).collect(),
        mask: mask.iter().map(|v| if *v == 0 { 0. } else { 1. }).collect(),
    }
}

fn compare(actual: &[f32], expected: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    let mut max_code = 0u8;
    let mut max_float = 0f32;
    let mut square = 0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        max_code = max_code.max(((*actual * 255.).round() as u8).abs_diff((*expected * 255.).round() as u8));
        let delta = (*actual - *expected).abs();
        max_float = max_float.max(delta);
        square += f64::from(delta).powi(2);
    }
    // Preserve the existing independent ONNX oracle's one-code-value gate.
    assert!(max_code <= 1, "decoded output differs by {max_code} code values");
    serde_json::json!({"maxAbsU8":max_code,"maxAbsFloat":max_float,"rmsFloat":(square / actual.len() as f64).sqrt()})
}

#[test]
#[ignore = "requires explicitly supplied pinned ONNX weights, independent oracle files and an unowned real-model memory window"]
fn real_onnx_cpu_metal_outputs_timings_and_drain() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let oracle = PathBuf::from(std::env::var_os("PHOTOCRAFT_ML_ORACLE_DIR").unwrap());
    let input = frozen_input(&oracle);
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("models");
    let service = RemovalService::new(Some(root.clone()));
    let mut records = Vec::new();
    let (id, variable, reference) = ("lama", "PHOTOCRAFT_ML_LAMA", "lama.rgb");
    let source = PathBuf::from(std::env::var_os(variable).unwrap());
    service.import(id, &source, &NoopControl).unwrap();
    let expected: Vec<f32> = fs::read(oracle.join(reference)).unwrap().iter().map(|v| f32::from(*v) / 255.).collect();
    let mut cpu = Vec::new();
    let transient = input.minimum_buffer_bytes_for(id).unwrap();
    for backend in [OnnxBackend::Cpu, OnnxBackend::Metal] {
        let service = RemovalService::new(Some(root.clone())).with_onnx_backend(backend);
        for run in 0..4 {
            let mut permit = service.admit(id, transient, 8 * 1024 * 1024 * 1024).unwrap();
            let out = permit.infer(&input, &NoopControl).unwrap();
            let execution = service.last_execution().unwrap();
            assert_eq!(execution.uses_gpu, backend == OnnxBackend::Metal);
            if run == 0 {
                assert!(out.load_ms > 0., "a backend change must invalidate the warm plan");
            } else {
                assert_eq!(out.load_ms, 0.);
            }
            let oracle_difference = compare(&out.rgb, &expected);
            let cpu_difference = if backend == OnnxBackend::Cpu {
                cpu = out.rgb.clone();
                None
            } else {
                Some(compare(&out.rgb, &cpu))
            };
            let reservation = permit.into_reservation();
            assert!(matches!(service.admit(id, transient, u64::MAX), Err(Error::Busy)));
            records.push(serde_json::json!({"modelId":id,"backend":backend,"run":run,"loadMs":out.load_ms,"forwardMs":out.forward_ms,
                    "execution":execution,"oracleDifference":oracle_difference,"cpuDifference":cpu_difference}));
            drop(out);
            drop(reservation);
        }
    }
    // Hardware timestamp counters are proof, measured separately because profiling
    // creates a command pass per dispatch and changes normal timing.
    let automatic = RemovalService::new(Some(root.clone()));
    let prime = automatic.infer(id, &input, &NoopControl).unwrap();
    assert!(automatic.last_execution().unwrap().uses_gpu);
    records.push(serde_json::json!({"modelId":id,"autoLoadMs":prime.load_ms,"autoForwardMs":prime.forward_ms,"execution":automatic.last_execution()}));
    drop(prime);
    let profiled = std::thread::spawn({
        let root = root.clone();
        let input = input.clone();
        move || {
            tract_metal::with_metal_stream(|stream| stream.enable_profiling()).unwrap();
            let service = RemovalService::new(Some(root)).with_onnx_backend(OnnxBackend::Auto);
            let mut permit = service.admit(id, transient, 8 * 1024 * 1024 * 1024).unwrap();
            let out = permit.infer(&input, &NoopControl).unwrap();
            assert!(service.last_execution().unwrap().uses_gpu);
            assert_eq!(out.load_ms, 0., "timestamp proof must measure only a warm forward");
            let events = tract_metal::with_metal_stream(|stream| stream.drain_profile()).unwrap();
            let elapsed_ns: u128 = events.iter().map(|(_, duration)| duration.as_nanos()).sum();
            assert!(!events.is_empty() && elapsed_ns > 0, "successful GPU timestamp dispatches are required");
            serde_json::json!({"modelId":id,"gpuTimestampDispatches":events.len(),"gpuTimestampTotalNs":elapsed_ns,
                    "profiledForwardMs":out.forward_ms,"execution":service.last_execution()})
        }
    })
    .join()
    .unwrap();
    records.push(profiled);
    // Switching back to CPU must not reuse the Auto/Metal plan.
    let cpu = RemovalService::new(Some(root.clone())).with_onnx_backend(OnnxBackend::Cpu);
    let out = cpu.infer(id, &input, &NoopControl).unwrap();
    assert!(out.load_ms > 0. && !cpu.last_execution().unwrap().uses_gpu);
    records.push(serde_json::json!({"modelId":id,"switchBackCpuLoadMs":out.load_ms}));
    // Cancellation is checked only after the noninterruptible forward drains.
    struct CancelOnForward(AtomicBool);
    impl RemovalControl for CancelOnForward {
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        fn progress(&self, _: f32, label: &str) {
            if label == "Reconstructing background locally" {
                self.0.store(true, Ordering::Release);
            }
        }
    }
    let gpu = RemovalService::new(Some(root.clone()));
    let mut permit = gpu.admit(id, transient, 8 * 1024 * 1024 * 1024).unwrap();
    assert!(matches!(permit.infer(&input, &CancelOnForward(AtomicBool::new(false))), Err(Error::Cancelled)));
    assert!(gpu.last_execution().is_none(), "cancelled output is not completed GPU evidence");
    assert!(matches!(gpu.admit(id, transient, u64::MAX), Err(Error::Busy)));
    drop(permit);
    let out = gpu.infer(id, &input, &NoopControl).unwrap();
    assert!(gpu.last_execution().unwrap().uses_gpu && out.load_ms == 0.);
    compare(&out.rgb, &expected);
    records.push(serde_json::json!({"modelId":id,"cancellationDrained":true,"nextForwardMs":out.forward_ms}));
    println!("{}", serde_json::json!({"kind":"matched CPU/Metal qualification","records":records}));
    service.remove(id).unwrap();
}

#[test]
#[ignore = "requires explicitly supplied pinned LaMa weights, independent oracle and an unowned real-model memory window"]
fn real_lama_gpu_budget_falls_back_and_recovers() {
    let _serial = ADMISSION_TEST.lock().unwrap();
    let source = PathBuf::from(std::env::var_os("PHOTOCRAFT_ML_LAMA").unwrap());
    let oracle = PathBuf::from(std::env::var_os("PHOTOCRAFT_ML_ORACLE_DIR").unwrap());
    let input = frozen_input(&oracle);
    let expected: Vec<f32> = fs::read(oracle.join("lama.rgb")).unwrap().iter().map(|v| f32::from(*v) / 255.).collect();
    let temp = tempfile::tempdir().unwrap();
    let service = RemovalService::new(Some(temp.path().join("models")));
    service.import("lama", &source, &NoopControl).unwrap();
    let transient = input.minimum_buffer_bytes_for("lama").unwrap();
    let base = u64::from(crate::catalog::find("lama").unwrap().memory_mib) * 1024 * 1024;
    let mut records = Vec::new();
    for (run, budget) in [base + transient, base + transient, 8 * 1024 * 1024 * 1024, base + transient].into_iter().enumerate() {
        let mut permit = service.admit("lama", transient, budget).unwrap();
        let out = permit.infer(&input, &NoopControl).unwrap();
        let info = service.last_execution().unwrap();
        assert_eq!(info.uses_gpu, run == 2, "only the larger budget fits the prepared GPU arena: {info:?}");
        if run == 1 {
            assert_eq!(out.load_ms, 0.);
        } else {
            assert!(out.load_ms > 0.);
        }
        if run != 2 {
            assert!(info.fallback_reason.as_ref().unwrap().contains("Metal needs an estimated"));
        }
        let difference = compare(&out.rgb, &expected);
        records.push(serde_json::json!({"run":run,"budgetBytes":budget,"transientBytes":transient,"execution":info,"loadMs":out.load_ms,
            "forwardMs":out.forward_ms,"oracleDifference":difference}));
    }
    println!("{}", serde_json::json!({"kind":"GPU arena budget fallback and recovery","runs":records}));
    // Warm metadata must never substitute for hash verification.
    let model = crate::catalog::find("lama").unwrap();
    let file = service.native.path(model, false).unwrap();
    let mut bytes = fs::read(&file).unwrap();
    bytes[1024] ^= 1;
    fs::write(&file, bytes).unwrap();
    assert!(matches!(service.infer("lama", &input, &NoopControl), Err(Error::Integrity(_))));
    assert!(service.last_execution().is_none());
    service.remove("lama").unwrap();
}
