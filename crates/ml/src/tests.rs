use super::*;

#[test]
fn malformed_crop_is_rejected_before_runtime_or_storage() {
    for input in [
        RemovalInput { width: 0, height: 1, rgb: vec![], mask: vec![] },
        RemovalInput { width: u32::MAX, height: u32::MAX, rgb: vec![], mask: vec![] },
        RemovalInput { width: 1, height: 1, rgb: vec![0.; 2], mask: vec![1.] },
        RemovalInput { width: 1, height: 1, rgb: vec![f32::NAN, 0., 0.], mask: vec![1.] },
        RemovalInput { width: 1, height: 1, rgb: vec![0.; 3], mask: vec![f32::INFINITY] },
        RemovalInput { width: 1, height: 1, rgb: vec![0.; 3], mask: vec![0.] },
        RemovalInput { width: 1, height: 1, rgb: vec![0.; 3], mask: vec![1.] },
    ] {
        assert!(input.validate().is_err(), "hostile/empty/full-hole crop accepted: {input:?}");
    }
}

/// Opt-in integration qualification. The caller supplies pinned artifacts and independently
/// generated ONNX Runtime oracle bytes; normal tests never download or bundle model data.
#[cfg(all(not(target_arch = "wasm32"), feature = "native-inference"))]
#[test]
#[ignore = "requires pinned LaMa and MI-GAN v1 models and independent oracle data"]
fn real_models_match_reference_and_reuse_one_warm_family() {
    use std::path::PathBuf;
    let oracle = PathBuf::from(std::env::var_os("PHOTOCRAFT_ML_ORACLE_DIR").unwrap());
    let image = std::fs::read(oracle.join("image.rgb")).unwrap();
    let mask = std::fs::read(oracle.join("mask.bin")).unwrap();
    assert_eq!(image.len(), 512 * 512 * 3);
    assert_eq!(mask.len(), 512 * 512);
    let input = RemovalInput {
        width: 512,
        height: 512,
        rgb: image.iter().map(|v| f32::from(*v) / 255.).collect(),
        mask: mask.iter().map(|v| if *v > 0 { 1. } else { 0. }).collect(),
    };
    input.validate().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let service = RemovalService::new(Some(dir.path().join("models")));
    for (id, var, reference) in [("lama", "PHOTOCRAFT_ML_LAMA", "lama.rgb"), ("migan-v1", "PHOTOCRAFT_ML_MIGAN", "migan.rgb")] {
        let model = PathBuf::from(std::env::var_os(var).unwrap());
        service.import(id, &model, &NoopControl).unwrap();
        let expected = std::fs::read(oracle.join(reference)).unwrap();
        assert_eq!(expected.len(), image.len());
        let cold = service.infer(id, &input, &NoopControl).unwrap();
        let warm = service.infer(id, &input, &NoopControl).unwrap();
        assert!(cold.load_ms > 0., "switching families must load the new plan");
        assert_eq!(warm.load_ms, 0., "the same family should reuse its runnable plan");
        assert_eq!(warm.rgb.len(), expected.len());
        assert_eq!(warm.inference_scale, 1.);
        let mut max_difference = 0u8;
        let mut sum_difference = 0u64;
        for (value, expected) in warm.rgb.iter().zip(&expected) {
            assert!(value.is_finite() && (0.0..=1.0).contains(value));
            let actual = (*value * 255.).round() as u8;
            let difference = actual.abs_diff(*expected);
            max_difference = max_difference.max(difference);
            sum_difference += u64::from(difference);
        }
        assert!(max_difference <= 1, "{id} differs from reference by {max_difference} code values");
        println!(
            "{id}: max_abs_u8={max_difference}, sum_abs_u8={sum_difference}, cold_load_ms={:.1}, cold_forward_ms={:.1}, warm_load_ms={:.1}, warm_forward_ms={:.1}",
            cold.load_ms, cold.forward_ms, warm.load_ms, warm.forward_ms
        );
    }
}

#[test]
fn crop_validation_accepts_finite_fractional_coverage_and_rejects_out_of_range_samples() {
    let valid = RemovalInput { width: 2, height: 1, rgb: vec![0., 0.5, 1., 1., 0.5, 0.], mask: vec![0., 0.25] };
    assert!(valid.validate().is_ok());
    for value in [-0.01, 1.01, f32::NEG_INFINITY] {
        let mut bad_rgb = valid.clone();
        bad_rgb.rgb[0] = value;
        assert!(bad_rgb.validate().is_err());
        let mut bad_mask = valid.clone();
        bad_mask.mask[1] = value;
        assert!(bad_mask.validate().is_err());
    }
    let oversized = RemovalInput { width: 16_777_217, height: 1, rgb: vec![], mask: vec![] };
    assert!(oversized.validate().is_err());
}

#[test]
fn osor_catalog_is_discoverable_without_enabling_its_runtime() {
    let service = RemovalService::default();
    let model = service.catalog().iter().find(|model| model.id == "osor-sdxl").expect("OSOR must stay discoverable");
    assert!(model.memory_mib >= 16 * 1024);
    assert!(model.url.is_empty(), "multi-file preparation must not use the single-file transport");
    assert_eq!(model.artifacts.len(), 4);
    assert_eq!(model.download_bytes, model.artifacts.iter().map(|artifact| artifact.bytes).sum::<u64>());
    let status = service.status().into_iter().find(|status| status.id == model.id).unwrap();
    assert!(!status.installed);
    assert!(!status.available);
    #[cfg(not(feature = "osor"))]
    assert!(status.reason.unwrap().contains("OSOR"));
}

#[test]
fn model_runtime_availability_is_specific_to_the_catalog_adapter() {
    let service = RemovalService::default();
    assert!(!service.model_runtime_available("not-a-model"));
    assert_eq!(service.model_runtime_available("lama"), cfg!(all(not(target_arch = "wasm32"), feature = "native-inference")));
    assert_eq!(service.model_runtime_available("osor-sdxl"), cfg!(all(not(target_arch = "wasm32"), feature = "osor")));
    #[cfg(all(not(target_arch = "wasm32"), feature = "osor", not(any(feature = "osor-metal", target_os = "macos"))))]
    assert!(!service.with_osor_backend(OsorBackend::Metal { ordinal: 0 }).model_runtime_available("osor-sdxl"));
}
