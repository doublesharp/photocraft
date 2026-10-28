//! Real-model timing and sampled RSS for a small repair in a synthetic 24 MP document.
//! Run with a previously installed model; this example never downloads weights:
//! `PHOTOCRAFT_MODEL_DIR=/managed/models cargo run --release -p photocraft-engine
//! --features native-inference --example generative_remove_bench -- migan-v1`
//! Pass `recommended` with `--features native-inference,osor` for two independent regions. Both
//! model components must already be installed; every run verifies exact document restoration.
use std::sync::Arc;
use std::time::{Duration, Instant};

use photocraft_engine::Session;
use photocraft_ml::RemovalService;
use photocraft_testkit::perf::{RssSampler, report, row};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let id = std::env::args().nth(1).unwrap_or_else(|| "migan-v1".into());
    let hybrid = id == "recommended";
    let directory = std::env::var_os("PHOTOCRAFT_MODEL_DIR")
        .filter(|path| !path.is_empty())
        .ok_or("set PHOTOCRAFT_MODEL_DIR to a managed directory containing the installed catalog artifact")?;
    let service = RemovalService::new(Some(directory.into()));
    if !hybrid && !service.catalog().iter().any(|model| model.id == id) {
        return Err("model must be lama, migan-v1, osor-sdxl or recommended".into());
    }
    #[cfg(all(feature = "osor", target_os = "macos"))]
    let service = service.with_osor_backend(photocraft_ml::OsorBackend::Metal { ordinal: 0 });
    let mut session = Session::new();
    session.set_removal_service(Arc::new(service));
    session.edit_prefs(|prefs| {
        prefs.generative_remove.enabled = true;
        prefs.generative_remove.enabled_models = vec![id.clone()];
        prefs.generative_remove.selected_model = id.clone();
        if id == "osor-sdxl" {
            prefs.performance.memory_usage_mb = 49152;
        }
    });
    if hybrid {
        session.execute("remove.setup", json!({"workflow":"recommended","memoryUsageMb":49152}))?;
    }
    session.execute("file.new", json!({"width":6000,"height":4000,"background":"white"}))?;
    session.execute("filter.render.clouds", json!({"seed":3}))?;
    session.execute("filter.noise.addNoise", json!({"amount":8,"seed":1}))?;
    session.execute("paint.pencil", json!({"points":[[3000,1960],[3000,2040]],"size":24,"color":"#000000"}))?;
    if hybrid {
        session.execute("paint.pencil", json!({"points":[[3600,2000]],"size":48,"color":"#000000"}))?;
    }
    let original = session.active().ok_or("benchmark document missing")?.doc.clone();
    let sampler = RssSampler::start(Duration::from_millis(2));
    let mut rows = Vec::new();
    let mut runs = Vec::new();
    for method in ["classical", if hybrid { "auto" } else { "ai" }] {
        let mut samples = Vec::new();
        let mut peak = None;
        for index in 0..3 {
            sampler.reset();
            let started = Instant::now();
            let mut params = json!({"points":[[3000,1960],[3000,2040]],"size":32,"method":method});
            if hybrid {
                params = json!({"method":method,"size":32,"strokes":[
                    {"points":[[3000,1960],[3000,2040]],"repairType":"detail"},
                    {"points":[[3600,2000]],"size":64,"repairType":"object"}]});
                if method == "auto" {
                    params["removeEffects"] = json!(true);
                }
            }
            if method == "ai" {
                params["modelId"] = json!(id);
                if id == "osor-sdxl" {
                    params["preview"] = json!(true);
                    params["removeEffects"] = json!(true);
                }
            }
            let preview = session.execute("paint.remove", params)?;
            let result = if let Some(id) = preview.get("previewId").and_then(serde_json::Value::as_u64) {
                session.execute("remove.preview.apply", json!({"id":id}))?
            } else {
                preview
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let rss = sampler.peak();
            runs.push(json!({"method":method,"iteration":index,"elapsedMs":elapsed_ms,"sampledPeakRssBytes":rss,"result":result}));
            if method != "classical" && index == 0 {
                rows.push(row("AI cold small repair, 24 MP", &[elapsed_ms], rss, None));
            } else {
                peak = peak.into_iter().chain(rss).max();
                samples.push(elapsed_ms);
            }
            session.execute("edit.undo", json!({}))?;
            if session.active().is_none_or(|state| state.doc != original) {
                return Err("benchmark undo did not restore the original document".into());
            }
        }
        rows.push(row(if method != "classical" { "AI warm small repair, 24 MP" } else { "Classical small repair, 24 MP" }, &samples, peak, None));
    }
    let mut output = report(
        "generative_remove_bench",
        json!({"modelId":id,"hybrid":hybrid,"width":6000,"height":4000,"fixture":"dense procedural clouds/noise and a black stroke","platform":std::env::consts::OS,"arch":std::env::consts::ARCH}),
        rows,
        Some(&sampler),
    );
    output["runs"] = json!(runs);
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
