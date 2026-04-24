use std::path::PathBuf;
use std::time::Instant;

use rayon::ThreadPoolBuilder;
use rvllm_loader::K2CpuExpertStore;
use serde_json::Value;

fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_opt_usize(k: &str) -> Option<usize> {
    std::env::var(k).ok().and_then(|s| s.parse().ok())
}

fn env_bool(k: &str, default: bool) -> bool {
    std::env::var(k)
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("k2-token0-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let token_id = env_opt_usize("K2_TOKEN_ID").unwrap_or(read_bos_token_id(&model_dir)?);
    let repeats = env_usize("K2_REPEATS", 3);
    let threads = env_usize("K2_THREADS", 16);
    let with_logits = env_bool("K2_LOGITS", true);

    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();

    let load_t0 = Instant::now();
    let store = K2CpuExpertStore::open(&model_dir).map_err(|e| format!("open: {e}"))?;
    let load_s = load_t0.elapsed().as_secs_f64();

    eprintln!(
        "k2-token0-bench: model={} token_id={} repeats={} threads={} hidden={} layers={} logits={}",
        model_dir.display(),
        token_id,
        repeats,
        threads,
        store.arch.hidden_size,
        store.arch.num_hidden_layers,
        with_logits,
    );
    eprintln!("load_s={load_s:.3}");

    let warm_t0 = Instant::now();
    let (warm_norm, warm_top1) = run_once(&store, token_id, with_logits)?;
    let warm_s = warm_t0.elapsed().as_secs_f64();
    eprintln!(
        "warmup_s={warm_s:.3} output_norm={warm_norm:.6} top1_token={}",
        warm_top1.unwrap_or(0)
    );

    let mut total_s = 0.0f64;
    let mut best_s = f64::INFINITY;
    let mut last_norm = 0.0f32;
    let mut last_top1 = None;
    for _ in 0..repeats {
        let t0 = Instant::now();
        let (out_norm, top1) = run_once(&store, token_id, with_logits)?;
        let dt = t0.elapsed().as_secs_f64();
        total_s += dt;
        best_s = best_s.min(dt);
        last_norm = out_norm;
        last_top1 = top1;
    }
    let avg_s = total_s / repeats as f64;
    println!(
        "{{\"token_id\":{},\"threads\":{},\"load_s\":{:.4},\"warmup_s\":{:.4},\"avg_s\":{:.4},\"best_s\":{:.4},\"out_l2\":{:.6},\"top1_token\":{}}}",
        token_id,
        threads,
        load_s,
        warm_s,
        avg_s,
        best_s,
        last_norm,
        last_top1.unwrap_or(0)
    );
    Ok(())
}

fn read_bos_token_id(model_dir: &std::path::Path) -> Result<usize, String> {
    let cfg_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read {}: {e}", cfg_path.display()))?;
    let v: Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", cfg_path.display()))?;
    let bos = v
        .get("bos_token_id")
        .and_then(Value::as_u64)
        .or_else(|| {
            v.get("text_config")
                .and_then(|tc| tc.get("bos_token_id"))
                .and_then(Value::as_u64)
        })
        .ok_or_else(|| format!("missing bos_token_id in {}", cfg_path.display()))?;
    usize::try_from(bos).map_err(|_| format!("bos_token_id out of range: {bos}"))
}

fn run_once(
    store: &K2CpuExpertStore,
    token_id: usize,
    with_logits: bool,
) -> Result<(f32, Option<usize>), String> {
    if with_logits {
        let logits = store
            .run_token0_logits(token_id)
            .map_err(|e| format!("run logits: {e}"))?;
        let top1 = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx);
        Ok((l2_norm(&logits), top1))
    } else {
        let hidden = store
            .run_token0_hidden(token_id)
            .map_err(|e| format!("run hidden: {e}"))?;
        Ok((l2_norm(&hidden), None))
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
