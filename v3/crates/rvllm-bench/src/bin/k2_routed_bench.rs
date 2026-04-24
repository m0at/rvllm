use std::path::PathBuf;
use std::time::Instant;

use rayon::ThreadPoolBuilder;
use rvllm_loader::K2CpuExpertStore;

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

fn env_experts(k: &str) -> Result<Vec<usize>, String> {
    let raw = std::env::var(k).unwrap_or_else(|_| "0,1,2,3,4,5,6,7".to_string());
    raw.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("bad expert id '{s}': {e}"))
        })
        .collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("k2-routed-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let layer = env_usize("K2_LAYER", 1);
    let repeats = env_usize("K2_REPEATS", 3);
    let threads = env_usize("K2_THREADS", 16);
    let experts = env_experts("K2_EXPERTS")?;
    if experts.is_empty() {
        return Err("K2_EXPERTS must not be empty".into());
    }

    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();

    let load_t0 = Instant::now();
    let store = K2CpuExpertStore::open(&model_dir).map_err(|e| format!("open: {e}"))?;
    let load_s = load_t0.elapsed().as_secs_f64();

    let mut x = vec![0.0f32; store.arch.hidden_size];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 97) as f32 - 48.0) / 48.0;
    }
    let weights = vec![1.0f32 / experts.len() as f32; experts.len()];

    eprintln!(
        "k2-routed-bench: model={} layer={} experts={:?} repeats={} threads={} hidden={} moe={}",
        model_dir.display(),
        layer,
        experts,
        repeats,
        threads,
        store.arch.hidden_size,
        store.arch.moe_intermediate_size,
    );
    eprintln!("load_s={load_s:.3}");

    let warm_t0 = Instant::now();
    let warm = store
        .run_routed_topk(layer, &experts, &weights, &x)
        .map_err(|e| format!("warmup: {e}"))?;
    let warm_s = warm_t0.elapsed().as_secs_f64();
    eprintln!("warmup_s={warm_s:.3} output_norm={:.6}", l2_norm(&warm));

    let mut total_s = 0.0f64;
    let mut best_s = f64::INFINITY;
    let mut last_norm = 0.0f32;
    for _ in 0..repeats {
        let t0 = Instant::now();
        let out = store
            .run_routed_topk(layer, &experts, &weights, &x)
            .map_err(|e| format!("run: {e}"))?;
        let dt = t0.elapsed().as_secs_f64();
        total_s += dt;
        best_s = best_s.min(dt);
        last_norm = l2_norm(&out);
    }

    let avg_s = total_s / repeats as f64;
    println!(
        "{{\"layer\":{},\"experts\":{},\"threads\":{},\"load_s\":{:.4},\"warmup_s\":{:.4},\"avg_s\":{:.4},\"best_s\":{:.4},\"out_l2\":{:.6}}}",
        layer,
        experts.len(),
        threads,
        load_s,
        warm_s,
        avg_s,
        best_s,
        last_norm
    );
    Ok(())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
