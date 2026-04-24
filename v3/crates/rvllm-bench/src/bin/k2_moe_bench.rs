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

fn main() {
    if let Err(e) = run() {
        eprintln!("k2-moe-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let layer = env_usize("K2_LAYER", 1);
    let repeats = env_usize("K2_REPEATS", 3);
    let threads = env_usize("K2_THREADS", 16);

    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();

    let load_t0 = Instant::now();
    let store = K2CpuExpertStore::open(&model_dir).map_err(|e| format!("open: {e}"))?;
    let load_s = load_t0.elapsed().as_secs_f64();

    let mut x = vec![0.0f32; store.arch.hidden_size];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 257) as f32 - 128.0) / 64.0;
    }

    eprintln!(
        "k2-moe-bench: model={} layer={} repeats={} threads={} hidden={} moe={} topk={} first_k_dense={}",
        model_dir.display(),
        layer,
        repeats,
        threads,
        store.arch.hidden_size,
        store.arch.moe_intermediate_size,
        store.arch.n_experts_per_tok,
        store.arch.first_k_dense,
    );
    eprintln!("load_s={load_s:.3}");

    let warm_t0 = Instant::now();
    let warm = store
        .run_moe_block(layer, &x)
        .map_err(|e| format!("warmup: {e}"))?;
    let warm_s = warm_t0.elapsed().as_secs_f64();
    eprintln!("warmup_s={warm_s:.3} output_norm={:.6}", l2_norm(&warm));

    let mut total_s = 0.0f64;
    let mut best_s = f64::INFINITY;
    let mut last_norm = 0.0f32;
    for _ in 0..repeats {
        let t0 = Instant::now();
        let out = store
            .run_moe_block(layer, &x)
            .map_err(|e| format!("run: {e}"))?;
        let dt = t0.elapsed().as_secs_f64();
        total_s += dt;
        best_s = best_s.min(dt);
        last_norm = l2_norm(&out);
    }
    let avg_s = total_s / repeats as f64;
    println!(
        "{{\"layer\":{},\"threads\":{},\"load_s\":{:.4},\"warmup_s\":{:.4},\"avg_s\":{:.4},\"best_s\":{:.4},\"out_l2\":{:.6}}}",
        layer, threads, load_s, warm_s, avg_s, best_s, last_norm
    );
    Ok(())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
