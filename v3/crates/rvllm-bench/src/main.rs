//! rvllm-bench: loads a model + kernels + cutlass .so + fa3 .so, runs
//! `iters` decode-step forwards on a fixed-batch bucket, and reports
//! tokens/sec.
//!
//! Env vars (all required for cuda path):
//!   RVLLM_MODEL_DIR   = HF snapshot dir with config.json + safetensors
//!   RVLLM_KERNELS_DIR = dir with manifest.json + compiled PTX
//!   RVLLM_CUTLASS_SO  = path to libcutlass_kernels.so
//!   RVLLM_FA3_SO      = path to libfa3_kernels.so
//!   RVLLM_POLICY      = path to policy.json
//!   RVLLM_BATCH       = batch size (default 128)
//!   RVLLM_ITERS       = decode-step iterations (default 100)
//!   RVLLM_WARMUP      = warmup iterations (default 10)
//!
//! Prints one JSON line per run: {batch, iters, tok_per_sec, ms_per_step}.

use std::path::PathBuf;
use std::time::Instant;

use rvllm_core::{ModelArch as HfModelArch, ModelConfig};
#[cfg(feature = "cuda")]
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use rvllm_runtime::{Bringup, EnginePaths};

fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn is_gemma4_model_dir(model_dir: &std::path::Path) -> Result<bool, String> {
    Ok(matches!(
        ModelConfig::load_hf(model_dir)
            .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?
            .architecture,
        HfModelArch::Gemma4
    ))
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let result = run();
    if let Err(e) = result {
        eprintln!("rvllm-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let paths = EnginePaths {
        model_dir: env_path("RVLLM_MODEL_DIR")?,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
    };
    let batch = env_u32("RVLLM_BATCH", 128);
    let iters = env_u32("RVLLM_ITERS", 100);
    let warmup = env_u32("RVLLM_WARMUP", 10);

    // Arena budget: model (~16 GB fp8) + kv (~8 GB) + scratch/workspace (~4 GB).
    // Override with RVLLM_ARENA_GB if GPU memory is constrained.
    let arena_gb: usize = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let arena_bytes: usize = arena_gb * 1024 * 1024 * 1024;

    eprintln!("== rvllm-bench v3 ==");
    eprintln!("model_dir   = {}", paths.model_dir.display());
    eprintln!("kernels_dir = {}", paths.kernels_dir.display());
    eprintln!("batch       = {batch}");
    eprintln!("iters       = {iters} (warmup {warmup})");

    let is_gemma4 = is_gemma4_model_dir(&paths.model_dir)?;

    if is_gemma4 {
        #[cfg(not(feature = "cuda"))]
        {
            return Err("Gemma4 bench requires rvllm-bench built with --features cuda".into());
        }
        #[cfg(feature = "cuda")]
        {
            eprintln!("== Gemma 4 detected, using Gemma4Bringup ==");
            let g4_paths = Gemma4EnginePaths {
                model_dir: paths.model_dir,
                kernels_dir: paths.kernels_dir,
                cutlass_so: paths.cutlass_so,
                fa3_so: paths.fa3_so,
                policy_json: paths.policy_json,
            };
            let t0 = Instant::now();
            let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
                .map_err(|e| format!("gemma4 bringup: {e}"))?;
            eprintln!(
                "bringup: {:.2}s | arch layers={} hidden={} heads={} sliding_kv={} global_kv={}",
                t0.elapsed().as_secs_f64(),
                g4.arch.num_hidden_layers,
                g4.arch.hidden_size,
                g4.arch.num_attention_heads,
                g4.arch.num_kv_heads_sliding,
                g4.arch.num_kv_heads_global,
            );
            eprintln!("arena used = {} MiB", g4.arena.used() / (1024 * 1024));
            let result = unsafe { g4.run_bench(batch, iters, warmup) };
            print_result(result);
            return Ok(());
        }
    }

    let t0 = Instant::now();
    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!(
        "bringup: {:.2}s | arch layers={} hidden={} heads={} kv_heads={}",
        t0.elapsed().as_secs_f64(),
        br.arch.num_hidden_layers,
        br.arch.hidden_size,
        br.arch.num_attention_heads,
        br.arch.num_key_value_heads,
    );
    eprintln!("arena used = {} MiB", br.arena.used() / (1024 * 1024));

    if std::env::var("RVLLM_SWEEP").ok().as_deref() == Some("1") {
        return run_sweep(&br, batch, iters, warmup);
    }

    let result =
        unsafe { br.run_bench(batch, iters, warmup) }.map_err(|e| format!("run_bench: {e}"))?;
    print_result(result);
    Ok(())
}

fn print_result(r: rvllm_runtime::bring_up::BenchResult) {
    let tok_per_sec = if r.total_ns > 0 {
        (r.iters as f64 * r.num_seqs as f64) * 1.0e9 / r.total_ns as f64
    } else {
        0.0
    };
    let ms_per_step = r.ns_per_step as f64 / 1.0e6;
    let ttft_str = match (r.ttft_ns, r.ttft_hot_ns) {
        (Some(cold), Some(hot)) => format!(
            " ttft_cold={:.2}ms ttft_hot={:.2}ms",
            cold as f64 / 1.0e6,
            hot as f64 / 1.0e6
        ),
        (Some(cold), None) => format!(" ttft={:.2}ms", cold as f64 / 1.0e6),
        _ => String::new(),
    };
    eprintln!(
        "bench: batch={} iters={} -> {:.0} tok/s ({:.3} ms/step){}",
        r.num_seqs, r.iters, tok_per_sec, ms_per_step, ttft_str
    );
    let ttft_json = match (r.ttft_ns, r.ttft_hot_ns) {
        (Some(cold), Some(hot)) => format!(
            ",\"ttft_cold_ms\":{:.3},\"ttft_hot_ms\":{:.3}",
            cold as f64 / 1.0e6,
            hot as f64 / 1.0e6
        ),
        (Some(cold), None) => format!(",\"ttft_ms\":{:.3}", cold as f64 / 1.0e6),
        _ => String::new(),
    };
    println!(
        "{{\"batch\":{},\"iters\":{},\"tok_per_sec\":{:.1},\"ms_per_step\":{:.4}{}}}",
        r.num_seqs, r.iters, tok_per_sec, ms_per_step, ttft_json
    );
}

fn run_sweep(br: &Bringup, batch: u32, iters: u32, warmup: u32) -> Result<(), String> {
    // Variant grid. Policy knows 40 non-residual + 10 residual (per the
    // autotune .so). Sample a promising subset.
    let nonres: &[u32] = &[0, 2, 5, 8, 10, 12, 14];
    let residuals: &[u32] = &[100, 102, 105, 108];

    let mut best = (u128::MAX, 0u32, 0u32);
    eprintln!("== sweep @ N={batch} ==");
    for &nr in nonres {
        for &r in residuals {
            let ck = br.arena.checkpoint();
            let res =
                unsafe { br.run_bench_with_variants(batch, iters, warmup, Some(nr), Some(r)) };
            unsafe { br.arena.restore(ck) };
            match res {
                Ok(r_) => {
                    let tok_per_sec = if r_.total_ns > 0 {
                        (r_.iters as f64 * r_.num_seqs as f64) * 1.0e9 / r_.total_ns as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "nonres={nr} res={r} -> {:.0} tok/s ({:.3} ms/step)",
                        tok_per_sec,
                        r_.ns_per_step as f64 / 1.0e6
                    );
                    println!(
                        "{{\"nonres\":{nr},\"res\":{r},\"tok_per_sec\":{:.1}}}",
                        tok_per_sec
                    );
                    if r_.ns_per_step < best.0 {
                        best = (r_.ns_per_step, nr, r);
                    }
                }
                Err(e) => {
                    eprintln!("nonres={nr} res={r} -> ERROR: {e}");
                }
            }
        }
    }
    eprintln!(
        "BEST: nonres={} res={} ({:.3} ms/step)",
        best.1,
        best.2,
        best.0 as f64 / 1.0e6
    );
    Ok(())
}
