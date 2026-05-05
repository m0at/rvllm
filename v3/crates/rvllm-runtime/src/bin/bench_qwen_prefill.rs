//! `bench-qwen-prefill`: time the per-token Qwen 3.6 prefill loop at
//! several prompt-length buckets, log the results in a stable format
//! that future Phase-3+ work in `v3/QWEN_BATCHED_PREFILL_PLAN.md` can
//! diff against.
//!
//! Loads a real Qwen 3.6 fp8 checkpoint and runs
//! `Qwen36Bringup::forward_qwen36_decode` with a synthetic
//! `[0, 1, 2, …, N-1]` prompt. The synthetic prompt is fine for TTFT
//! measurement: the per-token loop's launch overhead, fence count and
//! kernel-work scaling are content-independent — what matters is the
//! shape `N`, not which tokens land in `prompt_ids`.
//!
//! Each bucket runs `--warmup` (default 1) untimed iterations followed
//! by `--iters` (default 3) timed iterations; we report the median of
//! the timed batch so a single GC / driver hiccup doesn't dominate.
//!
//! Gated behind `required-features = ["gb10"]` like
//! `probe-gemma4-load`. Run with:
//!
//! ```bash
//! RVLLM_MODEL_DIR=/home/r00t/.vllm/models/qwen3-6-35b-a3b-fp8 \
//! RVLLM_KERNELS_DIR=/home/r00t/workspace/upstream/rvllm-serve/kernels \
//! RVLLM_ARENA_GB=50 \
//! cargo run --release -p rvllm-runtime --bin bench-qwen-prefill --features gb10
//! ```
//!
//! Optional flags via env (kept simple — no clap dep):
//! * `BENCH_LENS=32,256,1024` (default `32,256,1024,4096`)
//! * `BENCH_WARMUP=2` (default 1)
//! * `BENCH_ITERS=5` (default 3)
//! * `BENCH_RESET_PER_RUN=1` (default 1) — reset linear/KV/conv state
//!   between runs to keep them independent.
//!
//! Output: one line of human-readable table + one line of JSON for
//! script-parseable comparison across commits.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use rvllm_runtime::gemma4_bring_up::Gemma4EnginePaths;
use rvllm_runtime::qwen36_bring_up::Qwen36Bringup;

fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|_| format!("missing env var: {name}"))
}

fn env_gb(name: &str, default_gb: u64) -> Result<u64, String> {
    match std::env::var(name) {
        Ok(v) => v
            .parse::<u64>()
            .map_err(|e| format!("{name}={v:?}: {e}")),
        Err(_) => Ok(default_gb),
    }
}

fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(default)
}

const UNUSED_PATH_MARKER: &str = "<unused-on-sm121>";
const MINIMAL_POLICY_JSON: &str =
    r#"{"revision":"bench-qwen-sm121","arch":"sm_121","variants":[],"entries":{}}"#;

fn resolve_policy_path(kernels_dir: &std::path::Path) -> Result<PathBuf, String> {
    if let Ok(p) = env_path("RVLLM_POLICY") {
        return Ok(p);
    }
    let target = kernels_dir.join(".bench-qwen-minimal-policy.json");
    if !target.exists() {
        std::fs::write(&target, MINIMAL_POLICY_JSON)
            .map_err(|e| format!("write minimal policy {}: {e}", target.display()))?;
    }
    Ok(target)
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[mid]
    } else {
        // Average of the two middle samples — `Duration` arithmetic
        // saturates which is fine for the values we deal with.
        (xs[mid - 1] + xs[mid]) / 2
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let kernels_dir = env_path("RVLLM_KERNELS_DIR")?;
    let cutlass_so = env_path("RVLLM_CUTLASS_SO")
        .unwrap_or_else(|_| PathBuf::from(UNUSED_PATH_MARKER));
    let fa3_so = env_path("RVLLM_FA3_SO")
        .unwrap_or_else(|_| PathBuf::from(UNUSED_PATH_MARKER));
    let policy_json = resolve_policy_path(&kernels_dir)?;
    let paths = Gemma4EnginePaths {
        model_dir,
        kernels_dir,
        cutlass_so,
        fa3_so,
        policy_json,
    };
    let arena_gb = env_gb("RVLLM_ARENA_GB", 50)?;
    let arena_bytes = (arena_gb as usize) * 1024 * 1024 * 1024;

    let lens = env_usize_list("BENCH_LENS", &[32, 256, 1024, 4096]);
    let warmup = env_usize("BENCH_WARMUP", 1);
    let iters = env_usize("BENCH_ITERS", 3).max(1);
    let reset_per_run = env_bool("BENCH_RESET_PER_RUN", true);

    eprintln!("== bench-qwen-prefill ==");
    eprintln!("  model_dir   = {}", paths.model_dir.display());
    eprintln!("  kernels_dir = {}", paths.kernels_dir.display());
    eprintln!("  arena       = {arena_gb} GB");
    eprintln!("  lens        = {lens:?}");
    eprintln!("  warmup      = {warmup}, iters = {iters}");
    eprintln!("  reset_state = {reset_per_run}");

    eprintln!("\n... loading Qwen36Bringup");
    let t0 = Instant::now();
    let bringup = Qwen36Bringup::load(paths, arena_bytes)
        .map_err(|e| format!("Qwen36Bringup::load: {e}"))?;
    eprintln!(
        "✓ Qwen36Bringup::load done in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let max_n = lens.iter().copied().max().unwrap_or(0);
    if max_n == 0 {
        return Err("BENCH_LENS produced no valid lengths".into());
    }
    let prompt_full: Vec<i32> = (0..max_n as i32).collect();

    println!();
    println!(
        "{:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "N", "median_ms", "min_ms", "max_ms", "tok_per_s"
    );

    // Stable JSON-line schema for diff-friendly bench tracking.
    // One line per bucket; tools can `jq -s 'add'` across runs.
    let mut json_lines: Vec<String> = Vec::with_capacity(lens.len());

    for &n in &lens {
        if n == 0 || n > max_n {
            continue;
        }
        let prompt = &prompt_full[..n];

        // Warmup. Errors here propagate — a broken bring-up shouldn't
        // be silently masked just because it's "warmup".
        for _ in 0..warmup {
            if reset_per_run {
                let _ = bringup.reset_linear_state();
                let _ = bringup.reset_kv_cache();
                let _ = bringup.reset_conv_state();
            }
            bringup
                .forward_qwen36_decode(prompt, 0, &[])
                .map_err(|e| format!("warmup forward (N={n}): {e:?}"))?;
        }

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            if reset_per_run {
                let _ = bringup.reset_linear_state();
                let _ = bringup.reset_kv_cache();
                let _ = bringup.reset_conv_state();
            }
            let t = Instant::now();
            bringup
                .forward_qwen36_decode(prompt, 0, &[])
                .map_err(|e| format!("timed forward (N={n}): {e:?}"))?;
            samples.push(t.elapsed());
        }
        let med = median(samples.clone());
        let mn = *samples.iter().min().unwrap();
        let mx = *samples.iter().max().unwrap();
        let tok_per_s = (n as f64) / med.as_secs_f64();
        println!(
            "{:>6}  {:>10.2}  {:>12.2}  {:>12.2}  {:>10.1}",
            n,
            med.as_secs_f64() * 1e3,
            mn.as_secs_f64() * 1e3,
            mx.as_secs_f64() * 1e3,
            tok_per_s,
        );
        json_lines.push(format!(
            "{{\"n\":{n},\"median_ms\":{:.4},\"min_ms\":{:.4},\"max_ms\":{:.4},\"tok_per_s\":{:.2}}}",
            med.as_secs_f64() * 1e3,
            mn.as_secs_f64() * 1e3,
            mx.as_secs_f64() * 1e3,
            tok_per_s,
        ));
    }

    println!("\nJSONL:");
    for l in &json_lines {
        println!("{l}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            eprintln!("\n✓ bench-qwen-prefill finished");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\n✗ bench-qwen-prefill failed: {e}");
            ExitCode::FAILURE
        }
    }
}
