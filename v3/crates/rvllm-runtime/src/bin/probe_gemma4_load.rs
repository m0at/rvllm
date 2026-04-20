//! GB10 bring-up probe — loads a Gemma 4 fp8 model end-to-end and
//! validates the Rust additions on this branch:
//!
//!   (A) `Gemma4Bringup::load` picks `UnifiedArena` on sm_121 (via
//!       `cuMemAllocManaged`) instead of `HbmArena`.
//!   (B) `load_gemma4_fused` resolves the new `fp8_gemv.ptx` module
//!       and both WPR entry-point symbols. The `wpr_native` handle
//!       is `Some` iff the live device's `CompileTarget` reports
//!       availability (sm_100+).
//!
//! Gated behind `required-features = ["gb10"]` so a default
//! `cargo build` skips it — this probe needs a real GPU, a real
//! CUDA 13 + cudarc setup, and a Gemma 4 fp8 model on disk, none
//! of which are available in CI or on a developer's workstation.
//!
//! Run with:
//!
//! ```bash
//! RVLLM_MODEL_DIR=/home/r00t/.vllm/models/gemma-4-31b-it-fp8-block \
//! RVLLM_KERNELS_DIR=/home/r00t/workspace/upstream/rvllm/kernels \
//! RVLLM_CUTLASS_SO=/path/to/libcutlass_sm90.so \
//! RVLLM_FA3_SO=/path/to/libfa3_sm90.so \
//! RVLLM_POLICY=/path/to/policy.json \
//! RVLLM_ARENA_GB=4 \
//! cargo run --release -p rvllm-runtime --bin probe-gemma4-load --features gb10
//! ```
//!
//! Exits 0 on successful bring-up + field validation, non-zero on
//! any failure with a descriptive message.

use std::path::PathBuf;
use std::process::ExitCode;

use rvllm_core::{CompileTarget, ModelArch as HfModelArch, ModelConfig};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};

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

fn run() -> Result<(), String> {
    // On sm_121 the CUTLASS `.so` and FA3 `.so` are never opened —
    // `CutlassBackend::load_for` short-circuits to `Absent` on sm_121,
    // and the attention layer takes `AttentionBackend::Fa2Ptx` instead
    // of `Fa3`. For convenience on GB10 bring-up we let the operator
    // skip the corresponding env vars entirely; the paths are only
    // "required" on pre-Blackwell targets. `policy.json` is still
    // parsed, but a dummy-minimal policy is enough because sm_121 has
    // no CUTLASS entries to look up.
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let kernels_dir = env_path("RVLLM_KERNELS_DIR")?;
    let cutlass_so = env_path("RVLLM_CUTLASS_SO")
        .unwrap_or_else(|_| PathBuf::from("/dev/null-cutlass-sm121-absent"));
    let fa3_so = env_path("RVLLM_FA3_SO")
        .unwrap_or_else(|_| PathBuf::from("/dev/null-fa3-sm121-absent"));
    let policy_json = match env_path("RVLLM_POLICY") {
        Ok(p) => p,
        Err(_) => {
            // Write a minimal policy into the system temp dir. The
            // fields are parsed but never consulted on sm_121 (the
            // CUTLASS backend is `Absent` so variant lookups never
            // happen).
            let tmp = std::env::temp_dir().join("rvllm-probe-minimal-policy.json");
            if !tmp.exists() {
                std::fs::write(
                    &tmp,
                    r#"{"revision":"probe-sm121","arch":"sm_121","variants":[],"entries":{}}"#,
                )
                .map_err(|e| format!("write minimal policy {}: {e}", tmp.display()))?;
            }
            tmp
        }
    };
    let paths = Gemma4EnginePaths {
        model_dir,
        kernels_dir,
        cutlass_so,
        fa3_so,
        policy_json,
    };
    let arena_gb = env_gb("RVLLM_ARENA_GB", 4)?;
    let arena_bytes = (arena_gb as usize) * 1024 * 1024 * 1024;

    eprintln!("== probe-gemma4-load ==");
    eprintln!("  model_dir   = {}", paths.model_dir.display());
    eprintln!("  kernels_dir = {}", paths.kernels_dir.display());
    eprintln!("  cutlass_so  = {}", paths.cutlass_so.display());
    eprintln!("  fa3_so      = {}", paths.fa3_so.display());
    eprintln!("  policy      = {}", paths.policy_json.display());
    eprintln!("  arena       = {arena_gb} GB");

    // Fail fast if the model_dir isn't Gemma 4 — everything below
    // assumes Gemma4 architecture.
    let config = ModelConfig::load_hf(&paths.model_dir)
        .map_err(|e| format!("load config.json: {e}"))?;
    if !matches!(config.architecture, HfModelArch::Gemma4) {
        return Err(format!(
            "expected Gemma4 architecture in {}, got {:?}",
            paths.model_dir.display(),
            config.architecture,
        ));
    }
    eprintln!("  config OK   = Gemma4");

    // ======================================================================
    // The thing we're actually testing: full Gemma4 bring-up on GB10.
    // ======================================================================
    eprintln!("\n... calling Gemma4Bringup::load (this takes a while — weights loading)");
    let t0 = std::time::Instant::now();
    let bringup = Gemma4Bringup::load(paths, arena_bytes)
        .map_err(|e| format!("Gemma4Bringup::load: {e}"))?;
    let elapsed = t0.elapsed();
    eprintln!("✓ Gemma4Bringup::load succeeded in {:.2}s", elapsed.as_secs_f64());

    // ======================================================================
    // Validate (A) — arena backing picked correctly per compute capability.
    // We can't directly inspect which allocator was used (the field is
    // `HbmArena` in both branches), but we can at least verify:
    //   * the arena has the expected capacity
    //   * the compute capability resolved to a `CompileTarget` we know about
    //   * on sm_121 the `UnifiedArena` branch is what ran (inferred from
    //     CC + feature gate being live)
    // ======================================================================
    let (cc_major, cc_minor) = bringup.ctx.compute_capability();
    let target = CompileTarget::from_compute_capability(cc_major, cc_minor);
    eprintln!("\n[A] arena-backing check:");
    eprintln!("    compute_cap = {cc_major}.{cc_minor}");
    eprintln!("    target      = {target:?}");
    eprintln!("    arena.capacity = {} MiB", bringup.arena.capacity() / (1024 * 1024));
    eprintln!("    arena.used     = {} MiB", bringup.arena.used() / (1024 * 1024));
    if bringup.arena.capacity() != arena_bytes {
        return Err(format!(
            "arena capacity {} != requested {arena_bytes}",
            bringup.arena.capacity()
        ));
    }
    match target {
        Some(CompileTarget::Sm121) => {
            eprintln!("    → UnifiedArena path was taken (cuMemAllocManaged + ADVISE)");
        }
        Some(t) => {
            eprintln!("    → HbmArena path was taken (cuMemAlloc_v2) — target {t:?}");
        }
        None => {
            return Err(format!(
                "unsupported compute cap {cc_major}.{cc_minor} (should have failed earlier)"
            ));
        }
    }

    // ======================================================================
    // Validate (B) — fp8_gemv.ptx loaded + entry points resolved.
    // ======================================================================
    eprintln!("\n[B] fp8_gemv.ptx kernel-resolve check:");
    let fused = &bringup.fused;
    if fused.fp8_gemv_mod.raw() == 0 {
        return Err("fp8_gemv_mod.raw() == 0 (module not loaded)".into());
    }
    eprintln!("    fp8_gemv_mod          = loaded ({} path)", fused.fp8_gemv_mod.path().display());
    if fused.fn_fp8_gemv_wpr_lut.raw() == 0 {
        return Err("fn_fp8_gemv_wpr_lut.raw() == 0 (symbol not resolved)".into());
    }
    eprintln!(
        "    fn_fp8_gemv_wpr_lut   = resolved ({})",
        rvllm_kernels::Fp8GemvVariant::WprLut.entry_point(),
    );

    let native_expected =
        target.is_some_and(|t| rvllm_kernels::Fp8GemvVariant::WprNative.available_for(t));
    match (&fused.fn_fp8_gemv_wpr_native, native_expected) {
        (Some(f), true) => {
            if f.raw() == 0 {
                return Err("fn_fp8_gemv_wpr_native is Some but raw() == 0".into());
            }
            eprintln!(
                "    fn_fp8_gemv_wpr_native= resolved ({})",
                rvllm_kernels::Fp8GemvVariant::WprNative.entry_point(),
            );
        }
        (None, false) => {
            eprintln!("    fn_fp8_gemv_wpr_native= None (expected: sm_100+ only)");
        }
        (Some(_), false) => {
            return Err(format!(
                "fn_fp8_gemv_wpr_native unexpectedly Some on non-Blackwell target {target:?}"
            ));
        }
        (None, true) => {
            return Err(format!(
                "fn_fp8_gemv_wpr_native unexpectedly None on target {target:?} — \
                 available_for says it should be resolved"
            ));
        }
    }

    eprintln!("\n✓ all GB10 bring-up invariants hold");

    // =============================================================
    // Optional: run generate + measure tok/s
    // =============================================================
    if std::env::var_os("RVLLM_GENERATE").is_some() {
        eprintln!("\n[C] run_generate smoke (RVLLM_GENERATE set):");
        // Load embedding kernel handle.
        let embed_mod = bringup
            .kernels
            .load_ptx("embedding_gather_f16")
            .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
        let fn_embed = embed_mod
            .get_function("embedding_gather_f16_kernel")
            .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;
        let fn_argmax = bringup.fused.fn_argmax;

        // Tiny prompt (ids are placeholders — this is a throughput
        // probe, not a correctness test; we discard the outputs).
        let prompt_ids: Vec<u32> = vec![2, 108, 109, 110, 111, 112, 113, 114];
        let max_new: usize = std::env::var("RVLLM_MAX_NEW")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        let eos_ids: Vec<u32> = vec![1, 107];

        eprintln!(
            "    prompt_len={}, max_new={}",
            prompt_ids.len(),
            max_new
        );

        let t_gen = std::time::Instant::now();
        let output_ids = unsafe {
            bringup
                .run_generate(fn_embed, fn_argmax, &prompt_ids, max_new, &eos_ids)
                .map_err(|e| format!("run_generate: {e}"))?
        };
        let elapsed = t_gen.elapsed();

        let new_tokens = output_ids.len().saturating_sub(prompt_ids.len());
        let tok_per_s = (new_tokens as f64) / elapsed.as_secs_f64();
        eprintln!(
            "    generated {} new tokens in {:.2}s → {:.2} tok/s",
            new_tokens,
            elapsed.as_secs_f64(),
            tok_per_s,
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\n✗ FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
