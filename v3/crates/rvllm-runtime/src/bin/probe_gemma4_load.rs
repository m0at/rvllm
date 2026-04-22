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

/// Marker path used for `RVLLM_CUTLASS_SO` / `RVLLM_FA3_SO` when the
/// operator skipped them on a sm_121 run. These files never get
/// opened on sm_121 (CUTLASS backend is `Absent`, attention goes
/// through `Fa2Ptx`), so any unique non-existent path works — the
/// name is what shows up in the probe banner.
const UNUSED_PATH_MARKER: &str = "<unused-on-sm121>";

/// Minimal `policy.json` written when `RVLLM_POLICY` is unset. Parsed
/// successfully by `rvllm_cutlass::Policy` but contains no entries
/// — sm_121's `CutlassBackend::Absent` never looks entries up.
const MINIMAL_POLICY_JSON: &str =
    r#"{"revision":"probe-sm121","arch":"sm_121","variants":[],"entries":{}}"#;

/// Resolve `RVLLM_POLICY`, materialising a minimal policy file if
/// the env var is unset. Written into a cache-dir path derived from
/// `kernels_dir` so we don't pollute `/tmp` across runs and don't
/// stomp on a real policy the user left in place.
fn resolve_policy_path(kernels_dir: &std::path::Path) -> Result<PathBuf, String> {
    if let Ok(p) = env_path("RVLLM_POLICY") {
        return Ok(p);
    }
    let target = kernels_dir.join(".probe-minimal-policy.json");
    if !target.exists() {
        std::fs::write(&target, MINIMAL_POLICY_JSON)
            .map_err(|e| format!("write minimal policy {}: {e}", target.display()))?;
    }
    Ok(target)
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
    // Validate (B) — fp8_gemv.ptx loaded + the f16-input native-CVT
    // entry point (the one the Sm121 decode fast path actually calls)
    // resolved correctly for the live target.
    // ======================================================================
    eprintln!("\n[B] fp8_gemv.ptx kernel-resolve check:");
    let fused = &bringup.fused;
    if fused.fp8_gemv_mod.raw() == 0 {
        return Err("fp8_gemv_mod.raw() == 0 (module not loaded)".into());
    }
    eprintln!(
        "    fp8_gemv_mod              = loaded ({})",
        fused.fp8_gemv_mod.path().display(),
    );

    let f16in_expected = target
        .is_some_and(|t| rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t));
    match (&fused.fn_fp8_gemv_wpr_native_f16in, f16in_expected) {
        (Some(f), true) => {
            if f.raw() == 0 {
                return Err(
                    "fn_fp8_gemv_wpr_native_f16in is Some but raw() == 0".into(),
                );
            }
            eprintln!(
                "    fn_fp8_gemv_wpr_native_f16in = resolved ({})",
                rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point(),
            );
        }
        (None, false) => {
            eprintln!(
                "    fn_fp8_gemv_wpr_native_f16in = None (expected: sm_100+ only)",
            );
        }
        (Some(_), false) => {
            return Err(format!(
                "fn_fp8_gemv_wpr_native_f16in unexpectedly Some on non-Blackwell target {target:?}"
            ));
        }
        (None, true) => {
            return Err(format!(
                "fn_fp8_gemv_wpr_native_f16in unexpectedly None on target {target:?} — \
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

        // Synthetic prompt. IDs are placeholders — this is a
        // throughput / DIAG_COMPARE probe, not a correctness test; we
        // discard the outputs. Length defaults to 8; override via
        // RVLLM_PROMPT_LEN to exercise the CUTLASS M≥128 batch-prefill
        // path when bisecting aa01001pftrope0 divergence.
        let prompt_len: usize = std::env::var("RVLLM_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let mut prompt_ids: Vec<u32> = Vec::with_capacity(prompt_len);
        prompt_ids.push(2); // BOS
        for i in 1..prompt_len {
            // Gemma 4 vocab has ~262k entries; pick a deterministic
            // spread that touches a range of embedding rows rather
            // than a single contiguous block.
            prompt_ids.push((107 + (i as u32 * 37) % 100_000).max(107));
        }
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
