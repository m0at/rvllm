//! `probe-qwen-fp8-proj`: validates that
//! `Qwen36Bringup::fp8_proj_dispatch` at m=1 produces byte-identical
//! output to a direct `Fp8GemvF16InLaunch` call against the same
//! weight + input. The dispatcher is the entry point that Phase 4
//! will use to flip the per-token loop to m=N; if its m=1 path
//! diverges from today's direct call site even by a single bit, the
//! flip would land silent garbage on top of working code (see Codex
//! round 16 #1 for the same class of bug). This probe gates the
//! Phase 3a scaffold against that.
//!
//! Methodology:
//!   1. Load Qwen36Bringup.
//!   2. Pick a representative full-attn projection: layer 3's
//!      `q_proj` (n=8192, k=2048 — the largest of the four).
//!   3. Stage a synthetic `[1, k]` f16 input on-device.
//!   4. Run `Fp8GemvF16InLaunch { m: 1, n, k }` → reference output.
//!   5. Run `bringup.fp8_proj_dispatch(... m=1 ...)` → dispatched output.
//!   6. DtoH both, compare byte-for-byte (cosine should be 1.0
//!      exactly — same kernel, same args).
//!
//! Also smokes the m≥2 error path: passing m=2 must return the
//! Phase-3b deferral error, not silently produce zeros.
//!
//! Gated behind `required-features = ["gb10"]`. Run with:
//!
//! ```bash
//! RVLLM_MODEL_DIR=/home/r00t/.vllm/models/qwen3-6-35b-a3b-fp8 \
//! RVLLM_KERNELS_DIR=/home/r00t/workspace/upstream/rvllm-serve/kernels \
//! RVLLM_ARENA_GB=50 \
//! cargo run --release -p rvllm-runtime --bin probe-qwen-fp8-proj --features cuda,gb10
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use rvllm_runtime::gemma4_bring_up::Gemma4EnginePaths;
use rvllm_runtime::qwen36_bring_up::Qwen36Bringup;

fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|_| format!("missing env var: {name}"))
}

fn env_gb(name: &str, default_gb: u64) -> Result<u64, String> {
    std::env::var(name)
        .map(|v| v.parse::<u64>().unwrap_or(default_gb))
        .or_else(|_| Ok::<u64, String>(default_gb))
}

const UNUSED_PATH_MARKER: &str = "<unused-on-sm121>";
const MINIMAL_POLICY_JSON: &str =
    r#"{"revision":"probe-qwen-sm121","arch":"sm_121","variants":[],"entries":{}}"#;

fn resolve_policy_path(kernels_dir: &std::path::Path) -> Result<PathBuf, String> {
    if let Ok(p) = env_path("RVLLM_POLICY") {
        return Ok(p);
    }
    let target = kernels_dir.join(".probe-qwen-fp8-proj-minimal-policy.json");
    if !target.exists() {
        std::fs::write(&target, MINIMAL_POLICY_JSON)
            .map_err(|e| format!("write minimal policy {}: {e}", target.display()))?;
    }
    Ok(target)
}

/// Quick f32 → f16 round-to-nearest-even (the runtime has its own
/// helper but it lives behind a private path; reproducing it here
/// keeps the probe self-contained).
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // NaN or Inf
        let m = if frac != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1f << 10) | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | (0x1f << 10);
    }
    if new_exp <= 0 {
        // subnormals — collapse to 0 for probe input simplicity
        return sign << 15;
    }
    let f10 = (frac >> 13) as u16;
    let round_bit = (frac >> 12) & 0x1;
    let sticky = (frac & 0x0fff) != 0;
    let mut out = (sign << 15) | ((new_exp as u16) << 10) | f10;
    if round_bit != 0 && (sticky || (f10 & 1) != 0) {
        out += 1;
    }
    out
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

    eprintln!("== probe-qwen-fp8-proj ==");
    eprintln!("  model_dir   = {}", paths.model_dir.display());
    eprintln!("  arena       = {arena_gb} GB");

    eprintln!("\n... loading Qwen36Bringup");
    let bringup = Qwen36Bringup::load(paths, arena_bytes)
        .map_err(|e| format!("Qwen36Bringup::load: {e}"))?;
    eprintln!("✓ load done");

    // Pick a real Qwen full-attn projection. Layer 3 in the Qwen 3.6
    // 35B-A3B layer-types order is full-attn (Phase-4 probes already
    // exercise it). Using the same shape keeps this probe tied to
    // production-realistic dimensions.
    let layer_idx = 3usize;
    let weight = match &bringup.model.layers[layer_idx].attn {
        rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => &fl.q_proj,
        _ => return Err(format!("layer {layer_idx} is not Full-attn — adjust probe")),
    };
    let n = weight.shape[0] as u32;
    let k = weight.shape[1] as u32;
    let blockscale = weight.blockscale_ptr.ok_or_else(|| "q_proj has no blockscale".to_string())?;
    eprintln!("  using layer {layer_idx} q_proj: n={n}, k={k}");

    let kernel_gemv = bringup
        .outside_kernels
        .fn_fp8_gemv_wpr_native_f16in
        .ok_or_else(|| "fn_fp8_gemv_wpr_native_f16in not loaded".to_string())?;

    // Allocate input + two output regions on-device.
    let in_bytes = (k as usize) * 2; // [1, k] f16
    let out_bytes = (n as usize) * 2; // [1, n] f16
    let in_region = bringup
        .arena
        .region("probe_q_in", in_bytes, 16)
        .map_err(|e| format!("alloc in_region: {e:?}"))?;
    let out_ref_region = bringup
        .arena
        .region("probe_q_out_ref", out_bytes, 16)
        .map_err(|e| format!("alloc ref: {e:?}"))?;
    let out_disp_region = bringup
        .arena
        .region("probe_q_out_disp", out_bytes, 16)
        .map_err(|e| format!("alloc disp: {e:?}"))?;

    // Synthetic but non-trivial input. A constant-1.0 row is fine
    // for an equivalence test — both code paths see identical bytes.
    // Using a deterministic pattern (sin-based) gives the f16 ALU
    // some work and surfaces any quantize-step divergence later.
    let mut in_host = Vec::with_capacity(k as usize);
    for i in 0..(k as usize) {
        let x = ((i as f32) * 0.0123).sin() * 0.5_f32;
        in_host.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
    }
    unsafe {
        in_region
            .copy_from_host(&in_host)
            .map_err(|e| format!("HtoD input: {e:?}"))?;
    }

    let stream_raw = bringup.stream.raw() as u64;

    // Reference: direct Fp8GemvF16InLaunch.
    unsafe {
        rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m: 1, n, k }.launch(
            kernel_gemv,
            out_ref_region.device_ptr(),
            weight.offset_bytes,
            blockscale,
            in_region.device_ptr(),
            stream_raw,
        )
    }
    .map_err(|e| format!("reference GEMV launch: {e:?}"))?;
    bringup
        .stream
        .fence()
        .map_err(|e| format!("ref fence: {e:?}"))?;

    // Dispatched: Qwen36Bringup::fp8_proj_dispatch at m=1.
    unsafe {
        bringup.fp8_proj_dispatch(
            kernel_gemv,
            out_disp_region.device_ptr(),
            weight.offset_bytes,
            blockscale,
            in_region.device_ptr(),
            /*m*/ 1,
            n,
            k,
            stream_raw,
        )
    }
    .map_err(|e| format!("dispatch m=1: {e:?}"))?;
    bringup
        .stream
        .fence()
        .map_err(|e| format!("disp fence: {e:?}"))?;

    // DtoH + compare.
    let mut ref_host = vec![0u8; out_bytes];
    let mut disp_host = vec![0u8; out_bytes];
    #[cfg(feature = "cuda")]
    unsafe {
        use cudarc::driver::sys::*;
        let r1 = cuMemcpyDtoH_v2(
            ref_host.as_mut_ptr() as *mut _,
            out_ref_region.device_ptr(),
            out_bytes,
        );
        let r2 = cuMemcpyDtoH_v2(
            disp_host.as_mut_ptr() as *mut _,
            out_disp_region.device_ptr(),
            out_bytes,
        );
        if r1 != CUresult::CUDA_SUCCESS || r2 != CUresult::CUDA_SUCCESS {
            return Err(format!(
                "DtoH copies failed: ref={r1:?}, disp={r2:?}"
            ));
        }
    }

    let mut diffs = 0usize;
    let mut first_diff: Option<usize> = None;
    for i in 0..out_bytes {
        if ref_host[i] != disp_host[i] {
            diffs += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }
    if diffs == 0 {
        eprintln!(
            "\n✓ m=1: dispatcher produces BYTE-IDENTICAL output to direct \
             Fp8GemvF16InLaunch (n={n}, k={k}, {out_bytes} bytes compared)"
        );
    } else {
        return Err(format!(
            "m=1 dispatch differs from reference: {diffs}/{out_bytes} \
             bytes differ; first diff at byte {} (ref=0x{:02x}, disp=0x{:02x})",
            first_diff.unwrap(),
            ref_host[first_diff.unwrap()],
            disp_host[first_diff.unwrap()],
        ));
    }

    // m=2 must return the Phase-3b deferral error.
    let res = unsafe {
        bringup.fp8_proj_dispatch(
            kernel_gemv,
            out_disp_region.device_ptr(),
            weight.offset_bytes,
            blockscale,
            in_region.device_ptr(),
            /*m*/ 2,
            n,
            k,
            stream_raw,
        )
    };
    match res {
        Err(e) => {
            let msg = format!("{e:?}");
            if msg.contains("Phase 3b") {
                eprintln!("✓ m=2: dispatcher correctly defers with Phase-3b sentinel error");
            } else {
                return Err(format!(
                    "m=2 returned an error but not the expected Phase-3b sentinel: {msg}"
                ));
            }
        }
        Ok(()) => {
            return Err(
                "m=2 unexpectedly succeeded — Phase 3b path is supposed to return Err today"
                    .into(),
            );
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            eprintln!("\n✓ probe-qwen-fp8-proj passed");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\n✗ probe-qwen-fp8-proj failed: {e}");
            ExitCode::FAILURE
        }
    }
}
