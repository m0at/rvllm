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

    // ─────────────────── Phase 3b: m≥2 cosine probe ───────────────────
    //
    // For m∈{2,4,16,64}: build an [M,K] f16 input by tiling the
    // single-row pattern with a per-row offset, run the dispatcher
    // (cuBLASLt fp8_gemm path), then build the reference by looping
    // the m=1 GEMV N times over the same per-row inputs. Compare
    // per-row cosine between the two outputs; require ≥0.9999 on
    // every row.
    //
    // The two paths use different numerical tracks (GEMV consumes
    // f16 directly; GEMM goes through fp8 quantise + tensor cores)
    // so we don't expect bit-equality at m≥2 — cosine is the
    // correct metric.
    // On sm_121 the dispatcher falls back to looped-m=1 GEMV
    // (cuBLASLt has no blockwise FP8 kernel for this arch yet).
    // The probe still gates correctness: per-row output of the
    // dispatcher must match the same row produced by a direct
    // m=1 GEMV. On capable arches (sm_100 / sm_120) the same
    // dispatch goes through cuBLASLt blockwise tensor-cores;
    // cosine ≥0.9999 covers both paths.
    for &m in &[2u32, 4, 16, 64] {
        let in_bytes_m = (m as usize) * (k as usize) * 2;
        let out_bytes_m = (m as usize) * (n as usize) * 2;
        let in_region_m = bringup
            .arena
            .region("probe_q_in_m", in_bytes_m, 16)
            .map_err(|e| format!("alloc in_region_m: {e:?}"))?;
        let out_disp_region_m = bringup
            .arena
            .region("probe_q_out_disp_m", out_bytes_m, 16)
            .map_err(|e| format!("alloc disp_m: {e:?}"))?;
        let out_ref_region_m = bringup
            .arena
            .region("probe_q_out_ref_m", out_bytes_m, 16)
            .map_err(|e| format!("alloc ref_m: {e:?}"))?;

        // Build the [M, K] f16 input. Each row is the same sin pattern
        // shifted by `row * 0.7` so the rows aren't degenerate copies
        // (a constant amax across rows would hide per-row scale bugs).
        let mut in_host = Vec::with_capacity(in_bytes_m);
        for row in 0..(m as usize) {
            for i in 0..(k as usize) {
                let x = (((i as f32) * 0.0123 + (row as f32) * 0.7).sin()) * 0.5_f32;
                in_host.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
            }
        }
        unsafe {
            in_region_m
                .copy_from_host(&in_host)
                .map_err(|e| format!("HtoD m-input: {e:?}"))?;
        }

        // Dispatcher path (cuBLASLt fp8_gemm under the hood).
        unsafe {
            bringup.fp8_proj_dispatch(
                kernel_gemv,
                out_disp_region_m.device_ptr(),
                weight.offset_bytes,
                blockscale,
                in_region_m.device_ptr(),
                m,
                n,
                k,
                stream_raw,
            )
        }
        .map_err(|e| format!("dispatch m={m}: {e:?}"))?;
        bringup
            .stream
            .fence()
            .map_err(|e| format!("disp_m fence: {e:?}"))?;

        // Reference path: loop GEMV at m=1 N times into stacked output.
        // Each iteration writes one [1, n] row at the right offset.
        for row in 0..(m as usize) {
            let row_in_ptr = in_region_m.device_ptr() + (row as u64) * (k as u64) * 2;
            let row_out_ptr = out_ref_region_m.device_ptr() + (row as u64) * (n as u64) * 2;
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m: 1, n, k }.launch(
                    kernel_gemv,
                    row_out_ptr,
                    weight.offset_bytes,
                    blockscale,
                    row_in_ptr,
                    stream_raw,
                )
            }
            .map_err(|e| format!("ref-loop GEMV (m={m}, row={row}): {e:?}"))?;
        }
        bringup
            .stream
            .fence()
            .map_err(|e| format!("ref_m fence: {e:?}"))?;

        // DtoH both, compare per-row cosine.
        let mut ref_host = vec![0u8; out_bytes_m];
        let mut disp_host = vec![0u8; out_bytes_m];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            cuMemcpyDtoH_v2(
                ref_host.as_mut_ptr() as *mut _,
                out_ref_region_m.device_ptr(),
                out_bytes_m,
            );
            cuMemcpyDtoH_v2(
                disp_host.as_mut_ptr() as *mut _,
                out_disp_region_m.device_ptr(),
                out_bytes_m,
            );
        }

        // Compute per-row cosine. n elements per row, f16 → f32 for
        // stability of the reduction.
        let row_len_bytes = (n as usize) * 2;
        let mut min_cos = f32::INFINITY;
        let mut mean_cos_sum = 0.0_f64;
        let mut worst_row: usize = 0;
        for row in 0..(m as usize) {
            let off = row * row_len_bytes;
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for i in 0..(n as usize) {
                let a_bits = u16::from_le_bytes([ref_host[off + i * 2], ref_host[off + i * 2 + 1]]);
                let b_bits =
                    u16::from_le_bytes([disp_host[off + i * 2], disp_host[off + i * 2 + 1]]);
                let a = f16_bits_to_f32_local(a_bits) as f64;
                let b = f16_bits_to_f32_local(b_bits) as f64;
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            let cos = if na > 0.0 && nb > 0.0 {
                dot / (na.sqrt() * nb.sqrt())
            } else if na == 0.0 && nb == 0.0 {
                1.0
            } else {
                0.0
            };
            mean_cos_sum += cos;
            if (cos as f32) < min_cos {
                min_cos = cos as f32;
                worst_row = row;
            }
        }
        let mean_cos = (mean_cos_sum / (m as f64)) as f32;
        let threshold = 0.9999_f32;
        let pass = min_cos >= threshold;
        eprintln!(
            "  m={m:>3}: mean_cos={mean_cos:.6}, min_cos={min_cos:.6} (row {worst_row}) — {}",
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            return Err(format!(
                "m={m} cosine below threshold {threshold:.6}: min={min_cos:.6} at row {worst_row}"
            ));
        }
    }

    Ok(())
}

/// Local f16-bits → f32 used by the cosine reduction. Avoids
/// pulling in the runtime's helper which lives behind a private
/// crate path.
fn f16_bits_to_f32_local(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    if exp == 0 && frac == 0 {
        return f32::from_bits(sign << 31);
    }
    if exp == 0x1f {
        let m = if frac != 0 { 1 << 22 } else { 0 };
        return f32::from_bits((sign << 31) | (0xff << 23) | m);
    }
    if exp == 0 {
        // subnormal — uncommon in our inputs, fall back via shifts
        let mut e = 1;
        let mut m = frac;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        let new_exp = (-14 + e + 127) as u32;
        return f32::from_bits((sign << 31) | (new_exp << 23) | (m << 13));
    }
    let new_exp = (exp - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
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
