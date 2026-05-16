//! Per-tensor FP8 E4M3 quantization with hard-fail clamp gate.
//!
//! Spec 06: "if clamp% > 0.001%, refuse to load — model is mis-scaled,
//! not a viable FP8 candidate."
//!
//! The Rust-side logic here is the reference (CPU) and the gate. The
//! GPU-side fused quantize kernel (faster, no DtoH round-trip) lives
//! in `rvllm-fused` and feeds its per-tensor clamp count back here.

use rayon::prelude::*;
use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};

pub const FP8_E4M3_MAX: f32 = 448.0;
/// Hard fail threshold: 1 clamp per 100,000 values = 10 ppm.
pub const CLAMP_PPM_LIMIT: f32 = 10.0;

/// Result of a per-tensor FP8 quantize pass.
#[derive(Debug, Clone)]
pub struct QuantResult {
    pub scale: f32,
    pub clamp_ppm: f32,
}

pub fn quantize_per_tensor_ref(values: &[f32]) -> QuantResult {
    let amax = values
        .par_iter()
        .copied()
        .map(f32::abs)
        .reduce(|| 0.0f32, f32::max)
        .max(1e-12);
    let scale = amax / FP8_E4M3_MAX;
    let inv = 1.0 / scale;
    let clamps: u64 = values
        .par_iter()
        .filter(|v| {
            let q = **v * inv;
            !(-FP8_E4M3_MAX..=FP8_E4M3_MAX).contains(&q)
        })
        .count() as u64;
    let ppm = if values.is_empty() {
        0.0
    } else {
        (clamps as f64 / values.len() as f64 * 1e6) as f32
    };
    QuantResult {
        scale,
        clamp_ppm: ppm,
    }
}

/// The gate: called after every per-tensor quantize to reject
/// mis-scaled weights.
pub fn check_clamp_gate(tensor_name: &str, ppm: f32, path: &std::path::Path) -> Result<()> {
    if ppm > CLAMP_PPM_LIMIT {
        return Err(RvllmError::Loader {
            err: LoaderError::Fp8MisScaled {
                tensor: tensor_name.to_string(),
                clamp_ppm: ppm,
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: Some(tensor_name.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_values_have_zero_clamp() {
        // All within ±448 after scaling ⇒ scale = 10/448, all q ∈ [-1,1].
        let vs = vec![1.0f32, -2.0, 3.0, -4.0, 10.0, -10.0];
        let r = quantize_per_tensor_ref(&vs);
        assert!(r.clamp_ppm.abs() < 1e-3);
    }

    #[test]
    fn gate_rejects_bad_tensor() {
        let err =
            check_clamp_gate("qkv.0", 500.0, std::path::Path::new("x.safetensors")).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("Fp8MisScaled"));
        assert!(s.contains("qkv.0"));
    }

    #[test]
    fn gate_accepts_good_tensor() {
        assert!(check_clamp_gate("qkv.0", 3.0, std::path::Path::new("x")).is_ok());
    }
}
