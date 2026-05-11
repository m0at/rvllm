//! Pure-CPU per-layer math reference for the Mistral 3.5 decoder.
//!
//! Sister module to [`super::mistral35_yarn`]; together they form
//! the deterministic baseline a future cosine-validation harness
//! diffs the CUDA forward against. Every function here is small,
//! O(N) in elements, no parallelism — clarity over speed since the
//! tests run on tiny fixtures and the production path is the GPU
//! kernel.
//!
//! Coverage:
//!   * `silu` (Swish, σ(x) · x): the activation Mistral's MLP
//!     uses (`hidden_act = "silu"`, asserted at parse time).
//!   * `silu_mul`: gate-MLP fusion `silu(x_gate) · x_up` — the
//!     element-wise step between the gate/up projections and the
//!     down projection.
//!   * `rms_norm`: weighted RMS normalisation,
//!     `out[i] = w[i] · x[i] / sqrt(mean(x²) + eps)`.
//!   * `softmax_row`: stable per-row softmax (used by attention
//!     scores; the reference takes f32 and returns f32 — the GPU
//!     kernel does the f16/bf16 cast separately).

/// Element-wise SiLU / Swish: `σ(x) · x = x / (1 + exp(-x))`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU element-wise to a slice, in-place.
pub fn silu_inplace(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = silu(*v);
    }
}

/// Mistral MLP fusion: `out[i] = silu(gate[i]) * up[i]`. Both
/// inputs must be the same length.
pub fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(
        gate.len(), up.len(),
        "silu_mul: gate.len()={} up.len()={}", gate.len(), up.len()
    );
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// Weighted RMS normalisation per the Llama / Mistral convention:
///   `out[i] = w[i] · x[i] / sqrt(mean(x²) + eps)`.
///
/// `eps` is the model's `rms_norm_eps` (1e-5 for Mistral 3.5).
/// `w` is the learned per-channel scale (the loaded RMSNorm
/// weight tensor); `out.len()` must equal `x.len()` and `w.len()`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(), w.len(),
        "rms_norm: x.len()={} w.len()={}", x.len(), w.len()
    );
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(&xi, &wi)| wi * xi * scale).collect()
}

/// Numerically stable per-row softmax. Returns a fresh `Vec<f32>`.
pub fn softmax_row(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 {
        // Degenerate — uniform fallback so callers don't return NaNs.
        return vec![1.0 / x.len() as f32; x.len()];
    }
    let inv = 1.0 / s;
    exps.into_iter().map(|e| e * inv).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "approx_eq failed: a={a} b={b} tol={tol}");
    }

    #[test]
    fn silu_known_values() {
        // x=0 → silu = 0 / (1+1) = 0
        approx_eq(silu(0.0), 0.0, 1e-6);
        // x=1 → silu = 1 / (1 + e^-1) = 1 / 1.36788 ≈ 0.73106
        approx_eq(silu(1.0), 0.7310586, 1e-5);
        // x=-1 → silu ≈ -0.26894
        approx_eq(silu(-1.0), -0.26894143, 1e-5);
        // Large positive → ≈ x
        approx_eq(silu(20.0), 20.0, 1e-5);
        // Large negative → ≈ 0
        assert!(silu(-20.0).abs() < 1e-6);
    }

    #[test]
    fn silu_inplace_matches_silu() {
        let mut a = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
        let b: Vec<f32> = a.iter().map(|&x| silu(x)).collect();
        silu_inplace(&mut a);
        for (x, y) in a.iter().zip(&b) {
            approx_eq(*x, *y, 1e-7);
        }
    }

    #[test]
    fn silu_mul_matches_componentwise() {
        let g = vec![1.0f32, 2.0, -1.0];
        let u = vec![0.5f32, -1.0, 3.0];
        let out = silu_mul(&g, &u);
        assert_eq!(out.len(), 3);
        approx_eq(out[0], silu(1.0) * 0.5, 1e-6);
        approx_eq(out[1], silu(2.0) * -1.0, 1e-6);
        approx_eq(out[2], silu(-1.0) * 3.0, 1e-6);
    }

    #[test]
    #[should_panic]
    fn silu_mul_rejects_size_mismatch() {
        let _ = silu_mul(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn rms_norm_unit_weight_is_normaliser() {
        // x = [1, 1, 1, 1], w = [1, 1, 1, 1], eps = 0.
        // mean_sq = 1; scale = 1; out = [1, 1, 1, 1].
        let x = vec![1.0f32; 4];
        let w = vec![1.0f32; 4];
        let out = rms_norm(&x, &w, 0.0);
        for v in out {
            approx_eq(v, 1.0, 1e-6);
        }
    }

    #[test]
    fn rms_norm_scaled_input_recovers_unit_norm() {
        // Doubling x preserves the *normalised* output (modulo the
        // per-channel weight). Concretely: y = x / sqrt(mean(x²)).
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let w = vec![1.0f32; 8];
        let y_a = rms_norm(&x, &w, 0.0);
        let x2: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
        let y_b = rms_norm(&x2, &w, 0.0);
        for (a, b) in y_a.iter().zip(&y_b) {
            approx_eq(*a, *b, 1e-5);
        }
    }

    #[test]
    fn rms_norm_applies_per_channel_weight() {
        let x = vec![1.0f32; 4];
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let out = rms_norm(&x, &w, 0.0);
        // mean_sq=1, scale=1 → out[i] = w[i]
        for (got, expect) in out.iter().zip(&w) {
            approx_eq(*got, *expect, 1e-6);
        }
    }

    #[test]
    fn rms_norm_eps_protects_zero_input() {
        let x = vec![0.0f32; 4];
        let w = vec![1.0f32; 4];
        let out = rms_norm(&x, &w, 1e-5);
        for v in out {
            assert!(v.is_finite(), "rms_norm produced non-finite on zero input");
        }
    }

    #[test]
    #[should_panic]
    fn rms_norm_rejects_size_mismatch() {
        let _ = rms_norm(&[1.0, 2.0], &[1.0], 1e-5);
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let s = softmax_row(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = s.iter().sum();
        approx_eq(sum, 1.0, 1e-5);
    }

    #[test]
    fn softmax_row_handles_large_values() {
        // Without max-subtract the exp would overflow for these.
        let s = softmax_row(&[1000.0, 1001.0, 999.0]);
        let sum: f32 = s.iter().sum();
        approx_eq(sum, 1.0, 1e-5);
        // The largest input dominates.
        assert!(s[1] > s[0] && s[0] > s[2]);
    }

    #[test]
    fn softmax_row_empty_input() {
        let s = softmax_row(&[]);
        assert!(s.is_empty());
    }
}
