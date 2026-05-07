//! Mistral 3.5 YaRN RoPE math helpers.
//!
//! Pure CPU arithmetic mirroring the canonical YaRN scaling formula
//! used by HuggingFace transformers' Mistral 3 / Mixtral / Llama
//! implementations. The future fused CUDA kernel
//! (`kernels/fused_yarn_rope_nvfp4kv_mistral.cu`) will recompute the
//! same per-frequency `inv_freq[i]` table on device, then apply the
//! `mscale` post-multiplier on cos/sin. Keeping the math in pure
//! Rust here gives us:
//!
//! 1. A unit-testable reference for the kernel to validate against.
//! 2. Startup-time computation of the RoPE table for prompts inside
//!    `original_max_position_embeddings` (no GPU work needed).
//! 3. A tested derivation of `mscale_actual` so the runtime applies
//!    the correct attention output scaling.
//!
//! The Mistral 3.5 NVFP4 checkpoint ships:
//!   rope_theta = 1_000_000.0
//!   rope_type  = "yarn"
//!   original_max_position_embeddings = 4096
//!   factor     = 64.0  (extends 4096 -> 262144)
//!   beta_fast  = 4.0
//!   beta_slow  = 1.0
//!   mscale         = 1.0
//!   mscale_all_dim = 0.0  (validated via Mistral35Arch::from_dir)

use rvllm_loader::mistral35_arch::YarnRopeConfig;

/// Frequency-correction-dim helper, the inverse of YaRN's
/// "rotational period equals max_pos / num_rotations" condition.
/// Returns the (real-valued) frequency index `i` at which the
/// corresponding rotation has period `original_max / num_rotations`
/// — used to bracket the extrapolation/interpolation ramp.
fn find_correction_dim(
    num_rotations: f32,
    dim: usize,
    rope_theta: f32,
    original_max_position_embeddings: usize,
) -> f32 {
    use std::f32::consts::PI;
    let dimf = dim as f32;
    let omp = original_max_position_embeddings as f32;
    // dim * ln(omp / (num_rotations * 2 * pi)) / (2 * ln(theta))
    (dimf * (omp / (num_rotations * 2.0 * PI)).ln()) / (2.0 * rope_theta.ln())
}

/// Bracket `[low_dim, high_dim]` for the YaRN ramp. Clamped to a
/// valid `[0, dim-1]` range and split when the two correction dims
/// collide.
///
/// Naming convention matches the canonical YaRN reference impl:
/// `beta_fast` (more rotations) corresponds to the higher-frequency
/// freq-index regime — i.e. the LOWER `dim` index — and is the
/// extrapolation boundary. `beta_slow` (fewer rotations) sits at
/// the LOW-frequency / interpolation side, mapping to a HIGHER
/// `dim` index. So `low_dim` comes from `beta_fast` and
/// `high_dim` from `beta_slow`.
fn yarn_correction_range(
    cfg: &YarnRopeConfig,
    head_dim: usize,
) -> (f32, f32) {
    let low = find_correction_dim(
        cfg.beta_fast,
        head_dim,
        cfg.rope_theta,
        cfg.original_max_position_embeddings,
    )
    .floor()
    .max(0.0);
    let high = find_correction_dim(
        cfg.beta_slow,
        head_dim,
        cfg.rope_theta,
        cfg.original_max_position_embeddings,
    )
    .ceil()
    .min((head_dim as f32) - 1.0);
    if (high - low).abs() < 1e-3 {
        // Degenerate: bracket has zero width. Add a hairline so the
        // ramp denominator stays non-zero.
        (low, low + 0.001)
    } else {
        (low, high)
    }
}

/// `mscale` post-multiplier applied to cos/sin. With Mistral's
/// `mscale_all_dim = 0`, the formula collapses to
/// `0.1 * mscale * ln(factor) + 1`. With `factor <= 1` the
/// multiplier is exactly 1.0.
pub fn yarn_mscale(cfg: &YarnRopeConfig) -> f32 {
    if cfg.factor <= 1.0 {
        return 1.0;
    }
    let scale = if cfg.mscale_all_dim > 0.0 {
        cfg.mscale_all_dim
    } else {
        cfg.mscale
    };
    0.1 * scale * cfg.factor.ln() + 1.0
}

/// Compute the YaRN-corrected `inv_freq[i]` table for `i in
/// 0..head_dim/2`. Output is `head_dim / 2` f32 values.
///
/// Algorithm (per the canonical YaRN paper / HF reference):
///   1. base inv_freq:  `1 / theta ** (2i / d)`
///   2. ramp from low → high correction dim, clamped to [0, 1]
///   3. final inv_freq = interp(low) * (1 - ramp) + extrap(high) * ramp
///      where interp = base / factor (squeeze period × factor)
///      and   extrap = base                  (no scaling)
///
/// Caller multiplies cos/sin by `yarn_mscale(cfg)` before applying
/// to Q/K so the attention output magnitude tracks the trained
/// `original_max` regime.
pub fn yarn_inv_freq(cfg: &YarnRopeConfig, head_dim: usize) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim % 2 == 0,
        "yarn_inv_freq: head_dim={head_dim} must be even and >0"
    );
    let half = head_dim / 2;
    let mut out = Vec::with_capacity(half);

    // Base inv_freq table.
    let inv_freq_base: Vec<f32> = (0..half)
        .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    let (low_dim, high_dim) = yarn_correction_range(cfg, head_dim);

    let inv_factor = if cfg.factor > 0.0 { 1.0 / cfg.factor } else { 1.0 };

    for i in 0..half {
        // ramp ∈ [0, 1]: 0 at the high-freq end (low i), 1 at the
        // low-freq end (high i). Per YaRN reference impl, this is
        // the proportion of *interpolation* applied at this index;
        // (1 - ramp) is the *extrapolation* proportion.
        let lin = ((i as f32) - low_dim) / (high_dim - low_dim);
        let ramp = lin.clamp(0.0, 1.0);
        let interp = inv_freq_base[i] * inv_factor; // squeezed (low freq)
        let extrap = inv_freq_base[i];              // unchanged (high freq)
        let combined = interp * ramp + extrap * (1.0 - ramp);
        out.push(combined);
    }
    out
}

/// One-shot helper that returns both the `inv_freq` table and the
/// `mscale` multiplier so a caller can build (cos, sin) tables in
/// one pass:
///
/// ```text
/// for pos in 0..max_pos:
///     for i in 0..d/2:
///         angle = pos * inv_freq[i]
///         cos[pos, i] = mscale * cos(angle)
///         sin[pos, i] = mscale * sin(angle)
/// ```
pub fn yarn_inv_freq_and_mscale(
    cfg: &YarnRopeConfig,
    head_dim: usize,
) -> (Vec<f32>, f32) {
    (yarn_inv_freq(cfg, head_dim), yarn_mscale(cfg))
}

/// Pre-computed cos/sin RoPE tables for Mistral 3.5 YaRN. Stored
/// in row-major `[max_pos, head_dim/2]` f32 layout — exactly the
/// shape the future fused CUDA kernel
/// (`fused_yarn_rope_nvfp4kv_mistral.cu`) takes as input. Tables
/// are mscale-applied (multiplied through cos/sin) so the kernel
/// just does a lookup + complex-number rotation per element pair.
#[derive(Clone, Debug)]
pub struct YarnRopeTables {
    /// `[max_pos, head_dim / 2]` f32 row-major. `cos[pos * (d/2) + i]
    /// = mscale * cos(pos * inv_freq[i])`.
    pub cos: Vec<f32>,
    /// Same shape as `cos`; `sin[pos * (d/2) + i] = mscale *
    /// sin(pos * inv_freq[i])`.
    pub sin: Vec<f32>,
    pub max_pos: usize,
    pub head_dim: usize,
    pub mscale: f32,
}

impl YarnRopeTables {
    /// Total f32 element count (`cos` + `sin` combined).
    pub fn total_elements(&self) -> usize {
        self.cos.len() + self.sin.len()
    }
    /// Total bytes for one `cos` or `sin` half (caller doubles for
    /// both buffers).
    pub fn half_bytes(&self) -> usize {
        self.max_pos * (self.head_dim / 2) * std::mem::size_of::<f32>()
    }
}

/// Build the YaRN cos/sin tables for `pos in 0..max_pos`. `max_pos`
/// is typically `original_max_position_embeddings` at startup
/// (4096 for Mistral 3.5) — positions beyond that fall through to
/// the device-side per-position table extension when the kernel
/// lands. With `head_dim=128, max_pos=4096` the tables are
/// `4096 * 64 * 4` = 1 MiB each = 2 MiB total, comfortably
/// resident in HBM.
pub fn build_yarn_rope_tables(
    cfg: &YarnRopeConfig,
    head_dim: usize,
    max_pos: usize,
) -> YarnRopeTables {
    assert!(
        head_dim > 0 && head_dim % 2 == 0,
        "build_yarn_rope_tables: head_dim={head_dim} must be even and >0"
    );
    assert!(max_pos > 0, "build_yarn_rope_tables: max_pos must be >0");
    let (inv_freq, mscale) = yarn_inv_freq_and_mscale(cfg, head_dim);
    let half = head_dim / 2;
    let total = max_pos * half;
    let mut cos = Vec::with_capacity(total);
    let mut sin = Vec::with_capacity(total);
    for pos in 0..max_pos {
        let posf = pos as f32;
        for i in 0..half {
            let angle = posf * inv_freq[i];
            cos.push(mscale * angle.cos());
            sin.push(mscale * angle.sin());
        }
    }
    YarnRopeTables { cos, sin, max_pos, head_dim, mscale }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mistral_yarn() -> YarnRopeConfig {
        YarnRopeConfig {
            rope_theta: 1_000_000.0,
            original_max_position_embeddings: 4096,
            factor: 64.0,
            beta_fast: 4.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        }
    }

    #[test]
    fn mscale_with_factor_64_matches_yarn_paper() {
        // 0.1 * 1.0 * ln(64) + 1 ≈ 0.1 * 4.15888 + 1 ≈ 1.41589
        let m = yarn_mscale(&mistral_yarn());
        assert!((m - 1.41589).abs() < 1e-3, "mscale={m}");
    }

    #[test]
    fn mscale_collapses_to_one_when_factor_le_one() {
        let mut cfg = mistral_yarn();
        cfg.factor = 1.0;
        assert_eq!(yarn_mscale(&cfg), 1.0);
        cfg.factor = 0.5;
        assert_eq!(yarn_mscale(&cfg), 1.0);
    }

    #[test]
    fn mscale_all_dim_overrides_when_positive() {
        let mut cfg = mistral_yarn();
        cfg.mscale_all_dim = 2.0;
        // Switches to the all-dim formula: 0.1 * 2.0 * ln(64) + 1
        let m = yarn_mscale(&cfg);
        assert!((m - (0.1 * 2.0 * 64.0_f32.ln() + 1.0)).abs() < 1e-3);
    }

    #[test]
    fn inv_freq_length_matches_half_head_dim() {
        let cfg = mistral_yarn();
        for d in [64, 128, 256] {
            let f = yarn_inv_freq(&cfg, d);
            assert_eq!(f.len(), d / 2);
            // All entries must be positive and finite.
            for v in &f {
                assert!(v.is_finite() && *v > 0.0, "bad inv_freq {v}");
            }
        }
    }

    #[test]
    fn inv_freq_at_high_freq_equals_extrapolation() {
        // The lowest-index frequency (highest rotational frequency)
        // sits in the extrapolation regime — should not be scaled
        // down by `factor`. inv_freq[0] = 1 / theta^0 = 1.0.
        let cfg = mistral_yarn();
        let f = yarn_inv_freq(&cfg, 128);
        assert!((f[0] - 1.0).abs() < 1e-6, "f[0]={}", f[0]);
    }

    #[test]
    fn inv_freq_at_low_freq_is_squeezed_by_factor() {
        // The highest-index frequency (lowest rotational frequency)
        // sits in the interpolation regime — divided by `factor=64`.
        // base = 1 / theta^((d-2)/d) ≈ 1 / theta for d=128 with one
        // d/2-th from the top.
        let cfg = mistral_yarn();
        let head_dim = 128usize;
        let f = yarn_inv_freq(&cfg, head_dim);
        let last = *f.last().unwrap();
        let base_last = 1.0 / cfg.rope_theta.powf((head_dim as f32 - 2.0) / head_dim as f32);
        // Within the interpolation window the result is base / factor.
        // Allow a generous tolerance because i = d/2-1 sits AT the
        // bracket edge, but tests assert it's *closer* to base/factor
        // than to base.
        let interp = base_last / cfg.factor;
        let extrap = base_last;
        assert!(
            (last - interp).abs() < (last - extrap).abs(),
            "last={last} interp={interp} extrap={extrap}"
        );
    }

    #[test]
    fn inv_freq_correction_range_is_within_head_dim() {
        let cfg = mistral_yarn();
        let head_dim = 128;
        let (lo, hi) = yarn_correction_range(&cfg, head_dim);
        assert!(lo >= 0.0 && lo <= head_dim as f32);
        assert!(hi > lo && hi <= head_dim as f32);
    }

    #[test]
    fn one_shot_helper_matches_individual_calls() {
        let cfg = mistral_yarn();
        let (a, b) = yarn_inv_freq_and_mscale(&cfg, 128);
        assert_eq!(a, yarn_inv_freq(&cfg, 128));
        assert_eq!(b, yarn_mscale(&cfg));
    }

    #[test]
    #[should_panic]
    fn yarn_inv_freq_rejects_odd_head_dim() {
        let cfg = mistral_yarn();
        let _ = yarn_inv_freq(&cfg, 65);
    }

    #[test]
    fn rope_tables_shape_and_pos_zero() {
        let cfg = mistral_yarn();
        let head_dim = 128usize;
        let max_pos = 32usize;
        let t = build_yarn_rope_tables(&cfg, head_dim, max_pos);
        assert_eq!(t.head_dim, head_dim);
        assert_eq!(t.max_pos, max_pos);
        assert_eq!(t.cos.len(), max_pos * (head_dim / 2));
        assert_eq!(t.sin.len(), max_pos * (head_dim / 2));
        // pos=0: cos = mscale, sin = 0 across all i.
        for i in 0..head_dim / 2 {
            assert!((t.cos[i] - t.mscale).abs() < 1e-5,
                    "pos=0,i={i} cos={} mscale={}", t.cos[i], t.mscale);
            assert!(t.sin[i].abs() < 1e-5,
                    "pos=0,i={i} sin={}", t.sin[i]);
        }
    }

    #[test]
    fn rope_tables_match_inv_freq_at_pos_one() {
        let cfg = mistral_yarn();
        let head_dim = 128usize;
        let t = build_yarn_rope_tables(&cfg, head_dim, 4);
        let (inv_freq, mscale) = yarn_inv_freq_and_mscale(&cfg, head_dim);
        let half = head_dim / 2;
        // pos=1: angle = 1 * inv_freq[i]
        for i in 0..half {
            let angle = inv_freq[i];
            let expect_cos = mscale * angle.cos();
            let expect_sin = mscale * angle.sin();
            let got_cos = t.cos[1 * half + i];
            let got_sin = t.sin[1 * half + i];
            assert!((got_cos - expect_cos).abs() < 1e-5,
                    "i={i} got_cos={got_cos} expect={expect_cos}");
            assert!((got_sin - expect_sin).abs() < 1e-5,
                    "i={i} got_sin={got_sin} expect={expect_sin}");
        }
    }

    #[test]
    fn rope_tables_mistral_steady_state_size() {
        // At Mistral 3.5's full original-max precompute window:
        //   max_pos=4096, head_dim=128 → 4096 * 64 = 262 144 f32
        //   = 1 MiB per buffer (cos + sin = 2 MiB total).
        let cfg = mistral_yarn();
        let t = build_yarn_rope_tables(&cfg, 128, 4096);
        assert_eq!(t.total_elements(), 4096 * 64 * 2);
        assert_eq!(t.half_bytes(), 4096 * 64 * 4);
        assert_eq!(t.cos.len() * 4, t.half_bytes());
    }

    #[test]
    #[should_panic]
    fn rope_tables_reject_zero_max_pos() {
        let _ = build_yarn_rope_tables(&mistral_yarn(), 128, 0);
    }

    #[test]
    fn ramp_continuity_no_nans() {
        // Sweep a few head_dim values; ensure the ramp denominator
        // never produces NaNs/infs even at boundary head_dims.
        let cfg = mistral_yarn();
        for d in [16, 32, 48, 64, 96, 128, 256] {
            let f = yarn_inv_freq(&cfg, d);
            for v in &f {
                assert!(v.is_finite(), "head_dim={d} produced non-finite {v}");
            }
        }
    }
}
