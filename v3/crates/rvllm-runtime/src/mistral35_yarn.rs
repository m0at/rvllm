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

/// Apply YaRN-RoPE rotation to one Q or K head vector at a single
/// position, using pre-computed cos/sin tables. Pure CPU
/// reference — the future fused CUDA kernel does the same math
/// in-place on Q/K activations during the projection epilogue.
///
/// Layout convention follows the canonical Llama / Mistral RoPE:
/// the head vector is split into two halves, where element `i`
/// pairs with element `i + half`:
///
/// ```text
///   x_rot[i]        = x[i]        * cos[i] - x[i + half] * sin[i]
///   x_rot[i + half] = x[i + half] * cos[i] + x[i]        * sin[i]
/// ```
///
/// This matches HuggingFace's `rotate_half` convention. Mistral's
/// reference impl + the existing rvllm Gemma/Qwen kernels use this
/// layout; the NVFP4 kernel will too. Returns a new
/// `Vec<f32>` so tests / probes can compare against numpy refs.
pub fn apply_rope_pair(
    x: &[f32],
    cos_pos: &[f32],
    sin_pos: &[f32],
) -> Vec<f32> {
    assert_eq!(
        x.len() % 2, 0,
        "apply_rope_pair: head_dim={} must be even", x.len()
    );
    let half = x.len() / 2;
    assert_eq!(
        cos_pos.len(), half,
        "apply_rope_pair: cos_pos len={} expected {half}", cos_pos.len()
    );
    assert_eq!(
        sin_pos.len(), half,
        "apply_rope_pair: sin_pos len={} expected {half}", sin_pos.len()
    );
    let mut out = vec![0.0f32; x.len()];
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        out[i]        = a * cos_pos[i] - b * sin_pos[i];
        out[i + half] = b * cos_pos[i] + a * sin_pos[i];
    }
    out
}

/// Convenience: index into a [`YarnRopeTables`] at `pos` and call
/// [`apply_rope_pair`]. Returns the rotated head vector.
pub fn apply_rope_at(
    tables: &YarnRopeTables,
    x: &[f32],
    pos: usize,
) -> Vec<f32> {
    assert!(
        pos < tables.max_pos,
        "apply_rope_at: pos={pos} >= max_pos={}", tables.max_pos
    );
    let half = tables.head_dim / 2;
    let off = pos * half;
    apply_rope_pair(x, &tables.cos[off..off + half], &tables.sin[off..off + half])
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
    fn apply_rope_at_pos_zero_is_identity_modulo_mscale() {
        // At pos=0, cos = mscale, sin = 0 → rotated vector is just
        // x * mscale (component-wise).
        let cfg = mistral_yarn();
        let head_dim = 8usize;
        let t = build_yarn_rope_tables(&cfg, head_dim, 4);
        let x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
        let rot = apply_rope_at(&t, &x, 0);
        for i in 0..head_dim {
            let expect = x[i] * t.mscale;
            assert!(
                (rot[i] - expect).abs() < 1e-5,
                "i={i} rot={} expect={}", rot[i], expect
            );
        }
    }

    #[test]
    fn apply_rope_pair_rotation_invariant() {
        // For any (a, b) pair, the rotation preserves the
        // sum-of-squares (modulo the mscale^2 factor when applied
        // through the table). Verify with a hand-picked angle.
        let cos = vec![0.6f32];
        let sin = vec![0.8f32]; // unit vector: cos^2 + sin^2 = 1
        let x = vec![3.0f32, 4.0f32]; // norm² = 25
        let r = apply_rope_pair(&x, &cos, &sin);
        // r[0] = 3*0.6 - 4*0.8 = 1.8 - 3.2 = -1.4
        // r[1] = 4*0.6 + 3*0.8 = 2.4 + 2.4 = 4.8
        // norm² = 1.96 + 23.04 = 25.0 ✓
        assert!((r[0] - (-1.4)).abs() < 1e-5);
        assert!((r[1] - 4.8).abs() < 1e-5);
        let n2 = r.iter().map(|v| v * v).sum::<f32>();
        assert!((n2 - 25.0).abs() < 1e-5);
    }

    #[test]
    fn apply_rope_at_pos_one_matches_explicit_rotation() {
        // pos=1 → angle_i = inv_freq[i]; verify one element pair
        // matches the cos/sin formula directly.
        let cfg = mistral_yarn();
        let head_dim = 8usize;
        let t = build_yarn_rope_tables(&cfg, head_dim, 4);
        let (inv_freq, mscale) = yarn_inv_freq_and_mscale(&cfg, head_dim);

        let x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
        let rot = apply_rope_at(&t, &x, 1);

        let half = head_dim / 2;
        for i in 0..half {
            let angle = inv_freq[i];
            let c = mscale * angle.cos();
            let s = mscale * angle.sin();
            let expect_lo = x[i] * c - x[i + half] * s;
            let expect_hi = x[i + half] * c + x[i] * s;
            assert!((rot[i]        - expect_lo).abs() < 1e-5);
            assert!((rot[i + half] - expect_hi).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic]
    fn apply_rope_pair_rejects_size_mismatch() {
        // cos has half = 3, but x has head_dim = 4 (half = 2).
        let _ = apply_rope_pair(&[1.0, 2.0, 3.0, 4.0], &[1.0; 3], &[0.0; 3]);
    }

    #[test]
    #[should_panic]
    fn apply_rope_at_rejects_out_of_range_pos() {
        let cfg = mistral_yarn();
        let t = build_yarn_rope_tables(&cfg, 8, 4);
        let _ = apply_rope_at(&t, &[1.0; 8], 100);
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
