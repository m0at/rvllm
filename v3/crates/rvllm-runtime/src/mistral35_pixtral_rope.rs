//! Pixtral 2D rotary position embedding — host-side cos/sin tables.
//!
//! Mirrors the kernel ABI in `kernels/pixtral_rotary_2d_bf16.cu`:
//!
//! ```text
//! cos[t, 0..R/2]   = cos(theta_i_row * row[t])    // i in [0, R/2)
//! cos[t, R/2..R]   = cos(theta_i_col * col[t])    // i in [0, R/2)
//! sin[...]         = same with sin()
//! ```
//!
//! where:
//!   * `R` is the rotary dim per head (Pixtral uses full rotary
//!     `R = head_dim = 104`).
//!   * `theta_i = base ** (-2*i / R)` for i in [0, R/2). The base is
//!     `vision_config.rope_theta = 10_000.0`.
//!   * `(row[t], col[t])` is the patch grid coordinate of token `t`
//!     in document order: `row = t / grid_w`, `col = t % grid_w`.
//!
//! NeoX rotate_half pairing matches the existing
//! `pixtral_rotary_2d_bf16` kernel, so the table layout above is
//! load-bearing — do not split row/col freqs across an
//! interleaved layout without also rewriting the kernel.
//!
//! Round-12 (Pixtral vision phase 1).

/// Pre-computed cos/sin tables for the Pixtral 2D RoPE applied to
/// Q + K of every block. Built once per image at request time
/// (cheap: a few µs of host work for typical grids).
#[derive(Debug, Clone)]
pub struct PixtralRopeTables {
    /// `[seq_len, rotary_dim]` row-major f32. seq_len = grid_h * grid_w.
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub seq_len: usize,
    pub rotary_dim: usize,
    pub grid_h: u32,
    pub grid_w: u32,
}

impl PixtralRopeTables {
    /// Mirrors HF transformers PixtralRotaryEmbedding (verified against
    /// `transformers.models.pixtral.modeling_pixtral`):
    ///
    /// ```text
    /// freqs = 1 / base^(arange(0, dim, 2) / dim)               # [dim/2]
    /// row_freqs = freqs[::2]                                    # [dim/4]
    /// col_freqs = freqs[1::2]                                   # [dim/4]
    /// freqs_h[h, k]  = h * row_freqs[k]
    /// freqs_w[w, k]  = w * col_freqs[k]
    /// inv_freq_2d[h, w, :] = cat(freqs_h[h], freqs_w[w])        # [dim/2]
    /// inv_freq      = inv_freq_2d.reshape(-1, dim/2)            # [N, dim/2]
    /// inv_freq      = cat([inv_freq, inv_freq], dim=-1)         # [N, dim]
    /// cos = cos(inv_freq)
    /// sin = sin(inv_freq)
    /// ```
    ///
    /// For dim=104 this yields 26 row freqs + 26 col freqs (= 52
    /// distinct per token) with the second-half [dim/2, dim) being a
    /// byte-identical copy of the first-half [0, dim/2). The repeat
    /// supports the NeoX rotate_half pairing the kernel implements
    /// (channel i pairs with channel i + dim/2).
    ///
    /// Round-12 phase 3-test (c) fix: the previous implementation used
    /// 52 row + 52 col distinct freqs (from `inv_freq[i]` for
    /// i in [0..52)), which produced cos=0.999 per block but
    /// accumulated to cos=0.69 over 48 blocks vs HF reference.
    pub fn build(grid_h: u32, grid_w: u32, head_dim: usize, rope_theta: f32) -> Self {
        let r = head_dim;
        assert!(r % 2 == 0, "rotary_dim must be even, got {r}");
        assert!((r / 2) % 2 == 0,
            "rotary_dim/2 must be even for NeoX rotate_half (got {})", r/2);
        let half = r / 2;
        let quarter = r / 4;
        let seq_len = (grid_h as usize) * (grid_w as usize);

        // freqs = 1.0 / base ** (arange(0, R, 2) / R)  → length R/2
        let mut freqs = Vec::with_capacity(half);
        let inv_r = 1.0 / (r as f32);
        for i in (0..r).step_by(2) {
            let exp = (i as f32) * inv_r;
            freqs.push(rope_theta.powf(-exp));
        }
        // row_freqs = freqs[::2], col_freqs = freqs[1::2]  → each R/4 long
        let row_freqs: Vec<f32> = freqs.iter().step_by(2).copied().collect();
        let col_freqs: Vec<f32> = freqs.iter().skip(1).step_by(2).copied().collect();
        debug_assert_eq!(row_freqs.len(), quarter);
        debug_assert_eq!(col_freqs.len(), quarter);

        let mut cos = vec![0.0f32; seq_len * r];
        let mut sin = vec![0.0f32; seq_len * r];

        for t in 0..seq_len {
            let row = (t / grid_w as usize) as f32;
            let col = (t % grid_w as usize) as f32;
            let row_off = t * r;
            // First half: [0..R/4) row freqs, [R/4..R/2) col freqs.
            for k in 0..quarter {
                let phase = row * row_freqs[k];
                cos[row_off + k] = phase.cos();
                sin[row_off + k] = phase.sin();
            }
            for k in 0..quarter {
                let phase = col * col_freqs[k];
                cos[row_off + quarter + k] = phase.cos();
                sin[row_off + quarter + k] = phase.sin();
            }
            // Second half [R/2..R): byte-identical copy of [0..R/2).
            // This satisfies the NeoX rotate_half pairing: channel i in
            // [0, R/2) pairs with channel i + R/2 in [R/2, R), and
            // they share the same cos/sin values.
            for k in 0..half {
                cos[row_off + half + k] = cos[row_off + k];
                sin[row_off + half + k] = sin[row_off + k];
            }
        }

        Self {
            cos, sin, seq_len, rotary_dim: r, grid_h, grid_w,
        }
    }

    /// Bytes the cos/sin upload will occupy on the device.
    pub fn device_bytes(&self) -> usize {
        self.cos.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_matches_grid_and_head_dim() {
        let t = PixtralRopeTables::build(4, 5, 104, 10_000.0);
        assert_eq!(t.seq_len, 20);
        assert_eq!(t.rotary_dim, 104);
        assert_eq!(t.cos.len(), 20 * 104);
        assert_eq!(t.sin.len(), 20 * 104);
        assert_eq!(t.grid_h, 4);
        assert_eq!(t.grid_w, 5);
    }

    #[test]
    fn position_zero_is_identity() {
        // At (row, col) = (0, 0), every cos = 1, every sin = 0.
        let t = PixtralRopeTables::build(3, 3, 104, 10_000.0);
        let r = 104;
        for i in 0..r {
            let c = t.cos[0 * r + i];
            let s = t.sin[0 * r + i];
            assert!((c - 1.0).abs() < 1e-6, "cos[0,{i}]={c}");
            assert!(s.abs() < 1e-6, "sin[0,{i}]={s}");
        }
    }

    #[test]
    fn row_quarter_uses_row_position() {
        // Channels [0..R/4) carry row freqs. Tokens 0 and grid_w sit
        // at (0,0) and (1,0). The row-quarter [0..R/4) must differ
        // between them (row=0 vs 1); the col-quarter [R/4..R/2) must
        // match (col=0 for both); the second-half [R/2..R) is a copy.
        let grid_h = 3u32;
        let grid_w = 4u32;
        let t = PixtralRopeTables::build(grid_h, grid_w, 104, 10_000.0);
        let r = 104;
        let quarter = r / 4;
        let half = r / 2;
        let t0_off = 0 * r;
        let t1_off = (grid_w as usize) * r;
        // col-quarter identical (both col=0)
        for k in quarter..half {
            let c0 = t.cos[t0_off + k];
            let c1 = t.cos[t1_off + k];
            assert!((c0 - c1).abs() < 1e-6,
                "col-quarter differs at k={k}: t0={c0} t1={c1}");
        }
        // row-quarter at k=0: row_freqs[0] = freqs[0] = 1.0, so at
        // row=1 we expect cos(1).
        let row_cos_t1_k0 = t.cos[t1_off + 0];
        assert!((row_cos_t1_k0 - (1.0_f32).cos()).abs() < 1e-5,
            "row cos at t=grid_w, k=0 should be cos(1)={}, got {}",
            (1.0_f32).cos(), row_cos_t1_k0);
    }

    #[test]
    fn col_quarter_uses_col_position() {
        // Tokens 0 and 1 sit at (0,0) and (0,1). row-quarter
        // identical; col-quarter differs.
        let t = PixtralRopeTables::build(2, 4, 104, 10_000.0);
        let r = 104;
        let quarter = r / 4;
        // row-quarter [0..R/4) — row=0 for both → all cos=1.
        for k in 0..quarter {
            assert!((t.cos[0 * r + k] - t.cos[1 * r + k]).abs() < 1e-6,
                "row-quarter differs between t=0 and t=1 at k={k}");
            assert!((t.cos[0 * r + k] - 1.0).abs() < 1e-6,
                "row=0 row-quarter should be cos(0)=1, got {}", t.cos[k]);
        }
        // col-quarter at k=R/4: col_freqs[0] = freqs[1] = 1/base^(2/R).
        // For R=104, base=10000: col_freqs[0] = 10000^(-2/104) ≈ 0.8385.
        // At col=1: cos(0.8385) ≈ 0.6688.
        let col_freq_0 = (10_000.0_f32).powf(-2.0 / 104.0);
        let expected = (col_freq_0).cos();
        let col_cos_t1_k0 = t.cos[1 * r + quarter + 0];
        assert!((col_cos_t1_k0 - expected).abs() < 1e-5,
            "col cos at t=(0,1), k=R/4 should be cos({col_freq_0})={expected}, got {col_cos_t1_k0}");
    }

    #[test]
    fn second_half_mirrors_first_half() {
        // The HF Pixtral convention duplicates [0..R/2) into [R/2..R)
        // so the NeoX rotate_half pairing (channel i pairs with i+R/2)
        // sees the same cos/sin in both. Verify byte-equal copy.
        let t = PixtralRopeTables::build(3, 5, 104, 10_000.0);
        let r = 104;
        let half = r / 2;
        for tok in 0..t.seq_len {
            for k in 0..half {
                let lo = t.cos[tok * r + k];
                let hi = t.cos[tok * r + half + k];
                assert!((lo - hi).abs() < 1e-9,
                    "second-half mismatch at tok={tok} k={k}: {lo} vs {hi}");
                let slo = t.sin[tok * r + k];
                let shi = t.sin[tok * r + half + k];
                assert!((slo - shi).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        let t = PixtralRopeTables::build(7, 11, 104, 10_000.0);
        for k in 0..t.cos.len() {
            let s = t.cos[k] * t.cos[k] + t.sin[k] * t.sin[k];
            assert!((s - 1.0).abs() < 1e-5,
                "cos²+sin² ≠ 1 at idx {k}: {s}");
        }
    }
}
