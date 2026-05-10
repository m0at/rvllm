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
    pub fn build(grid_h: u32, grid_w: u32, head_dim: usize, rope_theta: f32) -> Self {
        // Pixtral applies rotary to the full head_dim; the kernel's
        // first-R/2 channels carry row info, the second-R/2 carry col.
        // R/2 must be even so the NeoX rotate_half paring works inside
        // each half — head_dim=104 → R/2=52 (even). Future Pixtral
        // checkpoints with non-even half would need a kernel rewrite.
        let r = head_dim;
        assert!(r % 2 == 0, "rotary_dim must be even, got {r}");
        assert!((r / 2) % 2 == 0,
            "rotary_dim/2 must be even for NeoX rotate_half (got {})", r/2);
        let half = r / 2;
        let seq_len = (grid_h as usize) * (grid_w as usize);

        // theta_i = base ** (-2*i / R) for i in [0, half).
        // The same theta_i is reused for the row half and the col half;
        // the position multiplier (row vs col) is what differs.
        let mut inv_freq = Vec::with_capacity(half);
        let inv_r = 1.0 / (r as f32);
        for i in 0..half {
            let exp = (-2.0_f32) * (i as f32) * inv_r;
            inv_freq.push(rope_theta.powf(exp));
        }

        let mut cos = vec![0.0f32; seq_len * r];
        let mut sin = vec![0.0f32; seq_len * r];

        for t in 0..seq_len {
            let row = (t / grid_w as usize) as f32;
            let col = (t % grid_w as usize) as f32;
            let row_off = t * r;
            for i in 0..half {
                let phase_row = row * inv_freq[i];
                cos[row_off + i] = phase_row.cos();
                sin[row_off + i] = phase_row.sin();
            }
            for i in 0..half {
                let phase_col = col * inv_freq[i];
                cos[row_off + half + i] = phase_col.cos();
                sin[row_off + half + i] = phase_col.sin();
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
    fn row_half_uses_row_position() {
        // Tokens 0 and grid_w sit at (0,0) and (1,0). Their col
        // halves (i ∈ [R/2, R)) must be identical (col=0 for both);
        // their row halves must differ (row=0 vs 1).
        let grid_h = 3u32;
        let grid_w = 4u32;
        let t = PixtralRopeTables::build(grid_h, grid_w, 104, 10_000.0);
        let r = 104;
        let half = r / 2;
        let t0_off = 0 * r;
        let t1_off = (grid_w as usize) * r;
        // Col halves identical
        for i in 0..half {
            let c0 = t.cos[t0_off + half + i];
            let c1 = t.cos[t1_off + half + i];
            assert!((c0 - c1).abs() < 1e-6,
                "col half differs at i={i}: t0={c0} t1={c1}");
        }
        // Row halves: at row=1 cos(theta_0 * 1) ≠ 1 (theta_0 = 1.0,
        // so cos(1) ≈ 0.5403).
        let row_cos_t1_i0 = t.cos[t1_off + 0];
        assert!((row_cos_t1_i0 - (1.0_f32).cos()).abs() < 1e-5,
            "row cos at t=grid_w, i=0 should be cos(1)={}; got {}",
            (1.0_f32).cos(), row_cos_t1_i0);
    }

    #[test]
    fn col_half_uses_col_position() {
        // Tokens 0 and 1 sit at (0,0) and (0,1). Row halves identical
        // (row=0); col halves differ (col=0 vs 1).
        let t = PixtralRopeTables::build(2, 4, 104, 10_000.0);
        let r = 104;
        let half = r / 2;
        for i in 0..half {
            assert!((t.cos[0 * r + i] - t.cos[1 * r + i]).abs() < 1e-6,
                "row half differs between t=0 and t=1 at i={i}");
        }
        let col_cos_t1_i0 = t.cos[1 * r + half + 0];
        assert!((col_cos_t1_i0 - (1.0_f32).cos()).abs() < 1e-5);
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
