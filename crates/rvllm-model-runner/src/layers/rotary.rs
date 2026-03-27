//! Rotary positional embedding (RoPE).

use half::f16;

use crate::bridge::{GpuBuffer, Result};

/// Cached rotary positional embeddings. Precomputes cos/sin tables so the
/// trig work happens once at construction, not on every forward pass.
pub struct RotaryEmbedding {
    cos_cache: Vec<f32>, // [max_seq_len, half_dim]
    sin_cache: Vec<f32>, // [max_seq_len, half_dim]
    head_dim: usize,
    half_dim: usize,
    _max_seq_len: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let table_len = max_seq_len * half_dim;
        let mut cos_cache = Vec::with_capacity(table_len);
        let mut sin_cache = Vec::with_capacity(table_len);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let theta = pos as f32 * freq;
                cos_cache.push(theta.cos());
                sin_cache.push(theta.sin());
            }
        }

        Self {
            cos_cache,
            sin_cache,
            head_dim,
            half_dim,
            _max_seq_len: max_seq_len,
        }
    }

    /// Apply RoPE, returning new query and key buffers.
    ///
    /// positions: [num_tokens]
    /// query: [num_tokens, num_heads * head_dim]
    /// key: [num_tokens, num_kv_heads * head_dim]
    pub fn apply(
        &self,
        positions: &[u32],
        query: &GpuBuffer<f16>,
        key: &GpuBuffer<f16>,
    ) -> Result<(GpuBuffer<f16>, GpuBuffer<f16>)> {
        let num_tokens = positions.len();
        let q_total = query.len() / num_tokens;
        let k_total = key.len() / num_tokens;

        let mut q_out = query.data.clone();
        let mut k_out = key.data.clone();

        let half_dim = self.half_dim;

        for t in 0..num_tokens {
            let pos = positions[t] as usize;
            let cache_off = pos * half_dim;

            let num_q_heads = q_total / self.head_dim;
            for h in 0..num_q_heads {
                let offset = t * q_total + h * self.head_dim;
                self.apply_rope_cached(&mut q_out, offset, cache_off);
            }

            let num_k_heads = k_total / self.head_dim;
            for h in 0..num_k_heads {
                let offset = t * k_total + h * self.head_dim;
                self.apply_rope_cached(&mut k_out, offset, cache_off);
            }
        }

        Ok((
            GpuBuffer::from_vec(q_out, query.shape.clone()),
            GpuBuffer::from_vec(k_out, key.shape.clone()),
        ))
    }

    /// Static forward that matches the old API -- builds a temporary cache.
    pub fn forward(
        positions: &[u32],
        query: &GpuBuffer<f16>,
        key: &GpuBuffer<f16>,
        head_dim: usize,
    ) -> Result<(GpuBuffer<f16>, GpuBuffer<f16>)> {
        let max_pos = positions.iter().copied().max().unwrap_or(0) as usize + 1;
        let emb = Self::new(head_dim, max_pos, 10000.0);
        emb.apply(positions, query, key)
    }

    /// Apply rotation to a single head using cached cos/sin tables.
    /// Processes 4 pairs per iteration to help autovectorization.
    #[inline]
    fn apply_rope_cached(&self, data: &mut [f16], offset: usize, cache_off: usize) {
        let half_dim = self.half_dim;
        let cos = &self.cos_cache[cache_off..cache_off + half_dim];
        let sin = &self.sin_cache[cache_off..cache_off + half_dim];

        let chunks = half_dim / 4;
        let _rem = half_dim % 4;

        for c in 0..chunks {
            let base = c * 4;
            let i0 = offset + base;
            let i1 = offset + half_dim + base;

            let x0_0 = data[i0].to_f32();
            let x0_1 = data[i0 + 1].to_f32();
            let x0_2 = data[i0 + 2].to_f32();
            let x0_3 = data[i0 + 3].to_f32();

            let x1_0 = data[i1].to_f32();
            let x1_1 = data[i1 + 1].to_f32();
            let x1_2 = data[i1 + 2].to_f32();
            let x1_3 = data[i1 + 3].to_f32();

            let c0 = cos[base];
            let c1 = cos[base + 1];
            let c2 = cos[base + 2];
            let c3 = cos[base + 3];

            let s0 = sin[base];
            let s1 = sin[base + 1];
            let s2 = sin[base + 2];
            let s3 = sin[base + 3];

            data[i0] = f16::from_f32(x0_0 * c0 - x1_0 * s0);
            data[i0 + 1] = f16::from_f32(x0_1 * c1 - x1_1 * s1);
            data[i0 + 2] = f16::from_f32(x0_2 * c2 - x1_2 * s2);
            data[i0 + 3] = f16::from_f32(x0_3 * c3 - x1_3 * s3);

            data[i1] = f16::from_f32(x0_0 * s0 + x1_0 * c0);
            data[i1 + 1] = f16::from_f32(x0_1 * s1 + x1_1 * c1);
            data[i1 + 2] = f16::from_f32(x0_2 * s2 + x1_2 * c2);
            data[i1 + 3] = f16::from_f32(x0_3 * s3 + x1_3 * c3);
        }

        // Handle remainder (head_dim/2 not divisible by 4).
        for i in (chunks * 4)..half_dim {
            let x0 = data[offset + i].to_f32();
            let x1 = data[offset + half_dim + i].to_f32();
            let cos_f = cos[i];
            let sin_f = sin[i];
            data[offset + i] = f16::from_f32(x0 * cos_f - x1 * sin_f);
            data[offset + half_dim + i] = f16::from_f32(x0 * sin_f + x1 * cos_f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0, all freqs are 0, cos=1, sin=0 -> identity.
        let head_dim = 4;
        let vals: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f16::from_f32(v)).collect();
        let q = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let k = GpuBuffer::from_vec(vals, vec![1, head_dim]);
        let (qr, kr) = RotaryEmbedding::forward(&[0], &q, &k, head_dim).unwrap();
        for i in 0..head_dim {
            assert!((qr.data[i].to_f32() - q.data[i].to_f32()).abs() < 0.01);
            assert!((kr.data[i].to_f32() - k.data[i].to_f32()).abs() < 0.01);
        }
    }

    #[test]
    fn rope_changes_nonzero_position() {
        let head_dim = 4;
        let vals: Vec<f16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&v| f16::from_f32(v)).collect();
        let q = GpuBuffer::from_vec(vals.clone(), vec![1, head_dim]);
        let k = GpuBuffer::from_vec(vals, vec![1, head_dim]);
        let (qr, _kr) = RotaryEmbedding::forward(&[100], &q, &k, head_dim).unwrap();
        // At pos=100 the rotation should differ from input.
        let changed = (0..head_dim).any(|i| (qr.data[i].to_f32() - q.data[i].to_f32()).abs() > 0.01);
        assert!(changed, "RoPE at pos=100 should change the vector");
    }
}
