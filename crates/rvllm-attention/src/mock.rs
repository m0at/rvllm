//! Mock attention backend for testing -- simple scaled dot-product on CPU data.

use half::f16;
use rvllm_core::prelude::Result;
use crate::buffer::GpuBuffer;

use crate::backend::AttentionBackend;

/// Mock attention backend that performs naive scaled dot-product attention
/// directly on the CPU-backed `GpuBuffer` data, without paging.
///
/// Useful for unit tests where correctness matters more than the paged layout.
pub struct MockAttentionBackend;

impl MockAttentionBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockAttentionBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionBackend for MockAttentionBackend {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        _block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        _max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        // Simple scaled dot-product: for each query token, attend over the
        // first `context_len` positions of key/value (treating them as flat
        // [total_positions, num_heads, head_dim]).

        let num_tokens = query.shape[0];
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];

        let kv_heads = key_cache.shape.get(2).copied().unwrap_or(num_heads);
        let kv_dim = key_cache.shape.last().copied().unwrap_or(head_dim);
        // Total positions stored flat in key_cache
        let total_kv_positions = if kv_heads > 0 && kv_dim > 0 {
            key_cache.data.len() / (kv_heads * kv_dim)
        } else {
            0
        };

        let mut output = vec![f16::ZERO; num_tokens * num_heads * head_dim];

        let ctx_len = context_lens
            .data
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as usize;
        let ctx_len = ctx_len.min(total_kv_positions);

        for t in 0..num_tokens {
            for h in 0..num_heads {
                let q_off = (t * num_heads + h) * head_dim;
                let q_vec: Vec<f32> = (0..head_dim)
                    .map(|d| query.data[q_off + d].to_f32())
                    .collect();

                let mut scores = Vec::with_capacity(ctx_len);
                for pos in 0..ctx_len {
                    let k_off = (pos * kv_heads + h) * kv_dim;
                    let dot: f32 = (0..head_dim)
                        .map(|d| q_vec[d] * key_cache.data[k_off + d].to_f32())
                        .sum();
                    scores.push(dot * scale);
                }

                if scores.is_empty() {
                    continue;
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_e: f32 = exps.iter().sum();

                let mut out_vec = vec![0.0f32; head_dim];
                for (pos, &w) in exps.iter().enumerate() {
                    let v_off = (pos * kv_heads + h) * kv_dim;
                    let weight = w / sum_e;
                    for d in 0..head_dim {
                        out_vec[d] += weight * value_cache.data[v_off + d].to_f32();
                    }
                }

                let o_off = (t * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    output[o_off + d] = f16::from_f32(out_vec[d]);
                }
            }
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_tokens, num_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "MockAttentionBackend"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_const_buf(val: f16, shape: Vec<usize>) -> GpuBuffer<f16> {
        let n: usize = shape.iter().product();
        GpuBuffer {
            data: vec![val; n],
            shape,
        }
    }

    #[test]
    fn mock_name() {
        assert_eq!(MockAttentionBackend::new().name(), "MockAttentionBackend");
    }

    #[test]
    fn mock_uniform_attention() {
        // With uniform keys, attention weights should be uniform and output
        // should equal the value vector.
        let num_tokens = 1;
        let num_heads = 2;
        let head_dim = 4;
        let ctx_len = 3;

        let one = f16::from_f32(1.0);
        let query = make_const_buf(one, vec![num_tokens, num_heads, head_dim]);
        let key_cache = make_const_buf(one, vec![ctx_len, num_heads, head_dim]);
        let value_cache = make_const_buf(one, vec![ctx_len, num_heads, head_dim]);
        let block_tables = GpuBuffer {
            data: vec![0],
            shape: vec![1, 1],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: vec![ctx_len as i32],
            shape: vec![1],
        };

        let backend = MockAttentionBackend::new();
        let out = backend
            .forward(&query, &key_cache, &value_cache, &block_tables, &context_lens, ctx_len, 1.0)
            .unwrap();

        assert_eq!(out.shape, vec![1, 2, 4]);
        // All output values should be ~1.0 (uniform attention over uniform values)
        for &v in &out.data {
            let f = v.to_f32();
            assert!((f - 1.0).abs() < 0.01, "expected ~1.0, got {f}");
        }
    }

    #[test]
    fn mock_zero_context() {
        let query = make_const_buf(f16::from_f32(1.0), vec![1, 1, 4]);
        let key_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 1, 4],
        };
        let value_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 1, 4],
        };
        let block_tables = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 0],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: vec![0],
            shape: vec![1],
        };

        let out = MockAttentionBackend::new()
            .forward(&query, &key_cache, &value_cache, &block_tables, &context_lens, 0, 1.0)
            .unwrap();
        // Output should be all zeros (no context to attend to)
        for &v in &out.data {
            assert_eq!(v, f16::ZERO);
        }
    }

    #[test]
    fn mock_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockAttentionBackend>();
    }
}
