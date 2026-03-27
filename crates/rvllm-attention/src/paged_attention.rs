//! PagedAttention V2 backend -- reference CPU implementation + CUDA placeholder.

use half::f16;
use rvllm_core::prelude::{LLMError, Result};
use crate::buffer::GpuBuffer;

use crate::backend::AttentionBackend;

/// PagedAttention V2 backend.
///
/// Provides a naive CPU reference implementation for correctness testing.
/// The CUDA kernel path is a placeholder for future integration.
pub struct PagedAttentionV2 {
    _private: (),
}

impl PagedAttentionV2 {
    pub fn new() -> Self {
        tracing::debug!("initializing PagedAttentionV2 backend");
        Self { _private: () }
    }
}

impl Default for PagedAttentionV2 {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionBackend for PagedAttentionV2 {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        // Extract dimensions from query shape [num_tokens, num_heads, head_dim]
        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D, got {} dims",
                query.shape.len()
            )));
        }
        let num_tokens = query.shape[0];
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];

        // key_cache shape: [num_blocks, block_size, num_heads, head_dim]
        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache must be 4-D, got {} dims",
                key_cache.shape.len()
            )));
        }
        let block_size = key_cache.shape[1];

        let num_seqs = context_lens.data.len();
        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_heads, head_dim],
            });
        }

        let max_blocks_per_seq = block_tables.shape.get(1).copied().unwrap_or(0);

        // Output: [num_tokens, num_heads, head_dim]
        let mut output = vec![f16::ZERO; num_tokens * num_heads * head_dim];

        // Naive CPU paged attention: for each sequence, gather KV from paged
        // blocks and compute scaled dot-product attention.
        let mut token_offset = 0usize;
        for seq_idx in 0..num_seqs {
            let ctx_len = context_lens.data[seq_idx] as usize;
            let ctx_len = ctx_len.min(max_context_len);
            // Number of query tokens for this sequence (for decode, typically 1)
            let seq_tokens = if seq_idx + 1 < num_seqs {
                // Heuristic: in a flat batch each seq contributes tokens up to
                // the next seq boundary. For a simple reference impl, assume
                // decode mode: 1 token per sequence.
                1
            } else {
                (num_tokens - token_offset).max(1)
            };

            for t in 0..seq_tokens {
                let q_base = (token_offset + t) * num_heads * head_dim;
                if q_base + num_heads * head_dim > query.data.len() {
                    break;
                }

                for h in 0..num_heads {
                    // Gather query vector for this head
                    let q_start = q_base + h * head_dim;
                    let q_vec: Vec<f32> = (0..head_dim)
                        .map(|d| query.data[q_start + d].to_f32())
                        .collect();

                    // Compute attention scores over context keys
                    let mut scores = Vec::with_capacity(ctx_len);
                    for pos in 0..ctx_len {
                        let block_idx = pos / block_size;
                        let block_off = pos % block_size;
                        if block_idx >= max_blocks_per_seq {
                            break;
                        }
                        let phys_block =
                            block_tables.data[seq_idx * max_blocks_per_seq + block_idx] as usize;
                        // key_cache layout: [num_blocks, block_size, num_heads, head_dim]
                        let k_base =
                            ((phys_block * block_size + block_off) * num_heads + h) * head_dim;

                        let dot: f32 = (0..head_dim)
                            .map(|d| q_vec[d] * key_cache.data[k_base + d].to_f32())
                            .sum();
                        scores.push(dot * scale);
                    }

                    // Softmax
                    if scores.is_empty() {
                        continue;
                    }
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Weighted sum of values
                    let mut out_vec = vec![0.0f32; head_dim];
                    for (pos, &w) in exp_scores.iter().enumerate() {
                        let block_idx = pos / block_size;
                        let block_off = pos % block_size;
                        let phys_block =
                            block_tables.data[seq_idx * max_blocks_per_seq + block_idx] as usize;
                        let v_base =
                            ((phys_block * block_size + block_off) * num_heads + h) * head_dim;
                        let weight = w / sum_exp;
                        for d in 0..head_dim {
                            out_vec[d] += weight * value_cache.data[v_base + d].to_f32();
                        }
                    }

                    // Write output
                    let o_start = (token_offset + t) * num_heads * head_dim + h * head_dim;
                    for d in 0..head_dim {
                        output[o_start + d] = f16::from_f32(out_vec[d]);
                    }
                }
            }
            token_offset += seq_tokens;
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_tokens, num_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "PagedAttentionV2"
    }
}

// --- CUDA kernel placeholder ---
// When building with the `cuda` feature, the actual kernel launch would go
// here behind an `unsafe` block calling into `rvllm_gpu::ffi`.
//
// unsafe fn paged_attention_v2_cuda(
//     output: *mut f16,
//     query: *const f16,
//     key_cache: *const f16,
//     value_cache: *const f16,
//     ...
// ) { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_attention_v2_name() {
        let pa = PagedAttentionV2::new();
        assert_eq!(pa.name(), "PagedAttentionV2");
    }

    #[test]
    fn paged_attention_v2_empty_batch() {
        let pa = PagedAttentionV2::new();
        let query = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 4, 8],
        };
        let key_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 16, 4, 8],
        };
        let value_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 16, 4, 8],
        };
        let block_tables = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 0],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: Vec::new(),
            shape: vec![0],
        };
        let out = pa
            .forward(&query, &key_cache, &value_cache, &block_tables, &context_lens, 0, 1.0)
            .unwrap();
        assert!(out.data.is_empty());
    }

    #[test]
    fn rejects_wrong_query_dims() {
        let pa = PagedAttentionV2::new();
        let query = GpuBuffer {
            data: vec![f16::ZERO; 16],
            shape: vec![4, 4], // 2-D instead of 3-D
        };
        let kc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 1],
        };
        let vc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 1],
        };
        let bt = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![1],
            shape: vec![1],
        };
        assert!(pa.forward(&query, &kc, &vc, &bt, &cl, 1, 1.0).is_err());
    }
}
