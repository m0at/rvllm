//! Multi-Latent Attention (MLA) backend for DeepSeek V3 style models.
//!
//! MLA stores compressed latent vectors instead of full K/V per head,
//! projecting back to K/V at attention time. This reduces KV cache size
//! by ~4-8x compared to standard multi-head attention.

use crate::backend::AttentionBackend;
use crate::buffer::GpuBuffer;
use half::f16;
use rvllm_core::prelude::{LLMError, Result};

pub struct MlaConfig {
    pub latent_dim: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
}

pub struct MlaAttentionBackend {
    config: MlaConfig,
    /// Down-projection matrix: [kv_lora_rank, num_kv_heads * head_dim]
    /// Projects compressed latents back to K space.
    k_down_proj: Vec<f32>,
    /// Down-projection matrix: [kv_lora_rank, num_kv_heads * head_dim]
    /// Projects compressed latents back to V space.
    v_down_proj: Vec<f32>,
}

impl MlaAttentionBackend {
    pub fn new(config: MlaConfig, k_down_proj: Vec<f32>, v_down_proj: Vec<f32>) -> Self {
        let expected = config.kv_lora_rank * config.num_kv_heads * config.head_dim;
        assert_eq!(
            k_down_proj.len(),
            expected,
            "k_down_proj size mismatch: expected {expected}, got {}",
            k_down_proj.len()
        );
        assert_eq!(
            v_down_proj.len(),
            expected,
            "v_down_proj size mismatch: expected {expected}, got {}",
            v_down_proj.len()
        );
        Self {
            config,
            k_down_proj,
            v_down_proj,
        }
    }

    /// Project a single latent vector [kv_lora_rank] to full KV [num_kv_heads * head_dim]
    /// using the given projection matrix.
    fn project_latent(&self, latent: &[f32], proj: &[f32], out: &mut [f32]) {
        let rank = self.config.kv_lora_rank;
        let out_dim = self.config.num_kv_heads * self.config.head_dim;
        debug_assert_eq!(latent.len(), rank);
        debug_assert_eq!(out.len(), out_dim);
        // out = proj^T @ latent
        // proj layout: [rank, out_dim] (row-major)
        for j in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..rank {
                acc += proj[i * out_dim + j] * latent[i];
            }
            out[j] = acc;
        }
    }
}

impl AttentionBackend for MlaAttentionBackend {
    /// MLA forward pass.
    ///
    /// The `key_cache` and `value_cache` here hold compressed latent vectors
    /// rather than full K/V. Their shape is:
    ///   key_cache (latent_cache): [num_blocks, block_size, 1, kv_lora_rank]
    ///   value_cache: unused (same buffer as key_cache; latents encode both K and V)
    ///
    /// We project each latent to full K and V, then run standard SDPA.
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        _value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D, got {} dims",
                query.shape.len()
            )));
        }
        let num_tokens = query.shape[0];
        let num_q_heads = query.shape[1];
        let head_dim = query.shape[2];

        if head_dim != self.config.head_dim {
            return Err(LLMError::GpuError(format!(
                "head_dim mismatch: query has {head_dim}, config has {}",
                self.config.head_dim
            )));
        }

        // key_cache shape: [num_blocks, block_size, 1, kv_lora_rank]
        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache (latent cache) must be 4-D, got {} dims",
                key_cache.shape.len()
            )));
        }
        let block_size = key_cache.shape[1];
        let kv_lora_rank = key_cache.shape[3];

        if kv_lora_rank != self.config.kv_lora_rank {
            return Err(LLMError::GpuError(format!(
                "kv_lora_rank mismatch: cache has {kv_lora_rank}, config has {}",
                self.config.kv_lora_rank
            )));
        }

        let num_seqs = context_lens.data.len();
        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_q_heads, head_dim],
            });
        }

        let max_blocks_per_seq = block_tables.shape.get(1).copied().unwrap_or(0);
        let num_kv_heads = self.config.num_kv_heads;
        let gqa_ratio = num_q_heads / num_kv_heads;
        let kv_full_dim = num_kv_heads * head_dim;

        let mut output = vec![f16::ZERO; num_tokens * num_q_heads * head_dim];

        // Scratch buffers for projected K and V per position
        let mut k_projected = vec![0.0f32; kv_full_dim];
        let mut v_projected = vec![0.0f32; kv_full_dim];
        let mut latent_f32 = vec![0.0f32; kv_lora_rank];

        let mut token_offset = 0usize;
        for seq_idx in 0..num_seqs {
            let ctx_len = (context_lens.data[seq_idx] as usize).min(max_context_len);
            let seq_tokens = if seq_idx + 1 < num_seqs {
                1
            } else {
                (num_tokens - token_offset).max(1)
            };

            for t in 0..seq_tokens {
                let q_base = (token_offset + t) * num_q_heads * head_dim;
                if q_base + num_q_heads * head_dim > query.data.len() {
                    break;
                }

                for qh in 0..num_q_heads {
                    let kv_head = qh / gqa_ratio;

                    // Gather query vector
                    let q_start = q_base + qh * head_dim;
                    let q_vec: Vec<f32> = (0..head_dim)
                        .map(|d| query.data[q_start + d].to_f32())
                        .collect();

                    // Compute attention scores: project each latent to K, dot with query
                    let mut scores = Vec::with_capacity(ctx_len);
                    // Store projected V for each position (needed for weighted sum)
                    let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(ctx_len);

                    for pos in 0..ctx_len {
                        let block_idx = pos / block_size;
                        let block_off = pos % block_size;
                        if block_idx >= max_blocks_per_seq {
                            break;
                        }
                        let phys_block =
                            block_tables.data[seq_idx * max_blocks_per_seq + block_idx] as usize;

                        // Read latent: [num_blocks, block_size, 1, kv_lora_rank]
                        let latent_base =
                            ((phys_block * block_size + block_off) * 1) * kv_lora_rank;
                        for i in 0..kv_lora_rank {
                            latent_f32[i] = key_cache.data[latent_base + i].to_f32();
                        }

                        // Project latent -> K
                        self.project_latent(&latent_f32, &self.k_down_proj, &mut k_projected);
                        // Project latent -> V
                        self.project_latent(&latent_f32, &self.v_down_proj, &mut v_projected);

                        // Dot product: q . k for this kv_head
                        let k_offset = kv_head * head_dim;
                        let dot: f32 = (0..head_dim)
                            .map(|d| q_vec[d] * k_projected[k_offset + d])
                            .sum();
                        scores.push(dot * scale);

                        // Store V slice for this kv_head
                        let v_offset = kv_head * head_dim;
                        v_cache.push(v_projected[v_offset..v_offset + head_dim].to_vec());
                    }

                    if scores.is_empty() {
                        continue;
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Weighted sum of values
                    let mut out_vec = vec![0.0f32; head_dim];
                    for (pos, &w) in exp_scores.iter().enumerate() {
                        let weight = w / sum_exp;
                        for d in 0..head_dim {
                            out_vec[d] += weight * v_cache[pos][d];
                        }
                    }

                    // Write output
                    let o_start =
                        (token_offset + t) * num_q_heads * head_dim + qh * head_dim;
                    for d in 0..head_dim {
                        output[o_start + d] = f16::from_f32(out_vec[d]);
                    }
                }
            }
            token_offset += seq_tokens;
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_tokens, num_q_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "MlaAttention-CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_projections(
        kv_lora_rank: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Create projection matrices that act as padded identity:
        // For each kv_head h, dimension d, proj[d][h*head_dim + d] = 1.0
        // This means latent[d] maps to key/value dimension d within each head.
        let out_dim = num_kv_heads * head_dim;
        let mut k_proj = vec![0.0f32; kv_lora_rank * out_dim];
        let mut v_proj = vec![0.0f32; kv_lora_rank * out_dim];

        // Map latent dimension i -> output dimension i (for first head)
        // so that the first head_dim latent dims become the K/V for head 0.
        for i in 0..kv_lora_rank.min(head_dim) {
            // K: identity mapping for head 0
            k_proj[i * out_dim + i] = 1.0;
            // V: identity mapping for head 0, scaled by 2.0 to differentiate
            v_proj[i * out_dim + i] = 1.0;
        }
        (k_proj, v_proj)
    }

    #[test]
    fn mla_name() {
        let config = MlaConfig {
            latent_dim: 512,
            num_kv_heads: 1,
            head_dim: 4,
            q_lora_rank: 256,
            kv_lora_rank: 4,
        };
        let (k, v) = make_identity_projections(4, 1, 4);
        let backend = MlaAttentionBackend::new(config, k, v);
        assert_eq!(backend.name(), "MlaAttention-CPU");
    }

    #[test]
    fn mla_empty_batch() {
        let config = MlaConfig {
            latent_dim: 512,
            num_kv_heads: 1,
            head_dim: 4,
            q_lora_rank: 256,
            kv_lora_rank: 4,
        };
        let (k, v) = make_identity_projections(4, 1, 4);
        let backend = MlaAttentionBackend::new(config, k, v);

        let query = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 1, 4],
        };
        let latent_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 16, 1, 4],
        };
        let value_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 16, 1, 4],
        };
        let block_tables = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 0],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: Vec::new(),
            shape: vec![0],
        };
        let out = backend
            .forward(
                &query,
                &latent_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                0,
                1.0,
            )
            .unwrap();
        assert!(out.data.is_empty());
    }

    #[test]
    fn mla_rejects_wrong_query_dims() {
        let config = MlaConfig {
            latent_dim: 512,
            num_kv_heads: 1,
            head_dim: 4,
            q_lora_rank: 256,
            kv_lora_rank: 4,
        };
        let (k, v) = make_identity_projections(4, 1, 4);
        let backend = MlaAttentionBackend::new(config, k, v);

        let query = GpuBuffer {
            data: vec![f16::ZERO; 16],
            shape: vec![4, 4],
        };
        let kc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 4],
        };
        let vc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 4],
        };
        let bt = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![1],
            shape: vec![1],
        };
        assert!(backend.forward(&query, &kc, &vc, &bt, &cl, 1, 1.0).is_err());
    }

    #[test]
    fn mla_single_token_attention() {
        // Single sequence, single token query, context length 2, 1 head, head_dim=4, kv_lora_rank=4
        let head_dim = 4;
        let kv_lora_rank = 4;
        let num_kv_heads = 1;
        let block_size = 16;

        let config = MlaConfig {
            latent_dim: 512,
            num_kv_heads,
            head_dim,
            q_lora_rank: 256,
            kv_lora_rank,
        };

        // Identity projections: latent passes through directly as K and V
        let (k_proj, v_proj) = make_identity_projections(kv_lora_rank, num_kv_heads, head_dim);
        let backend = MlaAttentionBackend::new(config, k_proj, v_proj);

        // Query: [1, 1, 4] -- single token, 1 head, head_dim=4
        let query = GpuBuffer {
            data: vec![
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
            ],
            shape: vec![1, 1, head_dim],
        };

        // Latent cache: [1 block, block_size=16, 1, kv_lora_rank=4]
        // Fill 2 positions with known latents
        let mut latent_data = vec![f16::ZERO; block_size * 1 * kv_lora_rank];
        // Position 0 latent: [1, 0, 0, 0] -> K=[1,0,0,0], V=[1,0,0,0]
        latent_data[0] = f16::from_f32(1.0);
        // Position 1 latent: [0, 1, 0, 0] -> K=[0,1,0,0], V=[0,1,0,0]
        latent_data[kv_lora_rank + 1] = f16::from_f32(1.0);

        let latent_cache = GpuBuffer {
            data: latent_data,
            shape: vec![1, block_size, 1, kv_lora_rank],
        };
        let value_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, block_size, 1, kv_lora_rank],
        };

        // Block table: seq 0 uses physical block 0
        let block_tables = GpuBuffer {
            data: vec![0i32],
            shape: vec![1, 1],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: vec![2],
            shape: vec![1],
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = backend
            .forward(
                &query,
                &latent_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                2,
                scale,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, head_dim]);
        assert_eq!(out.data.len(), head_dim);

        // Query=[1,0,0,0], K0=[1,0,0,0], K1=[0,1,0,0]
        // score0 = 1.0 * scale, score1 = 0.0 * scale
        // After softmax: w0 = exp(scale)/Z, w1 = exp(0)/Z
        // Output = w0*V0 + w1*V1 = w0*[1,0,0,0] + w1*[0,1,0,0]
        // = [w0, w1, 0, 0]
        let s0 = (1.0f32 * scale).exp();
        let s1 = (0.0f32).exp(); // = 1.0
        let z = s0 + s1;
        let expected_0 = s0 / z;
        let expected_1 = s1 / z;

        let tol = 0.01;
        assert!((out.data[0].to_f32() - expected_0).abs() < tol);
        assert!((out.data[1].to_f32() - expected_1).abs() < tol);
        assert!((out.data[2].to_f32()).abs() < tol);
        assert!((out.data[3].to_f32()).abs() < tol);
    }

    #[test]
    fn mla_gqa_multiple_query_heads() {
        // Test GQA: 4 query heads, 1 KV head (gqa_ratio=4)
        let head_dim = 2;
        let kv_lora_rank = 2;
        let num_kv_heads = 1;
        let num_q_heads = 4;
        let block_size = 4;

        let config = MlaConfig {
            latent_dim: 64,
            num_kv_heads,
            head_dim,
            q_lora_rank: 32,
            kv_lora_rank,
        };

        let (k_proj, v_proj) = make_identity_projections(kv_lora_rank, num_kv_heads, head_dim);
        let backend = MlaAttentionBackend::new(config, k_proj, v_proj);

        // Query: [1, 4, 2] -- 1 token, 4 heads
        let query = GpuBuffer {
            data: vec![
                // Head 0
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                // Head 1
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                // Head 2
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                // Head 3
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
            ],
            shape: vec![1, num_q_heads, head_dim],
        };

        // Single position latent: [1, 0] -> K=[1,0], V=[1,0]
        let mut latent_data = vec![f16::ZERO; block_size * 1 * kv_lora_rank];
        latent_data[0] = f16::from_f32(1.0);

        let latent_cache = GpuBuffer {
            data: latent_data,
            shape: vec![1, block_size, 1, kv_lora_rank],
        };
        let value_cache = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, block_size, 1, kv_lora_rank],
        };
        let block_tables = GpuBuffer {
            data: vec![0i32],
            shape: vec![1, 1],
        };
        let context_lens: GpuBuffer<i32> = GpuBuffer {
            data: vec![1],
            shape: vec![1],
        };

        let out = backend
            .forward(
                &query,
                &latent_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                1,
                1.0,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, num_q_heads, head_dim]);
        // With single context position, softmax weight is 1.0 for it.
        // All heads share the same KV head, so output = V = [1, 0] for all heads.
        let tol = 0.01;
        for h in 0..num_q_heads {
            assert!(
                (out.data[h * head_dim].to_f32() - 1.0).abs() < tol,
                "head {h} dim 0"
            );
            assert!(
                (out.data[h * head_dim + 1].to_f32()).abs() < tol,
                "head {h} dim 1"
            );
        }
    }

    #[test]
    fn project_latent_correctness() {
        let config = MlaConfig {
            latent_dim: 64,
            num_kv_heads: 2,
            head_dim: 3,
            q_lora_rank: 32,
            kv_lora_rank: 2,
        };
        let out_dim = 2 * 3; // num_kv_heads * head_dim = 6

        // proj: [2, 6] -- row-major
        // Row 0: [1, 2, 3, 4, 5, 6]
        // Row 1: [7, 8, 9, 10, 11, 12]
        let proj: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let (_, v_proj) = (proj.clone(), proj.clone());

        let backend = MlaAttentionBackend::new(config, proj, v_proj);

        let latent = vec![1.0f32, 2.0];
        let mut out = vec![0.0f32; out_dim];
        backend.project_latent(&latent, &backend.k_down_proj, &mut out);

        // out[j] = sum_i proj[i*6+j] * latent[i]
        // out[0] = 1*1 + 7*2 = 15
        // out[1] = 2*1 + 8*2 = 18
        // out[2] = 3*1 + 9*2 = 21
        // out[3] = 4*1 + 10*2 = 24
        // out[4] = 5*1 + 11*2 = 27
        // out[5] = 6*1 + 12*2 = 30
        assert_eq!(out, vec![15.0, 18.0, 21.0, 24.0, 27.0, 30.0]);
    }
}
