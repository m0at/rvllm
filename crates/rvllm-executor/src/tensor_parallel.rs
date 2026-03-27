//! Tensor parallelism primitives for multi-GPU inference.
//!
//! Provides column-parallel and row-parallel linear layer abstractions that
//! shard weight matrices across GPUs and use NCCL AllReduce to synchronize
//! partial results.
//!
//! In tensor parallelism:
//! - **Column-parallel linear** (QKV projections, gate/up): each rank holds
//!   a vertical slice of the weight matrix `W[:, shard]`. The forward pass
//!   computes `Y_shard = X @ W_shard` independently -- no communication needed
//!   on the output (results are concatenated logically).
//! - **Row-parallel linear** (output projection, down_proj): each rank holds
//!   a horizontal slice `W[shard, :]`. The forward pass computes a partial
//!   result `Y_partial = X_shard @ W_shard`, then AllReduce(Sum) across ranks
//!   to produce the full output.

use rvllm_core::prelude::{LLMError, Result};

/// Parallelism strategy for a linear layer in tensor parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStyle {
    /// Weight is split along the output (column) dimension.
    /// No communication on forward output -- results are kept sharded.
    ColumnParallel,
    /// Weight is split along the input (row) dimension.
    /// AllReduce(Sum) on the forward output to combine partial results.
    RowParallel,
    /// Weight is replicated across all ranks (embeddings, norms).
    Replicated,
}

/// Configuration for a tensor-parallel group.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of GPUs (ranks) in the tensor-parallel group.
    pub tp_size: usize,
    /// This rank's index within the group.
    pub rank: usize,
}

impl TensorParallelConfig {
    /// Create a new tensor-parallel config.
    pub fn new(tp_size: usize, rank: usize) -> Result<Self> {
        if tp_size == 0 {
            return Err(LLMError::ConfigError(
                "tensor_parallel_size must be >= 1".into(),
            ));
        }
        if rank >= tp_size {
            return Err(LLMError::ConfigError(format!(
                "rank {} >= tensor_parallel_size {}", rank, tp_size
            )));
        }
        Ok(Self { tp_size, rank })
    }

    /// Whether this is actually a multi-GPU config (tp > 1).
    pub fn is_parallel(&self) -> bool {
        self.tp_size > 1
    }

    /// Compute the shard size for a dimension of the given total size.
    pub fn shard_size(&self, total: usize) -> Result<usize> {
        if total % self.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "dimension {} not divisible by tp_size {}", total, self.tp_size
            )));
        }
        Ok(total / self.tp_size)
    }
}

/// Metadata for a column-parallel linear layer.
///
/// Used for QKV projections, gate_proj, and up_proj. The weight matrix
/// `[out_features, in_features]` is sharded along dim 0 (out_features)
/// so each rank holds `[out_features / tp_size, in_features]`.
#[derive(Debug, Clone)]
pub struct ColumnParallelLinear {
    /// Full (un-sharded) output dimension.
    pub full_out_features: usize,
    /// Input dimension (same on all ranks).
    pub in_features: usize,
    /// Whether to gather output across ranks (false = keep sharded for fused ops).
    pub gather_output: bool,
    /// TP configuration.
    pub tp: TensorParallelConfig,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear descriptor.
    pub fn new(
        in_features: usize,
        out_features: usize,
        gather_output: bool,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        if out_features % tp.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "out_features {} not divisible by tp_size {}",
                out_features, tp.tp_size
            )));
        }
        Ok(Self {
            full_out_features: out_features,
            in_features,
            gather_output,
            tp,
        })
    }

    /// Output dimension on this rank.
    pub fn shard_out_features(&self) -> usize {
        self.full_out_features / self.tp.tp_size
    }

    /// Compute output shape for a given batch size.
    ///
    /// If `gather_output` is true, returns the full output dimension.
    /// Otherwise returns the per-rank sharded dimension.
    pub fn output_shape(&self, batch_size: usize) -> (usize, usize) {
        if self.gather_output {
            (batch_size, self.full_out_features)
        } else {
            (batch_size, self.shard_out_features())
        }
    }

    /// Forward pass: `Y_shard = X @ W_shard^T`
    ///
    /// Each rank computes independently with its local weight shard.
    /// Input `x` is `[batch_size, in_features]`.
    /// Weight shard is `[shard_out_features, in_features]`.
    /// Output is `[batch_size, shard_out_features]`.
    ///
    /// No inter-GPU communication needed (unless `gather_output` is true,
    /// which requires AllGather).
    pub fn forward(&self, x: &[f32], weight_shard: &[f32]) -> Result<Vec<f32>> {
        let in_f = self.in_features;
        let out_f = self.shard_out_features();

        if x.len() % in_f != 0 {
            return Err(LLMError::ModelError(format!(
                "column-parallel forward: input length {} not divisible by in_features {}",
                x.len(), in_f
            )));
        }
        let batch_size = x.len() / in_f;

        if weight_shard.len() != out_f * in_f {
            return Err(LLMError::ModelError(format!(
                "column-parallel forward: weight_shard length {} != {} * {} = {}",
                weight_shard.len(), out_f, in_f, out_f * in_f
            )));
        }

        // Y = X @ W^T  where W is [out_f, in_f], so W^T is [in_f, out_f]
        let mut output = vec![0.0f32; batch_size * out_f];
        for b in 0..batch_size {
            for o in 0..out_f {
                let mut sum = 0.0f32;
                for i in 0..in_f {
                    sum += x[b * in_f + i] * weight_shard[o * in_f + i];
                }
                output[b * out_f + o] = sum;
            }
        }

        Ok(output)
    }
}

/// Metadata for a row-parallel linear layer.
///
/// Used for output projection (o_proj) and down_proj. The weight matrix
/// `[out_features, in_features]` is sharded along dim 1 (in_features)
/// so each rank holds `[out_features, in_features / tp_size]`.
///
/// After the local matmul, an AllReduce(Sum) combines partial results.
#[derive(Debug, Clone)]
pub struct RowParallelLinear {
    /// Output dimension (same on all ranks).
    pub out_features: usize,
    /// Full (un-sharded) input dimension.
    pub full_in_features: usize,
    /// Whether input is already sharded across ranks.
    pub input_is_parallel: bool,
    /// TP configuration.
    pub tp: TensorParallelConfig,
}

impl RowParallelLinear {
    /// Create a new row-parallel linear descriptor.
    pub fn new(
        in_features: usize,
        out_features: usize,
        input_is_parallel: bool,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        if in_features % tp.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "in_features {} not divisible by tp_size {}",
                in_features, tp.tp_size
            )));
        }
        Ok(Self {
            out_features,
            full_in_features: in_features,
            input_is_parallel,
            tp,
        })
    }

    /// Input dimension on this rank (sharded).
    pub fn shard_in_features(&self) -> usize {
        self.full_in_features / self.tp.tp_size
    }

    /// Forward pass: compute partial result `Y_partial = X_shard @ W_shard^T`.
    ///
    /// Input `x_shard` is `[batch_size, shard_in_features]`.
    /// Weight shard is `[out_features, shard_in_features]`.
    /// Returns partial output `[batch_size, out_features]`.
    ///
    /// The caller must AllReduce(Sum) the partial outputs from all ranks
    /// to get the final result. This is NOT done here to allow fusing the
    /// AllReduce with other operations (e.g., residual add).
    pub fn forward_partial(&self, x_shard: &[f32], weight_shard: &[f32]) -> Result<Vec<f32>> {
        let in_f = self.shard_in_features();
        let out_f = self.out_features;

        if x_shard.len() % in_f != 0 {
            return Err(LLMError::ModelError(format!(
                "row-parallel forward: input length {} not divisible by shard_in_features {}",
                x_shard.len(), in_f
            )));
        }
        let batch_size = x_shard.len() / in_f;

        if weight_shard.len() != out_f * in_f {
            return Err(LLMError::ModelError(format!(
                "row-parallel forward: weight_shard length {} != {} * {} = {}",
                weight_shard.len(), out_f, in_f, out_f * in_f
            )));
        }

        // Y_partial = X_shard @ W_shard^T
        let mut output = vec![0.0f32; batch_size * out_f];
        for b in 0..batch_size {
            for o in 0..out_f {
                let mut sum = 0.0f32;
                for i in 0..in_f {
                    sum += x_shard[b * in_f + i] * weight_shard[o * in_f + i];
                }
                output[b * out_f + o] = sum;
            }
        }

        Ok(output)
    }

    /// Reduce partial outputs from all ranks by summing element-wise.
    ///
    /// This is the CPU mock of AllReduce(Sum). In production, this would
    /// be an NCCL AllReduce call on GPU buffers.
    pub fn reduce_partial_outputs(partials: &[Vec<f32>]) -> Result<Vec<f32>> {
        if partials.is_empty() {
            return Err(LLMError::ModelError(
                "reduce_partial_outputs: no partials provided".into(),
            ));
        }

        let len = partials[0].len();
        for (i, p) in partials.iter().enumerate() {
            if p.len() != len {
                return Err(LLMError::ModelError(format!(
                    "reduce_partial_outputs: partial {} has length {} != expected {}",
                    i,
                    p.len(),
                    len
                )));
            }
        }

        let mut result = vec![0.0f32; len];
        for partial in partials {
            for (r, &p) in result.iter_mut().zip(partial.iter()) {
                *r += p;
            }
        }

        Ok(result)
    }
}

/// Describes the full tensor-parallel layout of a transformer layer.
///
/// This captures where AllReduce calls are needed in a standard
/// transformer forward pass:
/// 1. QKV projection (column-parallel) -- no comm
/// 2. Attention computation -- local
/// 3. Output projection (row-parallel) -- **AllReduce here**
/// 4. Gate/Up projection (column-parallel) -- no comm
/// 5. Down projection (row-parallel) -- **AllReduce here**
#[derive(Debug, Clone)]
pub struct TransformerLayerParallel {
    /// QKV projection: column-parallel, output kept sharded.
    pub qkv_proj: ColumnParallelLinear,
    /// Output projection: row-parallel, requires AllReduce.
    pub o_proj: RowParallelLinear,
    /// Gate projection (SwiGLU first half): column-parallel.
    pub gate_proj: ColumnParallelLinear,
    /// Up projection (SwiGLU second half): column-parallel.
    pub up_proj: ColumnParallelLinear,
    /// Down projection: row-parallel, requires AllReduce.
    pub down_proj: RowParallelLinear,
}

impl TransformerLayerParallel {
    /// Construct the parallel layout for a standard LLaMA-style transformer layer.
    ///
    /// - `hidden_size`: model hidden dimension (e.g. 4096)
    /// - `num_heads`: number of attention heads (e.g. 32)
    /// - `head_dim`: dimension per head (e.g. 128)
    /// - `intermediate_size`: MLP intermediate dimension (e.g. 11008)
    /// - `num_kv_heads`: number of KV heads for GQA (e.g. 8)
    pub fn new_llama(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_kv_heads: usize,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let qkv_out = (num_heads + 2 * num_kv_heads) * head_dim;

        // Validate divisibility
        if num_heads % tp.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "num_heads {} not divisible by tp_size {}",
                num_heads, tp.tp_size
            )));
        }
        if num_kv_heads % tp.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "num_kv_heads {} not divisible by tp_size {}",
                num_kv_heads, tp.tp_size
            )));
        }
        if intermediate_size % tp.tp_size != 0 {
            return Err(LLMError::ConfigError(format!(
                "intermediate_size {} not divisible by tp_size {}",
                intermediate_size, tp.tp_size
            )));
        }

        let qkv_proj = ColumnParallelLinear::new(hidden_size, qkv_out, false, tp.clone())?;
        let o_proj = RowParallelLinear::new(
            num_heads * head_dim,
            hidden_size,
            true,
            tp.clone(),
        )?;
        let gate_proj =
            ColumnParallelLinear::new(hidden_size, intermediate_size, false, tp.clone())?;
        let up_proj =
            ColumnParallelLinear::new(hidden_size, intermediate_size, false, tp.clone())?;
        let down_proj =
            RowParallelLinear::new(intermediate_size, hidden_size, true, tp)?;

        Ok(Self {
            qkv_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Classify a weight name into a parallelism style.
///
/// Mirrors the logic in `rvllm-model-loader::shard::classify_shard_dim`.
pub fn classify_parallel_style(name: &str) -> ParallelStyle {
    // Column-parallel: output dimension split
    if name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains("attn.q")
        || name.contains("attn.k")
        || name.contains("attn.v")
        || name.contains("ffn.gate")
        || name.contains("ffn.up")
        || name.contains("c_attn")
        || name.contains("c_fc")
    {
        return ParallelStyle::ColumnParallel;
    }

    // Row-parallel: input dimension split (AllReduce after)
    if name.contains("o_proj")
        || name.contains("down_proj")
        || name.contains("attn.o")
        || name.contains("ffn.down")
        || name.contains("c_proj")
    {
        return ParallelStyle::RowParallel;
    }

    ParallelStyle::Replicated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp_config_valid() {
        let tp = TensorParallelConfig::new(4, 2).unwrap();
        assert_eq!(tp.tp_size, 4);
        assert_eq!(tp.rank, 2);
        assert!(tp.is_parallel());
    }

    #[test]
    fn tp_config_single() {
        let tp = TensorParallelConfig::new(1, 0).unwrap();
        assert!(!tp.is_parallel());
    }

    #[test]
    fn tp_config_rank_out_of_bounds() {
        assert!(TensorParallelConfig::new(2, 2).is_err());
    }

    #[test]
    fn tp_config_zero_size() {
        assert!(TensorParallelConfig::new(0, 0).is_err());
    }

    #[test]
    fn shard_size_computation() {
        let tp = TensorParallelConfig::new(4, 0).unwrap();
        assert_eq!(tp.shard_size(128).unwrap(), 32);
        assert!(tp.shard_size(13).is_err());
    }

    #[test]
    fn column_parallel_basic() {
        let tp = TensorParallelConfig::new(2, 0).unwrap();
        let col = ColumnParallelLinear::new(4, 8, false, tp).unwrap();
        assert_eq!(col.shard_out_features(), 4);
        assert_eq!(col.output_shape(2), (2, 4));
    }

    #[test]
    fn column_parallel_gathered() {
        let tp = TensorParallelConfig::new(2, 0).unwrap();
        let col = ColumnParallelLinear::new(4, 8, true, tp).unwrap();
        assert_eq!(col.output_shape(3), (3, 8));
    }

    #[test]
    fn column_parallel_indivisible() {
        let tp = TensorParallelConfig::new(3, 0).unwrap();
        assert!(ColumnParallelLinear::new(4, 8, false, tp).is_err());
    }

    #[test]
    fn column_parallel_forward() {
        // tp=1, in_features=3, out_features=2
        // W = [[1,0,0],[0,1,0]] (2x3)
        // x = [[1,2,3],[4,5,6]] (2x3)
        // Y = x @ W^T = [[1,2],[4,5]]
        let tp = TensorParallelConfig::new(1, 0).unwrap();
        let col = ColumnParallelLinear::new(3, 2, false, tp).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let y = col.forward(&x, &w).unwrap();
        assert_eq!(y, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn column_parallel_forward_sharded() {
        // tp=2, in_features=4, out_features=4
        // Full W = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        // Rank 0 shard: rows 0..2 = [[1,0,0,0],[0,1,0,0]]
        // Rank 1 shard: rows 2..4 = [[0,0,1,0],[0,0,0,1]]
        // x = [[1,2,3,4]] (1x4)
        // Rank 0: Y_shard = [[1,2]]
        // Rank 1: Y_shard = [[3,4]]
        // Concatenated: [[1,2,3,4]] = x (identity)
        let tp0 = TensorParallelConfig::new(2, 0).unwrap();
        let col0 = ColumnParallelLinear::new(4, 4, false, tp0).unwrap();
        let w0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y0 = col0.forward(&x, &w0).unwrap();
        assert_eq!(y0, vec![1.0, 2.0]);

        let tp1 = TensorParallelConfig::new(2, 1).unwrap();
        let col1 = ColumnParallelLinear::new(4, 4, false, tp1).unwrap();
        let w1 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let y1 = col1.forward(&x, &w1).unwrap();
        assert_eq!(y1, vec![3.0, 4.0]);
    }

    #[test]
    fn row_parallel_basic() {
        let tp = TensorParallelConfig::new(2, 0).unwrap();
        let row = RowParallelLinear::new(8, 4, true, tp).unwrap();
        assert_eq!(row.shard_in_features(), 4);
    }

    #[test]
    fn row_parallel_indivisible() {
        let tp = TensorParallelConfig::new(3, 0).unwrap();
        assert!(RowParallelLinear::new(7, 4, true, tp).is_err());
    }

    #[test]
    fn row_parallel_forward_and_reduce() {
        // tp=2, in_features=4, out_features=2
        // Full W = [[1,1,1,1],[2,2,2,2]] (2x4)
        // Rank 0 shard (cols 0..2): [[1,1],[2,2]]
        // Rank 1 shard (cols 2..4): [[1,1],[2,2]]
        // x = [[1,1,1,1]] -> x_shard0 = [[1,1]], x_shard1 = [[1,1]]
        // Y_partial0 = [[1*1+1*1, 2*1+2*1]] = [[2, 4]]
        // Y_partial1 = [[1*1+1*1, 2*1+2*1]] = [[2, 4]]
        // Y = [[4, 8]]
        let tp0 = TensorParallelConfig::new(2, 0).unwrap();
        let row0 = RowParallelLinear::new(4, 2, true, tp0).unwrap();
        let w0 = vec![1.0, 1.0, 2.0, 2.0]; // [2, 2]
        let x0 = vec![1.0, 1.0]; // [1, 2]
        let y0 = row0.forward_partial(&x0, &w0).unwrap();
        assert_eq!(y0, vec![2.0, 4.0]);

        let tp1 = TensorParallelConfig::new(2, 1).unwrap();
        let row1 = RowParallelLinear::new(4, 2, true, tp1).unwrap();
        let w1 = vec![1.0, 1.0, 2.0, 2.0];
        let x1 = vec![1.0, 1.0];
        let y1 = row1.forward_partial(&x1, &w1).unwrap();
        assert_eq!(y1, vec![2.0, 4.0]);

        let reduced = RowParallelLinear::reduce_partial_outputs(&[y0, y1]).unwrap();
        assert_eq!(reduced, vec![4.0, 8.0]);
    }

    #[test]
    fn reduce_empty_partials() {
        let result = RowParallelLinear::reduce_partial_outputs(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn reduce_mismatched_lengths() {
        let result =
            RowParallelLinear::reduce_partial_outputs(&[vec![1.0, 2.0], vec![3.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn classify_weight_names() {
        assert_eq!(
            classify_parallel_style("layers.0.self_attn.q_proj.weight"),
            ParallelStyle::ColumnParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.self_attn.k_proj.weight"),
            ParallelStyle::ColumnParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.self_attn.v_proj.weight"),
            ParallelStyle::ColumnParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.self_attn.o_proj.weight"),
            ParallelStyle::RowParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.mlp.gate_proj.weight"),
            ParallelStyle::ColumnParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.mlp.up_proj.weight"),
            ParallelStyle::ColumnParallel
        );
        assert_eq!(
            classify_parallel_style("layers.0.mlp.down_proj.weight"),
            ParallelStyle::RowParallel
        );
        assert_eq!(
            classify_parallel_style("model.embed_tokens.weight"),
            ParallelStyle::Replicated
        );
        assert_eq!(
            classify_parallel_style("model.norm.weight"),
            ParallelStyle::Replicated
        );
        assert_eq!(
            classify_parallel_style("lm_head.weight"),
            ParallelStyle::Replicated
        );
    }

    #[test]
    fn transformer_layer_parallel_llama_7b() {
        // LLaMA-7B: hidden=4096, heads=32, head_dim=128, intermediate=11008, kv_heads=32
        let tp = TensorParallelConfig::new(2, 0).unwrap();
        let layer = TransformerLayerParallel::new_llama(
            4096, 32, 128, 11008, 32, tp,
        );
        // 11008 % 2 == 0 -> should succeed
        assert!(layer.is_ok());
        let l = layer.unwrap();
        assert_eq!(l.qkv_proj.shard_out_features(), (32 + 64) * 128 / 2);
        assert_eq!(l.o_proj.shard_in_features(), 32 * 128 / 2);
        assert_eq!(l.gate_proj.shard_out_features(), 11008 / 2);
        assert_eq!(l.down_proj.shard_in_features(), 11008 / 2);
    }

    #[test]
    fn transformer_layer_parallel_llama_70b_gqa() {
        // LLaMA-70B: hidden=8192, heads=64, head_dim=128, intermediate=28672, kv_heads=8
        let tp = TensorParallelConfig::new(8, 3).unwrap();
        let layer = TransformerLayerParallel::new_llama(
            8192, 64, 128, 28672, 8, tp,
        );
        assert!(layer.is_ok());
        let l = layer.unwrap();
        // QKV out = (64 + 2*8) * 128 = 10240
        assert_eq!(l.qkv_proj.shard_out_features(), 10240 / 8);
        assert_eq!(l.o_proj.shard_in_features(), 64 * 128 / 8);
        assert_eq!(l.gate_proj.shard_out_features(), 28672 / 8);
    }

    #[test]
    fn transformer_layer_parallel_bad_divisibility() {
        // 5 heads not divisible by tp=2
        let tp = TensorParallelConfig::new(2, 0).unwrap();
        let result = TransformerLayerParallel::new_llama(
            4096, 5, 128, 11008, 5, tp,
        );
        assert!(result.is_err());
    }

    #[test]
    fn end_to_end_tp2_identity() {
        // Verify that splitting an identity-like forward across 2 ranks
        // and reducing gives the correct result.

        // Layer: in=4, hidden=4, out=4 (identity-ish)
        // Column-parallel: each rank computes half the output features
        // Row-parallel: each rank uses half the input features, then reduce

        // Step 1: Column-parallel (like gate_proj)
        // W_full = I(4x4), sharded along dim 0
        let tp0 = TensorParallelConfig::new(2, 0).unwrap();
        let tp1 = TensorParallelConfig::new(2, 1).unwrap();

        let col0 = ColumnParallelLinear::new(4, 4, false, tp0.clone()).unwrap();
        let col1 = ColumnParallelLinear::new(4, 4, false, tp1.clone()).unwrap();

        // W_shard0 = top half of identity: [[1,0,0,0],[0,1,0,0]]
        // W_shard1 = bottom half: [[0,0,1,0],[0,0,0,1]]
        let w_col0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let w_col1 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let x = vec![10.0, 20.0, 30.0, 40.0]; // [1, 4]

        let h0 = col0.forward(&x, &w_col0).unwrap(); // [1, 2] = [10, 20]
        let h1 = col1.forward(&x, &w_col1).unwrap(); // [1, 2] = [30, 40]
        assert_eq!(h0, vec![10.0, 20.0]);
        assert_eq!(h1, vec![30.0, 40.0]);

        // Step 2: Row-parallel (like down_proj)
        // W_full = I(4x4), sharded along dim 1
        let row0 = RowParallelLinear::new(4, 4, true, tp0).unwrap();
        let row1 = RowParallelLinear::new(4, 4, true, tp1).unwrap();

        // W_shard0 = left half of identity: [[1,0],[0,1],[0,0],[0,0]]
        // W_shard1 = right half: [[0,0],[0,0],[1,0],[0,1]]
        let w_row0 = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let w_row1 = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];

        let y0 = row0.forward_partial(&h0, &w_row0).unwrap(); // partial [1,4]
        let y1 = row1.forward_partial(&h1, &w_row1).unwrap(); // partial [1,4]

        let y = RowParallelLinear::reduce_partial_outputs(&[y0, y1]).unwrap();
        assert_eq!(y, vec![10.0, 20.0, 30.0, 40.0]);
    }
}
