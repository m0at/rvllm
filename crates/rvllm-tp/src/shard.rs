use crate::{LLMError, Result};

/// How to distribute a weight matrix across TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Shard along rows (output dimension). Each rank gets rows [rank*chunk..(rank+1)*chunk].
    /// Used for: O projection, down projection.
    /// Requires all-reduce after GEMM.
    RowParallel,
    /// Shard along columns (input dimension). Each rank gets columns [rank*chunk..(rank+1)*chunk].
    /// Used for: QKV projection, gate+up projection.
    /// Requires all-gather after GEMM (or concat of partial outputs).
    ColumnParallel,
    /// Full copy on every rank. Used for: embeddings, layer norms, lm_head.
    Replicate,
}

/// Standard TP sharding plan for a transformer layer.
pub struct TransformerTpPlan {
    pub qkv_proj: ShardStrategy,
    pub o_proj: ShardStrategy,
    pub gate_up_proj: ShardStrategy,
    pub down_proj: ShardStrategy,
    pub embed: ShardStrategy,
    pub norm: ShardStrategy,
    pub lm_head: ShardStrategy,
}

impl TransformerTpPlan {
    /// The standard Megatron-LM sharding plan used by most LLM serving systems.
    pub fn standard() -> Self {
        Self {
            qkv_proj: ShardStrategy::ColumnParallel,
            o_proj: ShardStrategy::RowParallel,
            gate_up_proj: ShardStrategy::ColumnParallel,
            down_proj: ShardStrategy::RowParallel,
            embed: ShardStrategy::Replicate,
            norm: ShardStrategy::Replicate,
            lm_head: ShardStrategy::Replicate,
        }
    }
}

/// Extract this rank's shard from a full weight tensor.
///
/// `weight` is the raw bytes of the full weight matrix.
/// `shape` is [out_dim, in_dim] (row-major).
/// `dtype_size` is bytes per element (2 for f16).
///
/// Returns the shard bytes for the given rank.
pub fn shard_weight(
    weight: &[u8],
    shape: &[usize],
    dtype_size: usize,
    strategy: ShardStrategy,
    rank: usize,
    world_size: usize,
) -> Result<Vec<u8>> {
    if shape.len() != 2 {
        return Err(LLMError::ConfigError(format!(
            "shard_weight expects 2D shape, got {}D",
            shape.len()
        )));
    }

    let (out_dim, in_dim) = (shape[0], shape[1]);
    let expected_bytes = out_dim * in_dim * dtype_size;

    if weight.len() < expected_bytes {
        return Err(LLMError::MemoryError(format!(
            "shard_weight: weight buffer too small, need {} got {}",
            expected_bytes,
            weight.len()
        )));
    }

    match strategy {
        ShardStrategy::Replicate => Ok(weight[..expected_bytes].to_vec()),

        ShardStrategy::RowParallel => {
            // Shard along output dim (rows). Each rank gets out_dim/world_size rows.
            if out_dim % world_size != 0 {
                return Err(LLMError::ConfigError(format!(
                    "out_dim {} not divisible by world_size {}",
                    out_dim, world_size
                )));
            }
            let rows_per_rank = out_dim / world_size;
            let row_bytes = in_dim * dtype_size;
            let start = rank * rows_per_rank * row_bytes;
            let end = start + rows_per_rank * row_bytes;
            Ok(weight[start..end].to_vec())
        }

        ShardStrategy::ColumnParallel => {
            // Shard along input dim (columns). Each rank gets in_dim/world_size columns.
            // Data is row-major, so we need to extract strided columns.
            if in_dim % world_size != 0 {
                return Err(LLMError::ConfigError(format!(
                    "in_dim {} not divisible by world_size {}",
                    in_dim, world_size
                )));
            }
            let cols_per_rank = in_dim / world_size;
            let col_bytes = cols_per_rank * dtype_size;
            let row_bytes = in_dim * dtype_size;
            let col_start = rank * col_bytes;

            let mut shard = Vec::with_capacity(out_dim * col_bytes);
            for row in 0..out_dim {
                let row_offset = row * row_bytes;
                let start = row_offset + col_start;
                shard.extend_from_slice(&weight[start..start + col_bytes]);
            }
            Ok(shard)
        }
    }
}

/// Compute the shard shape after sharding.
pub fn shard_shape(
    shape: &[usize; 2],
    strategy: ShardStrategy,
    world_size: usize,
) -> [usize; 2] {
    match strategy {
        ShardStrategy::Replicate => *shape,
        ShardStrategy::RowParallel => [shape[0] / world_size, shape[1]],
        ShardStrategy::ColumnParallel => [shape[0], shape[1] / world_size],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_weight(out_dim: usize, in_dim: usize) -> Vec<u8> {
        // Fill with sequential bytes for easy verification. dtype_size=1 for simplicity.
        (0..out_dim * in_dim).map(|i| i as u8).collect()
    }

    #[test]
    fn replicate_returns_full_copy() {
        let w = make_weight(4, 4);
        let shard = shard_weight(&w, &[4, 4], 1, ShardStrategy::Replicate, 0, 2).unwrap();
        assert_eq!(shard, w);
    }

    #[test]
    fn row_parallel_shard() {
        // 4x4 matrix, world_size=2
        // Rank 0 gets rows 0-1, rank 1 gets rows 2-3
        let w = make_weight(4, 4);
        let s0 = shard_weight(&w, &[4, 4], 1, ShardStrategy::RowParallel, 0, 2).unwrap();
        let s1 = shard_weight(&w, &[4, 4], 1, ShardStrategy::RowParallel, 1, 2).unwrap();
        assert_eq!(s0.len(), 8); // 2 rows * 4 cols
        assert_eq!(s1.len(), 8);
        assert_eq!(&s0, &w[0..8]);
        assert_eq!(&s1, &w[8..16]);
    }

    #[test]
    fn column_parallel_shard() {
        // 4x4 matrix, world_size=2
        // Rank 0 gets cols 0-1, rank 1 gets cols 2-3
        let w = make_weight(4, 4);
        let s0 = shard_weight(&w, &[4, 4], 1, ShardStrategy::ColumnParallel, 0, 2).unwrap();
        let s1 = shard_weight(&w, &[4, 4], 1, ShardStrategy::ColumnParallel, 1, 2).unwrap();
        assert_eq!(s0.len(), 8); // 4 rows * 2 cols
        assert_eq!(s1.len(), 8);
        // Row 0: [0,1,2,3] -> rank0=[0,1], rank1=[2,3]
        // Row 1: [4,5,6,7] -> rank0=[4,5], rank1=[6,7]
        assert_eq!(s0, vec![0, 1, 4, 5, 8, 9, 12, 13]);
        assert_eq!(s1, vec![2, 3, 6, 7, 10, 11, 14, 15]);
    }

    #[test]
    fn row_parallel_f16() {
        // 4x2 matrix with dtype_size=2 (f16), world_size=2
        // Total: 4*2*2 = 16 bytes
        let w: Vec<u8> = (0..16).collect();
        let s0 = shard_weight(&w, &[4, 2], 2, ShardStrategy::RowParallel, 0, 2).unwrap();
        let s1 = shard_weight(&w, &[4, 2], 2, ShardStrategy::RowParallel, 1, 2).unwrap();
        assert_eq!(s0.len(), 8); // 2 rows * 2 cols * 2 bytes
        assert_eq!(&s0, &w[0..8]);
        assert_eq!(&s1, &w[8..16]);
    }

    #[test]
    fn indivisible_dim_errors() {
        let w = make_weight(5, 4);
        assert!(shard_weight(&w, &[5, 4], 1, ShardStrategy::RowParallel, 0, 2).is_err());
        let w2 = make_weight(4, 5);
        assert!(shard_weight(&w2, &[4, 5], 1, ShardStrategy::ColumnParallel, 0, 2).is_err());
    }

    #[test]
    fn wrong_shape_rank_errors() {
        let w = make_weight(4, 4);
        assert!(shard_weight(&w, &[4], 1, ShardStrategy::RowParallel, 0, 2).is_err());
    }

    #[test]
    fn shard_shape_row() {
        assert_eq!(shard_shape(&[8, 4], ShardStrategy::RowParallel, 2), [4, 4]);
    }

    #[test]
    fn shard_shape_col() {
        assert_eq!(
            shard_shape(&[8, 4], ShardStrategy::ColumnParallel, 4),
            [8, 1]
        );
    }

    #[test]
    fn shard_shape_replicate() {
        assert_eq!(shard_shape(&[8, 4], ShardStrategy::Replicate, 4), [8, 4]);
    }

    #[test]
    fn standard_tp_plan() {
        let plan = TransformerTpPlan::standard();
        assert_eq!(plan.qkv_proj, ShardStrategy::ColumnParallel);
        assert_eq!(plan.o_proj, ShardStrategy::RowParallel);
        assert_eq!(plan.gate_up_proj, ShardStrategy::ColumnParallel);
        assert_eq!(plan.down_proj, ShardStrategy::RowParallel);
        assert_eq!(plan.embed, ShardStrategy::Replicate);
        assert_eq!(plan.norm, ShardStrategy::Replicate);
    }
}
