use std::sync::Arc;

use crate::comm::TpCommunicator;
use crate::shard::ShardStrategy;
use crate::Result;

/// A linear layer sharded across TP ranks.
///
/// Each rank holds its portion of the weight matrix. After the local GEMM,
/// a collective operation combines partial results:
/// - RowParallel: all-reduce sum (each rank computed partial output, sum them)
/// - ColumnParallel: all-gather (each rank computed a slice of the output)
/// - Replicate: no communication needed
pub struct TpLinearLayer {
    pub strategy: ShardStrategy,
    /// This rank's weight shard as raw f16 bytes.
    /// Shape depends on strategy:
    ///   RowParallel:    [out_dim / world_size, in_dim]
    ///   ColumnParallel: [out_dim, in_dim / world_size]
    ///   Replicate:      [out_dim, in_dim]
    pub weight_shard: Vec<u8>,
    /// Shard shape: [shard_out_dim, shard_in_dim]
    pub shard_shape: [usize; 2],
    pub communicator: Arc<dyn TpCommunicator>,
}

impl TpLinearLayer {
    pub fn new(
        strategy: ShardStrategy,
        weight_shard: Vec<u8>,
        shard_shape: [usize; 2],
        communicator: Arc<dyn TpCommunicator>,
    ) -> Self {
        Self {
            strategy,
            weight_shard,
            shard_shape,
            communicator,
        }
    }

    /// Forward pass: local GEMM + collective communication.
    ///
    /// `input` is [batch_size, in_dim] as f16 bytes.
    /// Returns [batch_size, out_dim] as f16 bytes (after communication).
    ///
    /// The caller is responsible for the actual GEMM kernel launch.
    /// This method handles only the communication pattern:
    ///
    /// For RowParallel:
    ///   1. Caller does GEMM: output_partial = input @ weight_shard^T
    ///      output_partial shape: [batch, out_dim/world_size] ... wait, no.
    ///      RowParallel shards rows: weight_shard is [out_dim/ws, in_dim]
    ///      So output_partial is [batch, out_dim/ws] -- a slice of the output.
    ///      Need all-gather to get full [batch, out_dim]? No -- standard Megatron:
    ///      RowParallel means input is partitioned, each rank has full out_dim.
    ///      Let me be precise:
    ///
    /// Megatron RowParallel (standard):
    ///   - Weight is sharded along in_dim: W_shard is [out_dim, in_dim/ws]
    ///   - Input is the column-parallel output from previous layer (already partitioned)
    ///   - Each rank computes: Y_partial = X_shard @ W_shard^T = [batch, out_dim]
    ///   - All-reduce sum to get Y = sum of Y_partial across ranks
    ///
    /// Megatron ColumnParallel:
    ///   - Weight is sharded along out_dim: W_shard is [out_dim/ws, in_dim]
    ///   - Input is replicated across all ranks
    ///   - Each rank computes: Y_shard = X @ W_shard^T = [batch, out_dim/ws]
    ///   - Outputs are concatenated (or used directly by the next RowParallel layer)
    ///
    /// So the communication after each layer:
    ///   ColumnParallel -> no comm needed if followed by RowParallel
    ///   RowParallel -> all-reduce sum
    ///
    /// This is the standard f-then-g approach from Megatron-LM.
    pub fn post_gemm_communicate(
        &self,
        gemm_output: &mut [u8],
        element_count: usize,
    ) -> Result<CommunicationResult> {
        if !self.communicator.world_size() > 1 {
            return Ok(CommunicationResult::InPlace);
        }

        match self.strategy {
            ShardStrategy::RowParallel => {
                // All-reduce sum: each rank has a partial sum, combine them
                self.communicator
                    .all_reduce_sum_f16(gemm_output, element_count)?;
                Ok(CommunicationResult::InPlace)
            }
            ShardStrategy::ColumnParallel => {
                // No communication needed here if the next layer is RowParallel
                // (standard Megatron pairing). The partial output is used directly.
                Ok(CommunicationResult::InPlace)
            }
            ShardStrategy::Replicate => {
                // No communication needed
                Ok(CommunicationResult::InPlace)
            }
        }
    }

    /// For the case where ColumnParallel output needs to be gathered
    /// (e.g., before a non-TP layer like a norm), use this explicitly.
    pub fn all_gather_column_output(
        &self,
        local_output: &[u8],
        gathered_output: &mut [u8],
        local_count: usize,
    ) -> Result<()> {
        self.communicator
            .all_gather_f16(local_output, gathered_output, local_count)
    }
}

/// Result of post-GEMM communication.
pub enum CommunicationResult {
    /// Output was modified in-place (all-reduce) or unchanged.
    InPlace,
}

/// Describes the full TP layout of a transformer attention block.
/// Maps layer weight names to their sharding + communication patterns.
pub struct TpAttentionLayout {
    pub qkv: TpLinearLayer,
    pub o_proj: TpLinearLayer,
}

/// Describes the full TP layout of a transformer MLP block.
pub struct TpMlpLayout {
    pub gate_up: TpLinearLayer,
    pub down: TpLinearLayer,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nccl::NcclTpCommunicator;

    #[test]
    fn single_rank_row_parallel_communicate() {
        let comm = Arc::new(NcclTpCommunicator::single_rank().unwrap());
        let layer = TpLinearLayer::new(
            ShardStrategy::RowParallel,
            vec![0u8; 16],
            [2, 4],
            comm,
        );
        let mut output = vec![1u8, 0, 2, 0];
        layer.post_gemm_communicate(&mut output, 2).unwrap();
        assert_eq!(output, vec![1, 0, 2, 0]);
    }

    #[test]
    fn single_rank_column_parallel_noop() {
        let comm = Arc::new(NcclTpCommunicator::single_rank().unwrap());
        let layer = TpLinearLayer::new(
            ShardStrategy::ColumnParallel,
            vec![0u8; 16],
            [4, 2],
            comm,
        );
        let mut output = vec![5u8, 6, 7, 8];
        layer.post_gemm_communicate(&mut output, 2).unwrap();
        assert_eq!(output, vec![5, 6, 7, 8]);
    }
}
