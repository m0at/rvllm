use std::sync::Arc;

use rvllm_gpu::nccl::{NcclComm, NcclDataType, NcclReduceOp, NcclUniqueId};

use crate::comm::TpCommunicator;
use crate::config::TpConfig;
use crate::Result;

/// Tensor-parallel communicator backed by the NCCL bindings in rvllm-gpu.
pub struct NcclTpCommunicator {
    comm: Arc<NcclComm>,
    config: TpConfig,
}

impl NcclTpCommunicator {
    /// Create a communicator for a single rank within a TP group.
    /// All ranks must call this with the same `unique_id`.
    pub fn new(unique_id: &NcclUniqueId, config: TpConfig) -> Result<Self> {
        let comm = NcclComm::new(unique_id, config.world_size, config.rank)?;
        Ok(Self {
            comm: Arc::new(comm),
            config,
        })
    }

    /// Create a single-rank communicator (no actual communication needed).
    pub fn single_rank() -> Result<Self> {
        let uid = NcclUniqueId::new();
        let config = TpConfig::single_gpu();
        Self::new(&uid, config)
    }

    pub fn config(&self) -> &TpConfig {
        &self.config
    }
}

impl TpCommunicator for NcclTpCommunicator {
    fn all_reduce_sum_f16(&self, buf: &mut [u8], count: usize) -> Result<()> {
        self.comm
            .all_reduce_in_place(buf, count, NcclDataType::Float16, NcclReduceOp::Sum)
    }

    fn all_gather_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        send_count: usize,
    ) -> Result<()> {
        self.comm
            .all_gather(input, output, send_count, NcclDataType::Float16)
    }

    fn reduce_scatter_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        recv_count: usize,
    ) -> Result<()> {
        self.comm.reduce_scatter(
            input,
            output,
            recv_count,
            NcclDataType::Float16,
            NcclReduceOp::Sum,
        )
    }

    fn barrier(&self) -> Result<()> {
        // NCCL doesn't have a native barrier. Use a zero-element all-reduce as a sync point.
        let mut dummy = [0u8; 0];
        self.comm
            .all_reduce_in_place(&mut dummy, 0, NcclDataType::Float16, NcclReduceOp::Sum)
    }

    fn world_size(&self) -> usize {
        self.config.world_size
    }

    fn rank(&self) -> usize {
        self.config.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_communicator() {
        let comm = NcclTpCommunicator::single_rank().unwrap();
        assert_eq!(comm.world_size(), 1);
        assert_eq!(comm.rank(), 0);
    }

    #[test]
    fn single_rank_all_reduce() {
        let comm = NcclTpCommunicator::single_rank().unwrap();
        // Two f16 values: bytes [1, 0, 2, 0]
        let mut buf = vec![1u8, 0, 2, 0];
        comm.all_reduce_sum_f16(&mut buf, 2).unwrap();
        assert_eq!(buf, vec![1, 0, 2, 0]);
    }

    #[test]
    fn single_rank_all_gather() {
        let comm = NcclTpCommunicator::single_rank().unwrap();
        let input = vec![10u8, 20, 30, 40];
        let mut output = vec![0u8; 4];
        comm.all_gather_f16(&input, &mut output, 2).unwrap();
        assert_eq!(output, input);
    }
}
