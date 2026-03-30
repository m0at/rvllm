use crate::Result;

/// Communication primitives for tensor parallelism.
///
/// Implementations wrap a collective communication backend (NCCL, mock, etc.)
/// and operate on raw byte buffers. The buffers are expected to live on GPU
/// memory in the CUDA path, or host memory in the mock path.
pub trait TpCommunicator: Send + Sync {
    /// All-reduce sum across all ranks. In-place on the buffer.
    /// `count` is the number of f16 elements in `buf`.
    fn all_reduce_sum_f16(&self, buf: &mut [u8], count: usize) -> Result<()>;

    /// All-gather: each rank contributes `send_count` f16 elements.
    /// `output` must hold `send_count * world_size` f16 elements.
    fn all_gather_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        send_count: usize,
    ) -> Result<()>;

    /// Reduce-scatter: reduce across ranks, each rank gets `recv_count` f16 elements.
    fn reduce_scatter_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        recv_count: usize,
    ) -> Result<()>;

    /// Barrier: block until all ranks have reached this point.
    fn barrier(&self) -> Result<()>;

    fn world_size(&self) -> usize;
    fn rank(&self) -> usize;
}
