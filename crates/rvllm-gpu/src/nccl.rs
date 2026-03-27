//! NCCL bindings for multi-GPU collective communication.
//!
//! Under the `cuda` feature, this module provides FFI bindings to the NCCL
//! library for AllReduce, AllGather, ReduceScatter, and Broadcast operations.
//! Under `mock-gpu`, it provides a single-rank mock that copies in-place.

use crate::{LLMError, Result};

/// NCCL reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Prod,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

/// NCCL data type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    /// 32-bit float.
    Float32,
    /// 16-bit float.
    Float16,
    /// 16-bit bfloat.
    BFloat16,
}

impl NcclDataType {
    /// Size in bytes of a single element.
    pub fn element_size(self) -> usize {
        match self {
            NcclDataType::Float32 => 4,
            NcclDataType::Float16 | NcclDataType::BFloat16 => 2,
        }
    }
}

/// Unique identifier for an NCCL communicator group.
#[derive(Debug, Clone)]
pub struct NcclUniqueId {
    /// Opaque bytes (128 bytes for real NCCL, arbitrary for mock).
    pub bytes: Vec<u8>,
}

impl NcclUniqueId {
    /// Generate a new unique ID for bootstrapping a communicator group.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Real NCCL: call ncclGetUniqueId
            let mut id_bytes = vec![0u8; 128];
            unsafe {
                let ret = ffi::ncclGetUniqueId(id_bytes.as_mut_ptr());
                if ret != 0 {
                    tracing::warn!(ret, "ncclGetUniqueId failed, using random fallback");
                    use std::time::SystemTime;
                    let seed = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    id_bytes[..8].copy_from_slice(&seed.to_le_bytes());
                }
            }
            Self { bytes: id_bytes }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self {
                bytes: vec![0xCC; 128],
            }
        }
    }
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to an NCCL communicator for a single rank.
///
/// Each GPU in a tensor-parallel group holds one `NcclComm`. Collectives
/// are called on all ranks simultaneously.
pub struct NcclComm {
    rank: usize,
    world_size: usize,
    #[cfg(feature = "cuda")]
    handle: *mut std::ffi::c_void,
}

// SAFETY: NCCL communicator handles are thread-safe for calls from the owning
// CUDA context. We ensure one NcclComm per GPU thread.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    /// Initialize a communicator for the given rank within a group.
    ///
    /// All ranks must call this with the same `unique_id` and `world_size`.
    pub fn new(unique_id: &NcclUniqueId, world_size: usize, rank: usize) -> Result<Self> {
        if rank >= world_size {
            return Err(LLMError::ConfigError(format!(
                "NCCL rank {} >= world_size {}", rank, world_size
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
            let ret = unsafe {
                ffi::ncclCommInitRank(
                    &mut handle,
                    world_size as i32,
                    unique_id.bytes.as_ptr(),
                    rank as i32,
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclCommInitRank failed with code {}", ret
                )));
            }
            tracing::info!(rank, world_size, "NCCL communicator initialized");
            Ok(Self { rank, world_size, handle })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = unique_id;
            tracing::info!(rank, world_size, "NCCL communicator initialized (mock)");
            Ok(Self { rank, world_size })
        }
    }

    /// Rank of this communicator within the group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks in the group.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// AllReduce: reduce `sendbuf` across all ranks and write the result to
    /// `recvbuf` on every rank.
    ///
    /// Under `mock-gpu`, this simply copies `sendbuf` to `recvbuf` since there
    /// is only a single (logical) rank.
    pub fn all_reduce(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        count: usize,
        dtype: NcclDataType,
        op: NcclReduceOp,
    ) -> Result<()> {
        let expected_bytes = count * dtype.element_size();
        if sendbuf.len() < expected_bytes || recvbuf.len() < expected_bytes {
            return Err(LLMError::MemoryError(format!(
                "all_reduce buffer too small: need {} bytes, send={} recv={}",
                expected_bytes,
                sendbuf.len(),
                recvbuf.len()
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let nccl_dtype = to_nccl_dtype(dtype);
            let nccl_op = to_nccl_op(op);
            let ret = unsafe {
                ffi::ncclAllReduce(
                    sendbuf.as_ptr() as *const std::ffi::c_void,
                    recvbuf.as_mut_ptr() as *mut std::ffi::c_void,
                    count,
                    nccl_dtype,
                    nccl_op,
                    self.handle,
                    std::ptr::null_mut(), // default stream
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclAllReduce failed with code {}", ret
                )));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (count, dtype, op);
            // Mock: single rank, just copy send to recv
            recvbuf[..expected_bytes].copy_from_slice(&sendbuf[..expected_bytes]);
        }

        tracing::trace!(rank = self.rank, count, ?op, "all_reduce complete");
        Ok(())
    }

    /// In-place AllReduce: reduce buffer across all ranks, writing result back.
    pub fn all_reduce_in_place(
        &self,
        buf: &mut [u8],
        count: usize,
        dtype: NcclDataType,
        op: NcclReduceOp,
    ) -> Result<()> {
        let expected_bytes = count * dtype.element_size();
        if buf.len() < expected_bytes {
            return Err(LLMError::MemoryError(format!(
                "all_reduce_in_place buffer too small: need {} bytes, got {}",
                expected_bytes,
                buf.len()
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let nccl_dtype = to_nccl_dtype(dtype);
            let nccl_op = to_nccl_op(op);
            let ptr = buf.as_mut_ptr() as *mut std::ffi::c_void;
            let ret = unsafe {
                ffi::ncclAllReduce(
                    ptr as *const std::ffi::c_void,
                    ptr,
                    count,
                    nccl_dtype,
                    nccl_op,
                    self.handle,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclAllReduce (in-place) failed with code {}", ret
                )));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (count, dtype, op);
            // Mock: single rank, buffer is already the result
        }

        tracing::trace!(rank = self.rank, count, ?op, "all_reduce_in_place complete");
        Ok(())
    }

    /// AllGather: each rank contributes `sendcount` elements; every rank
    /// receives all data concatenated in rank order.
    pub fn all_gather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        sendcount: usize,
        dtype: NcclDataType,
    ) -> Result<()> {
        let send_bytes = sendcount * dtype.element_size();
        let recv_bytes = send_bytes * self.world_size;
        if sendbuf.len() < send_bytes {
            return Err(LLMError::MemoryError(format!(
                "all_gather sendbuf too small: need {} bytes, got {}",
                send_bytes,
                sendbuf.len()
            )));
        }
        if recvbuf.len() < recv_bytes {
            return Err(LLMError::MemoryError(format!(
                "all_gather recvbuf too small: need {} bytes, got {}",
                recv_bytes,
                recvbuf.len()
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let nccl_dtype = to_nccl_dtype(dtype);
            let ret = unsafe {
                ffi::ncclAllGather(
                    sendbuf.as_ptr() as *const std::ffi::c_void,
                    recvbuf.as_mut_ptr() as *mut std::ffi::c_void,
                    sendcount,
                    nccl_dtype,
                    self.handle,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclAllGather failed with code {}", ret
                )));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (sendcount, dtype);
            // Mock: single rank, just copy
            recvbuf[..send_bytes].copy_from_slice(&sendbuf[..send_bytes]);
        }

        tracing::trace!(rank = self.rank, sendcount, "all_gather complete");
        Ok(())
    }

    /// ReduceScatter: reduce across ranks, then scatter equal-sized chunks.
    ///
    /// Each rank ends up with `recvcount` reduced elements.
    pub fn reduce_scatter(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        recvcount: usize,
        dtype: NcclDataType,
        op: NcclReduceOp,
    ) -> Result<()> {
        let recv_bytes = recvcount * dtype.element_size();
        let send_bytes = recv_bytes * self.world_size;
        if sendbuf.len() < send_bytes {
            return Err(LLMError::MemoryError(format!(
                "reduce_scatter sendbuf too small: need {} bytes, got {}",
                send_bytes,
                sendbuf.len()
            )));
        }
        if recvbuf.len() < recv_bytes {
            return Err(LLMError::MemoryError(format!(
                "reduce_scatter recvbuf too small: need {} bytes, got {}",
                recv_bytes,
                recvbuf.len()
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let nccl_dtype = to_nccl_dtype(dtype);
            let nccl_op = to_nccl_op(op);
            let ret = unsafe {
                ffi::ncclReduceScatter(
                    sendbuf.as_ptr() as *const std::ffi::c_void,
                    recvbuf.as_mut_ptr() as *mut std::ffi::c_void,
                    recvcount,
                    nccl_dtype,
                    nccl_op,
                    self.handle,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclReduceScatter failed with code {}", ret
                )));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (recvcount, dtype, op);
            // Mock: single rank, copy the rank-0 chunk
            let offset = self.rank * recv_bytes;
            recvbuf[..recv_bytes].copy_from_slice(&sendbuf[offset..offset + recv_bytes]);
        }

        tracing::trace!(rank = self.rank, recvcount, ?op, "reduce_scatter complete");
        Ok(())
    }

    /// Broadcast `count` elements from `root` to all other ranks.
    pub fn broadcast(
        &self,
        buf: &mut [u8],
        count: usize,
        dtype: NcclDataType,
        root: usize,
    ) -> Result<()> {
        let expected_bytes = count * dtype.element_size();
        if buf.len() < expected_bytes {
            return Err(LLMError::MemoryError(format!(
                "broadcast buffer too small: need {} bytes, got {}",
                expected_bytes,
                buf.len()
            )));
        }

        #[cfg(feature = "cuda")]
        {
            let nccl_dtype = to_nccl_dtype(dtype);
            let ptr = buf.as_mut_ptr() as *mut std::ffi::c_void;
            let ret = unsafe {
                ffi::ncclBroadcast(
                    ptr as *const std::ffi::c_void,
                    ptr,
                    count,
                    nccl_dtype,
                    root as i32,
                    self.handle,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 {
                return Err(LLMError::GpuError(format!(
                    "ncclBroadcast failed with code {}", ret
                )));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (count, dtype, root);
            // Mock: single rank, buffer already has the data
        }

        tracing::trace!(rank = self.rank, count, root, "broadcast complete");
        Ok(())
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !self.handle.is_null() {
                unsafe {
                    ffi::ncclCommDestroy(self.handle);
                }
            }
        }
        tracing::debug!(rank = self.rank, "NCCL communicator destroyed");
    }
}

/// Convenience wrapper managing an entire NCCL communicator group.
///
/// Holds the unique ID and all per-rank communicators. In production,
/// each rank would live in a separate process; this is for single-process
/// multi-GPU (threading model).
pub struct NcclGroup {
    comms: Vec<NcclComm>,
}

impl NcclGroup {
    /// Create a group of communicators for `world_size` ranks.
    pub fn new(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(LLMError::ConfigError(
                "NCCL world_size must be >= 1".into(),
            ));
        }

        let unique_id = NcclUniqueId::new();
        let mut comms = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            comms.push(NcclComm::new(&unique_id, world_size, rank)?);
        }
        tracing::info!(world_size, "NCCL group created");
        Ok(Self { comms })
    }

    /// Get the communicator for the given rank.
    pub fn comm(&self, rank: usize) -> Option<&NcclComm> {
        self.comms.get(rank)
    }

    /// Number of ranks in this group.
    pub fn world_size(&self) -> usize {
        self.comms.len()
    }
}

// ---------------------------------------------------------------------------
// CUDA FFI declarations (only compiled with `cuda` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod ffi {
    extern "C" {
        pub fn ncclGetUniqueId(id: *mut u8) -> i32;

        pub fn ncclCommInitRank(
            comm: *mut *mut std::ffi::c_void,
            nranks: i32,
            unique_id: *const u8,
            rank: i32,
        ) -> i32;

        pub fn ncclAllReduce(
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            count: usize,
            datatype: i32,
            op: i32,
            comm: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        pub fn ncclAllGather(
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            sendcount: usize,
            datatype: i32,
            comm: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        pub fn ncclReduceScatter(
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            recvcount: usize,
            datatype: i32,
            op: i32,
            comm: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        pub fn ncclBroadcast(
            sendbuf: *const std::ffi::c_void,
            recvbuf: *mut std::ffi::c_void,
            count: usize,
            datatype: i32,
            root: i32,
            comm: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        pub fn ncclCommDestroy(comm: *mut std::ffi::c_void) -> i32;
    }
}

/// Convert our dtype enum to NCCL's integer type codes.
#[cfg(feature = "cuda")]
fn to_nccl_dtype(dtype: NcclDataType) -> i32 {
    match dtype {
        NcclDataType::Float32 => 7,  // ncclFloat32
        NcclDataType::Float16 => 6,  // ncclFloat16
        NcclDataType::BFloat16 => 9, // ncclBfloat16
    }
}

/// Convert our reduce-op enum to NCCL's integer op codes.
#[cfg(feature = "cuda")]
fn to_nccl_op(op: NcclReduceOp) -> i32 {
    match op {
        NcclReduceOp::Sum => 0,  // ncclSum
        NcclReduceOp::Prod => 1, // ncclProd
        NcclReduceOp::Max => 2,  // ncclMax
        NcclReduceOp::Min => 3,  // ncclMin
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_id_creates() {
        let id = NcclUniqueId::new();
        assert_eq!(id.bytes.len(), 128);
    }

    #[test]
    fn mock_comm_init() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 4, 2).unwrap();
        assert_eq!(comm.rank(), 2);
        assert_eq!(comm.world_size(), 4);
    }

    #[test]
    fn rank_out_of_bounds() {
        let id = NcclUniqueId::new();
        let result = NcclComm::new(&id, 2, 5);
        assert!(result.is_err());
    }

    #[test]
    fn mock_all_reduce() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let send: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // two f32-sized chunks
        let mut recv = vec![0u8; 8];
        comm.all_reduce(&send, &mut recv, 2, NcclDataType::Float32, NcclReduceOp::Sum)
            .unwrap();
        assert_eq!(recv, send);
    }

    #[test]
    fn mock_all_reduce_in_place() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let mut buf: Vec<u8> = vec![10, 20, 30, 40];
        comm.all_reduce_in_place(&mut buf, 2, NcclDataType::Float16, NcclReduceOp::Sum)
            .unwrap();
        assert_eq!(buf, vec![10, 20, 30, 40]);
    }

    #[test]
    fn mock_all_gather() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let send = vec![1u8, 2, 3, 4];
        let mut recv = vec![0u8; 4];
        comm.all_gather(&send, &mut recv, 1, NcclDataType::Float32).unwrap();
        assert_eq!(recv, send);
    }

    #[test]
    fn mock_reduce_scatter() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let send = vec![5u8, 6, 7, 8];
        let mut recv = vec![0u8; 4];
        comm.reduce_scatter(&send, &mut recv, 1, NcclDataType::Float32, NcclReduceOp::Sum)
            .unwrap();
        assert_eq!(recv, send);
    }

    #[test]
    fn mock_broadcast() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let mut buf = vec![42u8, 43, 44, 45];
        comm.broadcast(&mut buf, 1, NcclDataType::Float32, 0).unwrap();
        assert_eq!(buf, vec![42, 43, 44, 45]);
    }

    #[test]
    fn all_reduce_buffer_too_small() {
        let id = NcclUniqueId::new();
        let comm = NcclComm::new(&id, 1, 0).unwrap();

        let send = vec![0u8; 2];
        let mut recv = vec![0u8; 2];
        let result =
            comm.all_reduce(&send, &mut recv, 4, NcclDataType::Float32, NcclReduceOp::Sum);
        assert!(result.is_err());
    }

    #[test]
    fn group_creation() {
        let group = NcclGroup::new(4).unwrap();
        assert_eq!(group.world_size(), 4);
        for rank in 0..4 {
            let c = group.comm(rank).unwrap();
            assert_eq!(c.rank(), rank);
            assert_eq!(c.world_size(), 4);
        }
    }

    #[test]
    fn group_zero_size_rejected() {
        let result = NcclGroup::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(NcclDataType::Float32.element_size(), 4);
        assert_eq!(NcclDataType::Float16.element_size(), 2);
        assert_eq!(NcclDataType::BFloat16.element_size(), 2);
    }
}
