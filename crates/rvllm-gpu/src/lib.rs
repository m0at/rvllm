//! GPU abstraction layer for vllm-rs.
//!
//! Provides a trait-based GPU memory allocator, buffer types, stream wrapper,
//! and device enumeration. The default `mock-gpu` feature supplies a pure-Rust
//! heap-backed implementation with zero unsafe for CPU-only testing.

pub mod allocator;
pub mod buffer;
pub mod cpu_buffer;
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda_allocator;
#[cfg(feature = "cuda")]
pub mod kernel_loader;
pub mod device;
mod ffi;
#[cfg(feature = "mock-gpu")]
pub mod mock;
pub mod nccl;
pub mod pinned_memory;
pub mod stream;

pub use rvllm_core::prelude::{LLMError, Result};

#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaGpuAllocator;

pub mod prelude {
    pub use crate::allocator::GpuAllocator;
    pub use crate::buffer::GpuBuffer;
    pub use crate::cpu_buffer::CpuBuffer;
    pub use crate::cuda_graph::{CudaGraph, CudaGraphPool, padded_batch_size, GRAPH_BATCH_SIZES};
    pub use crate::device::{list_devices, GpuDevice, MemoryInfo};
    #[cfg(feature = "cuda")]
    pub use crate::cublas::CublasHandle;
    #[cfg(feature = "cuda")]
    pub use crate::cuda_allocator::CudaGpuAllocator;
    #[cfg(feature = "cuda")]
    pub use crate::kernel_loader::{launch_config, KernelLoader};
    #[cfg(feature = "mock-gpu")]
    pub use crate::mock::MockGpuAllocator;
    pub use crate::nccl::{NcclComm, NcclDataType, NcclGroup, NcclReduceOp, NcclUniqueId};
    pub use crate::pinned_memory::{PinnedBuffer, PinnedPool};
    pub use crate::stream::GpuStream;
    pub use crate::{LLMError, Result};
}
