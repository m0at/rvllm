//! GPU abstraction layer for vllm-rs.
//!
//! Provides a trait-based GPU memory allocator, buffer types, stream wrapper,
//! and device enumeration. The default `mock-gpu` feature supplies a pure-Rust
//! heap-backed implementation with zero unsafe for CPU-only testing.

pub mod allocator;
pub mod buffer;
pub mod cpu_buffer;
#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cublas_ops;
#[cfg(feature = "cublaslt")]
pub mod cublaslt_ops;
#[cfg(feature = "cuda")]
pub mod cuda_allocator;
pub mod cuda_graph;
pub mod device;
mod ffi;
#[cfg(feature = "cuda")]
pub mod fp8_quantize;
#[cfg(feature = "cuda")]
pub mod cooperative;
#[cfg(feature = "cuda")]
pub mod kernel_loader;
#[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
pub mod mock;
pub mod nccl;
pub mod pinned_memory;
pub mod stream;

pub use rvllm_core::prelude::{LLMError, Result};

#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaGpuAllocator;

#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaStream;

pub mod prelude {
    pub use crate::allocator::GpuAllocator;
    pub use crate::buffer::GpuBuffer;
    pub use crate::cpu_buffer::CpuBuffer;
    #[cfg(feature = "cuda")]
    pub use crate::cublas::CublasHandle;
    #[cfg(feature = "cublaslt")]
    pub use crate::cublaslt_ops::{CublasLtOps, CUBLASLT_M_THRESHOLD};
    #[cfg(feature = "cuda")]
    pub use crate::cuda_allocator::CudaGpuAllocator;
    #[cfg(feature = "cuda")]
    pub use crate::CudaContext;
    #[cfg(feature = "cuda")]
    pub use crate::CudaStream;
    pub use crate::cuda_graph::{padded_batch_size, CudaGraph, CudaGraphPool, GRAPH_BATCH_SIZES};
    pub use crate::device::{list_devices, GpuDevice, MemoryInfo};
    #[cfg(feature = "cuda")]
    pub use crate::kernel_loader::{launch_config, KernelLoader};
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    pub use crate::mock::MockGpuAllocator;
    pub use crate::nccl::{NcclComm, NcclDataType, NcclGroup, NcclReduceOp, NcclUniqueId};
    pub use crate::pinned_memory::{PinnedBuffer, PinnedPool};
    pub use crate::stream::GpuStream;
    pub use crate::{LLMError, Result};
}
