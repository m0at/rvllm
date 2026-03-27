#![forbid(unsafe_code)]
//! Physical memory management for vllm-rs.
//!
//! Provides GPU and CPU memory pools with slab allocation, free-list tracking,
//! and swap-to-CPU support for KV cache eviction.

pub mod block;
pub mod cpu_pool;
pub mod gpu_pool;
pub mod pool;
pub mod swap;

pub use block::{CpuBlock, DeviceType, PhysicalBlock};
pub use cpu_pool::CpuMemoryPool;
pub use gpu_pool::GpuMemoryPool;
pub use pool::MemoryPool;
pub use swap::SwapManager;
