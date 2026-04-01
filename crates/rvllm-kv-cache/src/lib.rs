#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Paged KV cache for vllm-rs.
//!
//! Provides a paged key-value cache data structure for attention kernels,
//! a cache engine for managing per-layer GPU/CPU caches, and cache operations
//! for reshaping and writing into paged buffers.

pub mod cache;
pub mod engine;
#[cfg(feature = "cuda")]
pub mod engine_cuda;
pub mod fp8_cache;
pub mod ops;
#[cfg(feature = "cuda")]
pub mod ops_cuda;

pub use cache::{CacheConfig, KVCache};
pub use engine::CacheEngine;
pub use fp8_cache::{
    dequantize_heads, quantize_heads, FP8CacheConfig, FP8CacheEngine, KVCacheDtype,
};
pub use ops::reshape_and_cache;
