//! Paged attention kernel dispatch and pluggable backend selection for vllm-rs.
//!
//! This crate provides the `AttentionBackend` trait, several implementations
//! (PagedAttentionV2, FlashAttentionPaged, FlashAttention2, MockAttentionBackend),
//! attention metadata, and a `select_backend` function that picks the best backend
//! based on GPU compute capability.

pub mod backend;
pub mod buffer;
pub mod flash_attention;
pub mod flash_attention_impl;
pub mod metadata;
pub mod mock;
pub mod paged_attention;
#[cfg(feature = "cuda")]
pub mod paged_attention_cuda;
pub mod sliding_window;

pub use backend::{select_backend, AttentionBackend};
pub use buffer::GpuBuffer;
pub use flash_attention::FlashAttentionPaged;
pub use flash_attention_impl::{FlashAttention2, FlashAttention2Config};
pub use metadata::AttentionMetadata;
pub use mock::MockAttentionBackend;
pub use paged_attention::PagedAttentionV2;
#[cfg(feature = "cuda")]
pub use paged_attention_cuda::CudaPagedAttention;
pub use sliding_window::{SlidingWindowAttention, SlidingWindowConfig};
