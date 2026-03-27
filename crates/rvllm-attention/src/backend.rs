//! AttentionBackend trait definition and backend selection.

use half::f16;
use rvllm_core::prelude::Result;
use crate::buffer::GpuBuffer;

use crate::flash_attention_impl::FlashAttention2;
use crate::paged_attention::PagedAttentionV2;

/// Trait for pluggable attention backends.
///
/// Implementations must be thread-safe (`Send + Sync`) so they can be shared
/// across worker threads.
pub trait AttentionBackend: Send + Sync {
    /// Run a single paged-attention forward pass.
    ///
    /// # Arguments
    /// * `query`           - `[num_tokens, num_heads, head_dim]`
    /// * `key_cache`       - `[num_blocks, block_size, num_heads, head_dim]`
    /// * `value_cache`     - same shape as key_cache
    /// * `block_tables`    - `[num_seqs, max_blocks_per_seq]`
    /// * `context_lens`    - `[num_seqs]`
    /// * `max_context_len` - maximum context length across all sequences
    /// * `scale`           - softmax scale factor (typically 1/sqrt(head_dim))
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>>;

    /// Human-readable name for logging / debug.
    fn name(&self) -> &str;
}

/// Select the best attention backend for the given compute capability.
///
/// Returns `FlashAttention2` (CPU reference) for SM >= 8.0, `PagedAttentionV2` otherwise.
/// When the `cuda` feature is enabled, use `FlashAttention2::with_cuda()` directly
/// to get the GPU-accelerated kernel path.
pub fn select_backend(compute_capability: (u32, u32)) -> Box<dyn AttentionBackend> {
    let (major, minor) = compute_capability;
    if major >= 8 {
        tracing::info!(
            backend = "FlashAttention2",
            sm = %format!("{major}.{minor}"),
            "selected FlashAttention-2 backend"
        );
        Box::new(FlashAttention2::new())
    } else {
        tracing::info!(
            backend = "PagedAttentionV2",
            sm = %format!("{major}.{minor}"),
            "selected paged attention v2 backend"
        );
        Box::new(PagedAttentionV2::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_flash_for_sm80() {
        let backend = select_backend((8, 0));
        assert_eq!(backend.name(), "FlashAttention2-CPU");
    }

    #[test]
    fn select_flash_for_sm90() {
        let backend = select_backend((9, 0));
        assert_eq!(backend.name(), "FlashAttention2-CPU");
    }

    #[test]
    fn select_paged_for_sm75() {
        let backend = select_backend((7, 5));
        assert_eq!(backend.name(), "PagedAttentionV2");
    }

    #[test]
    fn backend_trait_is_object_safe() {
        fn _assert_object_safe(_: &dyn AttentionBackend) {}
    }
}
