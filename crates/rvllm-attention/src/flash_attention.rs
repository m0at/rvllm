//! Flash Attention paged backend -- stub for SM >= 8.0 devices.

use half::f16;
use rvllm_core::prelude::Result;
use crate::buffer::GpuBuffer;

use crate::backend::AttentionBackend;

/// Flash Attention with paged KV cache support.
///
/// Auto-selected when the device compute capability is SM >= 8.0.
/// Currently a stub -- the actual flash-attention kernel integration is TODO.
pub struct FlashAttentionPaged {
    _private: (),
}

impl FlashAttentionPaged {
    pub fn new() -> Self {
        tracing::debug!("initializing FlashAttentionPaged backend (stub)");
        Self { _private: () }
    }
}

impl Default for FlashAttentionPaged {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionBackend for FlashAttentionPaged {
    fn forward(
        &self,
        _query: &GpuBuffer<f16>,
        _key_cache: &GpuBuffer<f16>,
        _value_cache: &GpuBuffer<f16>,
        _block_tables: &GpuBuffer<i32>,
        _context_lens: &GpuBuffer<i32>,
        _max_context_len: usize,
        _scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        todo!("FlashAttentionPaged: kernel integration not yet implemented")
    }

    fn name(&self) -> &str {
        "FlashAttentionPaged"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_attention_name() {
        let fa = FlashAttentionPaged::new();
        assert_eq!(fa.name(), "FlashAttentionPaged");
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn flash_attention_forward_panics() {
        let fa = FlashAttentionPaged::new();
        let q = GpuBuffer {
            data: vec![f16::ZERO; 32],
            shape: vec![1, 4, 8],
        };
        let kc = GpuBuffer {
            data: vec![f16::ZERO; 512],
            shape: vec![1, 16, 4, 8],
        };
        let vc = GpuBuffer {
            data: vec![f16::ZERO; 512],
            shape: vec![1, 16, 4, 8],
        };
        let bt = GpuBuffer {
            data: vec![0i32],
            shape: vec![1, 1],
        };
        let cl: GpuBuffer<i32> = GpuBuffer {
            data: vec![1],
            shape: vec![1],
        };
        let _ = fa.forward(&q, &kc, &vc, &bt, &cl, 1, 0.125);
    }
}
