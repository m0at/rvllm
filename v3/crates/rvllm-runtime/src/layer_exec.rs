//! Layer forward as a pure function per spec 09.
//!
//! The full v3 `forward` signature is:
//!   `fn forward(input, weights, kv, scratch, meta, out, scope)`
//! where every argument is borrowed (no hidden state mutation on a
//! long-lived `&mut self`).
//!
//! This module holds the public signature. The body (11 kernel launches
//! per layer) is wired when rvllm-runtime gains an actual GPU-backed
//! implementation.

use rvllm_core::Result;

/// Shape for the per-layer input / output tensor.
#[derive(Copy, Clone, Debug)]
pub struct LayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
}

/// Pure-function layer forward. `input` → `out` via 11 kernel launches.
/// Implementation pending Phase D wiring; this stub defines the
/// signature so downstream crates compile against it.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    _layer_index: u32,
    _dims: LayerDims,
    // &Tensor<...> for input, &LayerWeights, &mut KvSlab, &mut LayerScratch,
    // &Metadata, &mut Tensor<...> for out, &CaptureScope — to be added
    // when the GPU types are wired up.
) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_forward_compiles() {
        let dims = LayerDims {
            num_tokens: 128,
            hidden: 3584,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate: 18944,
        };
        assert!(forward(0, dims).is_ok());
    }
}
