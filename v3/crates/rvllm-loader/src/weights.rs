//! Named, typed weight storage for a loaded model.
//!
//! Fields are explicit — not indexed `Vec<CudaSlice<f16>>` that can
//! silently desync (v2's `ModelWeightsStore` was 13 parallel Vecs
//! keyed by integer, one index off would point at a different weight).
//!
//! FP8-quantized weights carry their scale alongside the data rather
//! than a separate parallel vector.

use rvllm_core::DType;

/// An f16 (or bf16) weight tensor: a region and a shape.
#[derive(Debug)]
pub struct F16Weight {
    /// Starting byte offset within the loader's HBM region.
    pub offset_bytes: u64,
    pub shape: Vec<usize>,
}

/// An FP8-quantized weight tensor with its per-tensor scale.
/// `scale_ptr` is the device pointer of the uploaded f32 scale scalar.
#[derive(Debug, Clone)]
pub struct Fp8Weight {
    pub offset_bytes: u64,
    pub scale_ptr: u64,
    pub shape: Vec<usize>,
    pub scale: f32,
    /// Clamp rate at quantization time; debug diagnostic.
    pub clamp_ppm: f32,
    pub dtype: DType, // Fp8E4M3 by default
    /// Per-channel (per-row) f32 scale vector on device. When set,
    /// cuBLASLt uses OUTER_VEC_32F mode instead of scalar scaling.
    /// Length = shape[0] (number of output rows).
    ///
    /// This is a **compatibility projection** of the underlying scale
    /// tensor: for weights whose source scale was 2-D blockwise
    /// `[N_blocks, K_blocks]`, the loader collapses it to a per-row
    /// vector by taking the first column-block's scale per row-block.
    /// Works for any GEMM path that only consumes a per-row scale;
    /// **does not** work for paths that read it as a 2-D blockscale
    /// — those must consult `blockscale_ptr` instead.
    pub channelscale_ptr: Option<u64>,
    /// Raw 2-D blockwise scale tensor `[N_blocks, K_blocks]` on device,
    /// f32. `None` when the source weight did not ship a 2-D
    /// blockscale (per-row vectors, synthesised fused weights, etc.).
    /// When `Some`, callers whose kernel ABI expects a 2-D tensor
    /// (e.g. `Fp8GemvBlockwiseF16InLaunch`) should consume this
    /// directly rather than `channelscale_ptr`, which would read
    /// the wrong shape.
    pub blockscale_ptr: Option<u64>,
    /// Shape of the 2-D blockscale tensor: `(N_blocks, K_blocks)`.
    /// Both 0 when `blockscale_ptr` is `None`.
    pub blockscale_n_blocks: u32,
    pub blockscale_k_blocks: u32,
}

/// One transformer layer's weights. Borrows into the model's HBM
/// slab; the borrow keeps the slab alive.
///
/// `qkv_bias` is the f16 concatenation of q_proj.bias || k_proj.bias ||
/// v_proj.bias, shape [q_dim + 2*kv_dim]. Applied after the fused QKV
/// GEMM. Qwen2.5 sets attention_bias=true; leaving this out produces
/// wrong logits.
#[derive(Debug)]
pub struct LayerWeights {
    pub qkv: Fp8Weight,
    pub qkv_bias: Option<F16Weight>,
    pub gate_up: Fp8Weight,
    pub o_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
}

/// The whole model's weights.
#[derive(Debug)]
pub struct LoadedModel {
    pub embedding: F16Weight,
    pub lm_head_fp8: Fp8Weight,
    pub final_norm: F16Weight,
    pub rope_cos: F16Weight,
    pub rope_sin: F16Weight,
    pub layers: Vec<LayerWeights>,
}
