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
#[derive(Debug)]
pub struct Fp8Weight {
    pub offset_bytes: u64,
    pub scale_ptr: u64,
    pub shape: Vec<usize>,
    pub scale: f32,
    /// Clamp rate at quantization time; debug diagnostic.
    pub clamp_ppm: f32,
    pub dtype: DType, // Fp8E4M3 by default
}

/// One transformer layer's weights. Borrows into the model's HBM
/// slab; the borrow keeps the slab alive.
#[derive(Debug)]
pub struct LayerWeights {
    pub qkv: Fp8Weight,
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
