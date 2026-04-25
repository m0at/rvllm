//! rvllm-fused: launcher descriptors + pure-Rust f32 reference
//! implementations of every fused kernel listed in spec 12.
//!
//! The invariants this crate carries:
//! - Every fused kernel has a reference in `reference.rs`. CI runs the
//!   PTX output against the reference and fails on cosine < 0.999
//!   (f16 outputs) or |Δ| > 1e-5 (f32).
//! - Every launcher validates shape and alignment *before* launch
//!   (`require_multiple(dim, 8, ...)` — the guard whose absence caused
//!   the April 16 vectorized-quantize ILLEGAL_ADDRESS hunt).
//! - No megakernels. Each fused kernel does one recognizable composite.

pub mod gemma4_launcher;
pub mod gemma4_reference;
pub mod launch_raw;
pub mod launcher;
pub mod m2_nvfp4;
pub mod m2_prefill;
pub mod reference;

pub use launch_raw::launch_raw;
pub use launcher::{
    require_multiple, AddBiasF16Launch, ArgmaxLaunch, EmbeddingGatherLaunch,
    FusedAddRmsnormFp8QuantLaunch, FusedRmsnormFp8QuantLaunch, FusedRopeCacheFp8KvLaunch,
    FusedRopeKvWriteLaunch, FusedSiluMulFp8QuantLaunch, QuantizeFp8PerTokenLaunch,
    ResidualAddF16Launch,
};
pub use m2_nvfp4::{
    decode_fp4_e2m1, decode_fp8_e4m3, nvfp4_matmul_ref, nvfp4_weight_at, M2Nvfp4MatmulShape,
    M2Nvfp4MosaicMemory, M2Nvfp4MosaicTilePlan, FP4_E2M1_LUT, M2_MOSAIC_DEFAULT_BK,
    M2_MOSAIC_DEFAULT_BN, NVFP4_GROUP,
};
pub use m2_prefill::{M2PrefillKvDType, M2PrefillScanShape};
pub use reference::{
    argmax_ref, embedding_gather_ref, fused_add_rmsnorm_fp8_quant_ref,
    fused_gelu_mul_fp8_quant_ref, fused_silu_mul_fp8_quant_ref, quantize_fp8_per_token_ref,
    residual_add_ref, rmsnorm_ref, rope_ref, FP8_E4M3_MAX,
};
