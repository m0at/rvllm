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

pub mod launch_raw;
pub mod launcher;
pub mod reference;

pub use launch_raw::launch_raw;
pub use launcher::{
    require_multiple, ArgmaxLaunch, FusedAddRmsnormFp8QuantLaunch, FusedRopeKvWriteLaunch,
    FusedSiluMulFp8QuantLaunch, QuantizeFp8PerTokenLaunch,
};
pub use reference::{
    argmax_ref, embedding_gather_ref, fused_add_rmsnorm_fp8_quant_ref,
    fused_silu_mul_fp8_quant_ref, quantize_fp8_per_token_ref, residual_add_ref, rmsnorm_ref,
    rope_ref, FP8_E4M3_MAX,
};
