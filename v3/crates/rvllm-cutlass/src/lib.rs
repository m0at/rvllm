//! rvllm-cutlass: variant catalog + policy loader + plan.
//!
//! The invariants this crate carries:
//! - **Schedule pairing is a type-level invariant.** `Variant<M, E>`
//!   requires `(M, E): MatchedPair`. Only matched pairs compile.
//!   This mirrors the CUDA `static_assert` in
//!   `cutlass_fp8_gemm_residual.cu` and makes the April-16 WS/Coop
//!   mis-pairing bug unrepresentable.
//! - **No autotune fallback chain.** A missing policy entry for a shape
//!   is a typed `CutlassError::AutotuneCacheMiss`. The engine refuses
//!   to start — no silent degradation to a default kernel.
//! - **Workspace is plan-owned.** `Fp8GemmPlan::workspace_bytes` is the
//!   authoritative number the allocator sizes against; if the runtime
//!   hands the kernel less, `check_workspace` returns `WorkspaceTooSmall`.

pub mod cublaslt;
pub mod lib_so;
pub mod plan;
pub mod policy;
pub mod schedule;
pub mod variants;
pub mod w4a8;

pub use cublaslt::CublasLt;
pub use lib_so::{CutlassBackend, CutlassLib};
pub use w4a8::W4a8Lib;
pub use plan::Fp8GemmPlan;
pub use policy::{Policy, PolicyEntry, ShapeKey};
pub use schedule::{Coop, Fp8Coop, Fp8WS, MatchedPair, Schedule, ScheduleTag, WS};
pub use variants::{
    canonical_variants, ClusterShape, TileShape, Variant, VariantDescriptor, VariantId,
    FP8_GEMM_COOP_128_128_128, FP8_GEMM_COOP_128_256_128, FP8_GEMM_FP8COOP_128_128_128,
    FP8_GEMM_FP8WS_64_128_128, FP8_GEMM_RESIDUAL_COOP, FP8_GEMM_WS_64_128_128,
};
