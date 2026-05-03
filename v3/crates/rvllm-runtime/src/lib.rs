//! rvllm-runtime: Engine + scheduler + layer_exec per specs 07, 09.
//!
//! The public API surface for v3 callers:
//! - `Engine::new()` → init
//! - `engine.step_launch()` → returns `PendingStep<'_>`
//! - `engine.step_collect(ticket)` → waits DtoH, returns per-request
//!   outputs
//!
//! One codepath. No sync vs pipelined duality. Graph replay is a
//! transparent implementation detail.

/// Cycle 56 step 7+8: shared CUDA-result-check macros. Wraps a
/// cudarc-driver call expression that returns CUresult and propagates
/// non-success as `RvllmError::Cuda` from the enclosing `Result<...>`
/// fn — the macro form is needed (vs a function) because the macro
/// `return Err(...)` lets callers skip the `?` and surrounding state
/// is correctly preserved in the early-return.
///
/// Two variants:
///   * `cuda_check!(call, op, stream)` — defaults to MemcpyFailed
///     `CudaErrorKind`; covers the common cuMemcpy/cuMemset cases.
///     Memcpy IS the more frequent operation; memset reuses this
///     macro because the surrounding op label disambiguates in logs.
///   * `cuda_check_kind!(kind, call, op, stream)` — explicit kind
///     for non-memcpy CUDA calls (e.g. cuStreamSynchronize →
///     StreamFailed, cuModuleLoad → ModuleLoadFailed). Use when
///     `MemcpyFailed` is misleading enough that it would mask the
///     actual failure class during triage.
///
/// Usage:
///   cuda_check!(cudarc::driver::sys::cuMemcpyDtoDAsync_v2(...),
///               "qkv_input_memcpy", stream);
///   cuda_check_kind!(StreamFailed,
///               cudarc::driver::sys::cuStreamSynchronize(stream),
///               "kv_dump_fence", stream);
///
/// Lives at crate root + `#[macro_export]` so submodules pick it up
/// via crate-root macro resolution.
#[cfg(feature = "cuda")]
#[macro_export]
#[allow(unused_macros)]
macro_rules! cuda_check {
    ($call:expr, $op:expr, $stream:expr) => {
        $crate::cuda_check_kind!(MemcpyFailed, $call, $op, $stream)
    };
}

#[cfg(feature = "cuda")]
#[macro_export]
#[allow(unused_macros)]
macro_rules! cuda_check_kind {
    ($kind:ident, $call:expr, $op:expr, $stream:expr) => {{
        let r = $call;
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::$kind,
                op: $op,
                ctx: rvllm_core::CudaCtx {
                    stream: $stream as u64,
                    kernel: "",
                    launch: None,
                    device: 0,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
    }};
}

pub mod bring_up;
pub mod engine;
pub mod gemma4_bring_up;
pub mod gemma4_layer_exec;
pub mod layer_exec;
pub mod sched_state;
pub mod scheduler;

pub use bring_up::{Bringup, EnginePaths, FusedModules, PplResult};
pub use engine::{Engine, PendingStep, StepOutput};
pub use layer_exec::{forward, LayerDims};
pub use sched_state::{ReqState, Request};
pub use scheduler::{bucket_for, BatchPlan, Scheduler, DECODE_BUCKETS};
