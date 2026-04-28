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

/// Cycle 56 step 7: shared CUDA-result-check macro hoisted from
/// gemma4_layer_exec.rs and gemma4_bring_up.rs (where it was
/// duplicated verbatim). Wraps a cudarc-driver call expression that
/// returns CUresult and propagates non-success as `RvllmError::Cuda`
/// from the enclosing `Result<...>` fn — the macro form is needed
/// (vs a function) because `return Err(...)` lets it short-circuit
/// without forcing every callsite to add `?`.
///
/// Usage:
///   cuda_check!(cudarc::driver::sys::cuMemcpyDtoDAsync_v2(...),
///               "qkv_input_memcpy", stream);
///
/// Lives at crate root + `#[macro_export]` so submodules pick it up
/// via `use crate::cuda_check;` — that's also why it's marked
/// `#[allow(unused_macros)]` for non-cuda builds.
#[cfg(feature = "cuda")]
#[macro_export]
#[allow(unused_macros)]
macro_rules! cuda_check {
    ($call:expr, $op:expr, $stream:expr) => {{
        let r = $call;
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::Cuda {
                kind: rvllm_core::CudaErrorKind::MemcpyFailed,
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
