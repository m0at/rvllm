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

/// Re-exports so downstream crates (rvllm-serve) can stay within the
/// allowed DAG (rvllm-core + rvllm-runtime only) while still naming
/// the kernel/mem types that the engine API surface needs.
pub use rvllm_kernels::KernelFn;

#[cfg(feature = "cuda")]
pub mod gpu_helpers {
    //! Tiny helpers exposed to downstream serve/bench-style crates so
    //! they can keep their direct dependency set to
    //! `{rvllm-core, rvllm-runtime}` only.

    use rvllm_mem::context::CudaContextHandle;

    /// Probe free HBM (bytes) on device 0. Used by serve startup to
    /// auto-size the arena. Fails loudly if the CUDA context can't
    /// be initialised.
    pub fn probe_free_bytes() -> Result<usize, String> {
        let probe = CudaContextHandle::init(0)
            .map_err(|e| format!("CudaContextHandle::init(0): {e}"))?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        let rc = unsafe {
            cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _)
        };
        drop(probe);
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemGetInfo_v2 rc={rc:?}"));
        }
        Ok(free)
    }
}
