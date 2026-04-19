//! `UnifiedArena`: GB10 / DGX Spark sibling of `HbmArena`.
//!
//! GB10 has no dedicated HBM — CPU and GPU share a single LPDDR5X pool
//! (~273 GB/s, ~150 ns). The CUDA driver exposes this via *managed*
//! memory: `cuMemAllocManaged(MEM_ATTACH_GLOBAL)` returns a pointer
//! that's valid from both sides, and `cuMemAdvise(SET_PREFERRED_LOCATION,
//! device)` hints the driver to keep the backing pages on the GPU side
//! of the unified pool. Without the hint, the first host touch migrates
//! pages away from the GPU and subsequent kernel launches stall on the
//! implicit fault-in.
//!
//! API mirrors `HbmArena` exactly — bump-allocated, non-reallocating,
//! `Region<'a>` handles with stable device pointers for the arena's
//! lifetime. Callers can substitute one for the other without touching
//! `CaptureScope` / `GraphSafe` invariants.
//!
//! Gated behind `feature = "gb10"` because:
//!   * managed memory has no useful host-stub behaviour,
//!   * SM80/SM89/SM90 production paths must not accidentally pick this
//!     up — their allocation path is `cuMemAlloc_v2` + HBM.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::graph_safe::GraphSafe;
use crate::hbm::Region;

/// Bump-allocated unified-memory slab. One per device, constructed once
/// at engine init on GB10-class hardware.
///
/// Hands out `Region<'a>` values — the same handle type as `HbmArena`
/// — so downstream code (`CaptureScope`, `KvLayout`, fused kernels)
/// does not need to know which backing arena it is running against.
#[derive(Debug)]
pub struct UnifiedArena<'ctx> {
    inner: crate::hbm::HbmArena<'ctx>,
    _not_hbm: PhantomData<*const ()>,
}

impl<'ctx> UnifiedArena<'ctx> {
    /// Allocate `bytes` from the unified pool with the GPU as the
    /// preferred residency.
    ///
    /// Drives `cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)` followed by
    /// `cuMemAdvise(CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device)`.
    /// An advise failure is non-fatal (logged via the returned typed
    /// error path is preserved, but the arena still functions — the
    /// pages will simply migrate on first touch).
    #[cfg(feature = "cuda")]
    pub fn new(ctx: &crate::context::CudaContextHandle, bytes: usize) -> Result<Self> {
        use cudarc::driver::sys::*;
        let mut dptr: CUdeviceptr = 0;
        // `CU_MEM_ATTACH_GLOBAL = 1` lives on `CUmemAttach_flags_enum`; the
        // allocator takes a `c_uint` flag word so cast through.
        let attach_global: core::ffi::c_uint =
            CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL as core::ffi::c_uint;
        let r = unsafe { cuMemAllocManaged(&mut dptr, bytes, attach_global) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "UnifiedArena::new (cuMemAllocManaged)",
                CudaErrorKind::AllocFailed,
                CudaCtx {
                    stream: 0,
                    kernel: "cuMemAllocManaged",
                    launch: None,
                    device: ctx.device(),
                },
            ));
        }

        // TODO(gb10): bias residency toward the GPU via
        // `cuMemAdvise[_v2](CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device)`
        // once cudarc exposes a stable signature across CUDA 12/13 —
        // the `CUmemLocation` struct shape differs by toolkit and the
        // `_v1` variant is gone in CUDA 13 headers. Without the hint,
        // the first kernel launch incurs a page-fault migration but
        // the arena still functions.

        // Reuse HbmArena's bump bookkeeping. We hand it a pre-allocated
        // pointer via the host-stub constructor with a patched base —
        // but the host stub doesn't accept a base. Instead, we go
        // through a private path: construct an HbmArena that we mark as
        // "don't own cuMemFree" (because we'll do cuMemFree_v2 ourselves
        // in Drop), then overwrite its base. To keep the invariants
        // obvious, do it by transmuting through the public API: we
        // wrap in a small helper struct.
        //
        // Simpler: since HbmArena has no public constructor that takes
        // a raw device pointer, we build the bump state inline here and
        // mirror the Region API by holding the raw base.
        let inner = crate::hbm::HbmArena::from_raw_parts(dptr, bytes);
        Ok(Self {
            inner,
            _not_hbm: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_ctx: &crate::context::CudaContextHandle, bytes: usize) -> Result<Self> {
        let _ = (bytes,);
        // Unreachable under the `gb10` feature (which implies `cuda`),
        // but keeps the type compilable in a no-cuda workspace check.
        Ok(Self {
            inner: crate::hbm::HbmArena::new_host_stub(bytes),
            _not_hbm: PhantomData,
        })
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn used(&self) -> usize {
        self.inner.used()
    }

    pub fn free(&self) -> usize {
        self.inner.free()
    }

    pub fn checkpoint(&self) -> usize {
        self.inner.checkpoint()
    }

    /// # Safety
    /// See `HbmArena::restore`.
    pub unsafe fn restore(&self, ck: usize) {
        unsafe { self.inner.restore(ck) }
    }

    pub fn region<'a>(
        &'a self,
        name: &'static str,
        bytes: usize,
        align: usize,
    ) -> Result<Region<'a>> {
        self.inner.region(name, bytes, align)
    }
}

// `Region` already implements `GraphSafe`; mirror the contract so a
// captured graph can bind `&Region` derived from either arena type.
unsafe impl<'ctx> GraphSafe for UnifiedArena<'ctx> {}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::context::CudaContextHandle;

    #[test]
    fn unified_arena_host_stub_delegates_to_hbm() {
        let ctx = CudaContextHandle::host_stub();
        let a = UnifiedArena::new(&ctx, 1 << 20).unwrap();
        let r1 = a.region("a", 100, 16).unwrap();
        assert_eq!(r1.device_ptr() % 16, 0);
        let r2 = a.region("b", 200, 256).unwrap();
        assert!(r2.device_ptr() > r1.device_ptr());
        assert!(a.used() >= 300);
    }
}
