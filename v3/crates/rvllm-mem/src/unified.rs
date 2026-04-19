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

        // Bias residency toward the GPU so the first kernel launch
        // doesn't page-fault a gigabyte of weights in from the CPU
        // side. `cuMemAdvise_v2` is wrapped by cudarc for every CUDA
        // toolkit from 12.02 onward (including 13.0x/13.02) and takes
        // a `CUmemLocation { type, id }` value. Advise failures are
        // non-fatal — without the hint the pages simply migrate on
        // first touch.
        let loc = CUmemLocation {
            type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: ctx.device(),
        };
        let _ = unsafe {
            cuMemAdvise_v2(
                dptr,
                bytes,
                CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                loc,
            )
        };

        // HbmArena::from_raw_parts wires the pre-allocated pointer
        // into the shared bump-allocator bookkeeping + Drop via
        // cuMemFree_v2, which correctly frees managed memory as well.
        let inner = crate::hbm::HbmArena::from_raw_parts(dptr, bytes);
        Ok(Self {
            inner,
            _not_hbm: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_ctx: &crate::context::CudaContextHandle, bytes: usize) -> Result<Self> {
        // No managed-memory allocator available without `cuda`; fall
        // back to the host-stub bookkeeping so the type stays
        // constructible in no-cuda workspace checks and unit tests.
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

// `Region<'a>` already implements `GraphSafe` in `hbm.rs` — capture
// binds `&Region`, never `&UnifiedArena` (same contract as HbmArena,
// which intentionally does not carry the impl either).

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
