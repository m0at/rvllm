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

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::hbm::Region;

/// Bump-allocated unified-memory slab. One per device, constructed once
/// at engine init on GB10-class hardware.
///
/// Hands out `Region<'a>` values — the same handle type as `HbmArena`
/// — so downstream code (`CaptureScope`, `KvLayout`, fused kernels)
/// does not need to know which backing arena it is running against.
///
/// Nominal newtype over `HbmArena` (distinct type to keep the two
/// flavours from being silently swapped) while inheriting `Send +
/// Sync`: managed memory has no thread affinity, unlike a `CUcontext`.
#[derive(Debug)]
pub struct UnifiedArena<'ctx> {
    inner: crate::hbm::HbmArena<'ctx>,
}

impl<'ctx> UnifiedArena<'ctx> {
    /// Allocate `bytes` from the unified pool with the GPU as the
    /// preferred residency.
    ///
    /// Drives `cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)` followed by
    /// `cuMemAdvise_v2(CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device)`.
    /// The advise call is a performance hint: its result is ignored
    /// because the arena is fully functional without it — the pages
    /// simply migrate on first GPU touch. Only the allocation itself
    /// is allowed to fail the constructor.
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
        Ok(Self { inner })
    }

    /// `gb10` + `!cuda` is a meaningless combination at runtime — the
    /// feature name literally means "GB10 hardware". Fail closed
    /// rather than silently returning a host-stub pointer that would
    /// SEGFAULT the first time a kernel tried to read it.
    #[cfg(not(feature = "cuda"))]
    pub fn new(ctx: &crate::context::CudaContextHandle, _bytes: usize) -> Result<Self> {
        Err(RvllmError::cuda(
            "UnifiedArena::new",
            CudaErrorKind::Other,
            CudaCtx {
                stream: 0,
                kernel: "UnifiedArena::new",
                launch: None,
                device: ctx.device(),
            },
        ))
    }

    /// Test-only stub constructor: host-side bump bookkeeping over a
    /// fake device base, never backed by real managed memory. Only
    /// the bookkeeping invariants (alignment, non-overlap, exhaustion)
    /// are exercisable through this constructor — device pointers
    /// returned by `region().device_ptr()` MUST NOT be dereferenced.
    #[cfg(any(test, not(feature = "cuda")))]
    pub fn host_stub(bytes: usize) -> Self {
        Self {
            inner: crate::hbm::HbmArena::new_host_stub(bytes),
        }
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

    /// Unwrap into the backing `HbmArena`. Used by `Bringup::load` to
    /// store a single arena type on the struct while still sourcing
    /// the backing memory from `cuMemAllocManaged` on GB10. Ownership
    /// of the device pointer transfers — Drop / `cuMemFree_v2` now
    /// lives on the returned `HbmArena`.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> crate::hbm::HbmArena<'ctx> {
        self.inner
    }
}

// `Region<'a>` already implements `GraphSafe` in `hbm.rs` — capture
// binds `&Region`, never `&UnifiedArena` (same contract as HbmArena,
// which intentionally does not carry the impl either).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_stub_bookkeeping_matches_hbm() {
        let a = UnifiedArena::host_stub(1 << 20);
        let r1 = a.region("a", 100, 16).unwrap();
        assert_eq!(r1.device_ptr() % 16, 0);
        let r2 = a.region("b", 200, 256).unwrap();
        assert!(r2.device_ptr() > r1.device_ptr());
        assert!(a.used() >= 300);
    }

    /// Compile-time check: `UnifiedArena<'_>` must inherit `Send + Sync`
    /// from its inner `HbmArena`. Managed memory has no thread
    /// affinity (unlike a `CUcontext`), so the type should be movable
    /// between worker threads. This assert-function will fail to
    /// compile if a marker makes the arena `!Send` or `!Sync`.
    #[allow(dead_code)]
    fn assert_send_sync() {
        fn check<T: Send + Sync>() {}
        check::<UnifiedArena<'static>>();
    }
}
