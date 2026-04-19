//! `HbmArena`: a single `cuMemAlloc` slab with bump-allocated `Region`s.
//!
//! The invariant this type carries is *no realloc*. Once `HbmArena::new`
//! returns, the arena's device base pointer never changes. `region()`
//! hands out sub-ranges that live for the arena's lifetime.
//!
//! `Region<'a>` is the handle. A captured CUDA graph binds device pointers
//! derived from `Region`s; because those pointers are stable for the
//! arena's lifetime and the borrow-checker keeps the arena alive longer
//! than any borrowed `Region`, replay is always sound.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicUsize, Ordering};

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::graph_safe::GraphSafe;

/// Bump-allocated HBM slab. One per device, constructed once at engine init.
#[derive(Debug)]
pub struct HbmArena<'ctx> {
    base: u64,
    capacity: usize,
    used: AtomicUsize,
    owns_cuda: bool,
    _ctx: PhantomData<&'ctx ()>,
}

impl<'ctx> Drop for HbmArena<'ctx> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if self.owns_cuda && self.base != 0 {
                let _ = cudarc::driver::sys::cuMemFree_v2(self.base);
            }
        }
    }
}

impl<'ctx> HbmArena<'ctx> {
    /// Construct a CPU-side test arena (no GPU). Useful for unit tests
    /// of the bookkeeping. Pretends to own `bytes` starting at some
    /// fake device base.
    pub fn new_host_stub(bytes: usize) -> Self {
        Self {
            base: 0x0001_0000_0000_0000, // fake device pointer
            capacity: bytes,
            used: AtomicUsize::new(0),
            owns_cuda: false,
            _ctx: PhantomData,
        }
    }

    /// Real constructor: allocate one HBM slab via `cuMemAlloc_v2`.
    /// Fails if the device doesn't have `bytes` free.
    #[cfg(feature = "cuda")]
    pub fn new(_ctx: &crate::context::CudaContextHandle, bytes: usize) -> Result<Self> {
        use cudarc::driver::sys::*;
        let mut dptr: CUdeviceptr = 0;
        let r = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "HbmArena::new (cuMemAlloc_v2)",
                CudaErrorKind::AllocFailed,
                rvllm_core::CudaCtx {
                    stream: 0,
                    kernel: "cuMemAlloc_v2",
                    launch: None,
                    device: _ctx.device(),
                },
            ));
        }
        Ok(Self {
            base: dptr,
            capacity: bytes,
            used: AtomicUsize::new(0),
            owns_cuda: true,
            _ctx: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_ctx: &crate::context::CudaContextHandle, bytes: usize) -> Result<Self> {
        Ok(Self::new_host_stub(bytes))
    }

    /// Wrap a pre-allocated device pointer + byte count as an arena.
    ///
    /// Ownership semantics:
    ///   * The arena takes *exclusive* ownership of `base`. The caller
    ///     must not retain a copy of the pointer, free it elsewhere,
    ///     or pass the same `base` to a second `from_raw_parts` call.
    ///   * On `Drop` (under `feature = "cuda"`) the arena calls
    ///     `cuMemFree_v2(base)`. This is the one CUDA deallocator that
    ///     correctly releases both dedicated-HBM allocations
    ///     (`cuMemAlloc_v2`) and managed-memory allocations
    ///     (`cuMemAllocManaged`), so the same seam works for both
    ///     arena flavours without a per-flavour teardown path.
    ///
    /// This is the private seam the `UnifiedArena` (managed-memory)
    /// constructor plugs into so the bump-allocator bookkeeping is
    /// shared across arena flavours.
    ///
    /// # Safety
    /// `base` must point to at least `bytes` of valid device-addressable
    /// memory allocated via `cuMemAlloc_v2` or `cuMemAllocManaged`
    /// (both pair with `cuMemFree_v2`). Anything else will leak or
    /// double-free at teardown.
    pub(crate) fn from_raw_parts(base: u64, bytes: usize) -> Self {
        Self {
            base,
            capacity: bytes,
            used: AtomicUsize::new(0),
            owns_cuda: true,
            _ctx: PhantomData,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn free(&self) -> usize {
        self.capacity - self.used()
    }

    /// Returns the current bump-pointer value. Paired with `restore` to
    /// free a block of scratch regions at once (e.g. between sweep
    /// iterations). The user is responsible for ensuring no outstanding
    /// `Region` borrows reference memory above the checkpoint — a safety
    /// that is enforced by the borrow checker when all regions allocated
    /// after the checkpoint have been dropped.
    pub fn checkpoint(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// Reset the bump pointer to an earlier checkpoint.
    ///
    /// # Safety
    /// Caller must ensure every `Region` allocated between `checkpoint`
    /// and this `restore` call has been dropped. Any live `Region`
    /// whose bytes lie above the restored pointer now aliases arena
    /// bytes that may be rewritten by subsequent `region` calls.
    pub unsafe fn restore(&self, ck: usize) {
        self.used.store(ck, Ordering::Release);
    }

    /// Carve a named, aligned region out of the arena. Takes `&self`
    /// (not `&mut self`) so a `CaptureScope` holding `&HbmArena` can
    /// still call `region` during init — but *inside* a captured
    /// region, `region` must not be reachable. (`CaptureScope::record`
    /// does not pass the arena in; see `capture.rs`.)
    pub fn region<'a>(
        &'a self,
        name: &'static str,
        bytes: usize,
        align: usize,
    ) -> Result<Region<'a>> {
        let align = align.max(1);
        let prev = self.used.load(Ordering::Acquire);
        let aligned_start = prev.next_multiple_of(align);
        let end = aligned_start.checked_add(bytes).ok_or_else(|| {
            RvllmError::cuda(
                "HbmArena::region",
                CudaErrorKind::AllocFailed,
                CudaCtx::setup(),
            )
        })?;
        if end > self.capacity {
            return Err(RvllmError::cuda(
                "HbmArena::region",
                CudaErrorKind::AllocFailed,
                CudaCtx::setup(),
            ));
        }
        // Monotonic bump; no contention in the single-worker model.
        self.used.store(end, Ordering::Release);
        Ok(Region {
            arena: self,
            name,
            offset: aligned_start,
            len: bytes,
        })
    }
}

/// A named, immutable range inside an `HbmArena`. Borrowing it prevents
/// the arena from being dropped; the device pointer is stable for the
/// region's lifetime.
#[derive(Debug)]
pub struct Region<'a> {
    arena: &'a HbmArena<'a>,
    name: &'static str,
    offset: usize,
    len: usize,
}

impl<'a> Region<'a> {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Device pointer of the region. Stable for `'a`.
    pub fn device_ptr(&self) -> u64 {
        self.arena.base + self.offset as u64
    }

    /// Synchronous H2D upload into this region. Fails if `src.len()`
    /// exceeds `self.len()`.
    ///
    /// # Safety
    /// Caller must ensure no concurrent kernel is reading the region.
    /// This function issues a synchronous cuMemcpyHtoD_v2 which
    /// serializes on the default stream; it's for load-time population,
    /// not the graph-captured fast path.
    pub unsafe fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        if src.len() > self.len {
            return Err(RvllmError::cuda(
                "Region::copy_from_host (len)",
                CudaErrorKind::AllocFailed,
                CudaCtx::setup(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            let r = cuMemcpyHtoD_v2(self.device_ptr(), src.as_ptr() as *const _, src.len());
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemcpyHtoD_v2",
                    CudaErrorKind::AllocFailed,
                    CudaCtx::setup(),
                ));
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = src;
        }
        Ok(())
    }
}

// A `Region` is GraphSafe: it borrows the arena, the arena is fixed-size
// and non-reallocating, and the region's device pointer is constant for
// the lifetime of the borrow. Capture may bind `&Region`.
unsafe impl<'a> GraphSafe for Region<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_allocation_is_monotonic_and_aligned() {
        let a = HbmArena::new_host_stub(1 << 20);
        let r1 = a.region("a", 100, 16).unwrap();
        assert_eq!(r1.device_ptr() % 16, 0);
        let r2 = a.region("b", 200, 256).unwrap();
        assert_eq!(r2.device_ptr() % 256, 0);
        assert!(r2.device_ptr() > r1.device_ptr());
        assert!(a.used() >= 300);
    }

    #[test]
    fn exhaustion_returns_err() {
        let a = HbmArena::new_host_stub(1024);
        let _ok = a.region("ok", 512, 1).unwrap();
        let err = a.region("too big", 1024, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("cuda"));
        assert!(s.contains("HbmArena::region"));
    }
}
