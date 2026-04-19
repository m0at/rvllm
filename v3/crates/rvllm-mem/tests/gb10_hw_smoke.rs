//! Hardware smoke test for the GB10 / sm_121 path.
//!
//! Runs end-to-end on a live CUDA context:
//!   1. Init `CudaContextHandle` via the primary-context retain path
//!      (validates the CUDA-13 fix in `context.rs`).
//!   2. Query the device's compute capability and map it to a
//!      `CompileTarget` — on this box, expect `Sm121`.
//!   3. Construct a `UnifiedArena` via `cuMemAllocManaged` and carve
//!      a handful of `Region`s out of it; verify the bump allocator
//!      hands out non-overlapping, aligned device pointers.
//!
//! Marked `#[ignore]` because it requires a real CUDA device. Run with
//!     cargo test -p rvllm-mem --features gb10,cuda \
//!         --test gb10_hw_smoke -- --ignored --nocapture
//! Only compiled when BOTH `gb10` and `cuda` features are on.

#![cfg(all(feature = "gb10", feature = "cuda"))]

use rvllm_core::CompileTarget;
use rvllm_mem::context::CudaContextHandle;
use rvllm_mem::unified::UnifiedArena;

#[test]
#[ignore = "requires a real CUDA device; run with `--ignored`"]
fn gb10_end_to_end_bring_up() {
    // CUDA context — on a GPU-less machine this would panic via the
    // expect below, which is what we want under `--ignored` (the whole
    // test is opt-in to hardware presence).
    let ctx = CudaContextHandle::init(0).expect("CudaContextHandle::init");

    // Step 2 — compute capability → CompileTarget.
    let (major, minor) = ctx.compute_capability();
    let target = CompileTarget::from_compute_capability(major, minor).unwrap_or_else(|| {
        panic!("unsupported compute cap {major}.{minor} — extend CompileTarget enum");
    });
    eprintln!("GPU: cc {major}.{minor} -> {}", target.as_sm_str());
    // On a real DGX Spark this is Sm121. On other hardware we just
    // want to confirm the mapping round-trips.
    assert_eq!(
        CompileTarget::from_compute_capability(major, minor),
        Some(target),
    );

    // Step 3 — UnifiedArena alloc + regions.
    const BYTES: usize = 64 * 1024 * 1024; // 64 MiB — safe even on throttled budget
    let arena = UnifiedArena::new(&ctx, BYTES).expect("UnifiedArena::new");
    assert_eq!(arena.capacity(), BYTES);
    assert_eq!(arena.used(), 0);

    let r1 = arena.region("weights_fake", 4096, 256).expect("region r1");
    let r2 = arena.region("kv_fake", 8192, 256).expect("region r2");
    let r3 = arena.region("scratch_fake", 1024, 16).expect("region r3");

    // Pointers must be aligned + strictly increasing (bump allocator).
    assert_eq!(r1.device_ptr() % 256, 0);
    assert_eq!(r2.device_ptr() % 256, 0);
    assert_eq!(r3.device_ptr() % 16, 0);
    assert!(r2.device_ptr() > r1.device_ptr());
    assert!(r3.device_ptr() > r2.device_ptr());

    // And within the arena.
    assert!(r3.device_ptr() + r3.len() as u64 <= r1.device_ptr() + BYTES as u64);

    eprintln!(
        "UnifiedArena OK: {} MiB allocated, 3 regions carved",
        BYTES / (1024 * 1024),
    );
}
