#!/usr/bin/env python3
# Warp-level MMA THROUGHPUT (not latency) microbench driver.
#
# Runs each of three MMA variants with 8 independent per-warp
# accumulators × 4 warps per CTA, 1 CTA. Reads clock64() deltas, picks
# best-of-10. Reports cycles/MMA at steady-state issue and effective
# per-warp TFLOPS.
#
# The sibling `mma_throughput_bench.py` measures LATENCY (dependent
# accumulator chain). This one fixes that and is the number to trust.

import pathlib, sys
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path("/home/r00t/workspace/upstream/rvllm-serve")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX  = REPO / "kernels" / ARCH / "mma_throughput_bench_indep.ptx"
ITERS = 4096            # must match BENCH_ITERS_INDEP in the .cu
MMA_UNROLL = 8          # must match
NUM_WARPS = 4
THREADS = NUM_WARPS * 32

def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what}: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None

CHECK(drv.cuInit(0), "init")
dev = CHECK(drv.cuDeviceGet(0), "dev")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctx"); CHECK(drv.cuCtxSetCurrent(ctx), "set")
mod = CHECK(drv.cuModuleLoadData(PTX.read_bytes() + b"\0"), "load")

clk_khz = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clk")
num_sms = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "sms")
print(f"SM clock: {clk_khz / 1000:.0f} MHz   SMs: {num_sms}")
print(f"Per block: {NUM_WARPS} warps × 32 lanes = {THREADS} threads")
print(f"Per warp:  {MMA_UNROLL} independent accumulators, {ITERS} total MMAs")
print()
clk_hz = clk_khz * 1000.0

d_cyc = CHECK(drv.cuMemAlloc(8 * NUM_WARPS), "cyc")
d_sink = CHECK(drv.cuMemAlloc(4), "sink")

def run(name: str, fn_name: bytes, k_dim: int):
    fn = CHECK(drv.cuModuleGetFunction(mod, fn_name), f"fn {name}")
    # Warmup.
    params = [np.array([int(d_cyc)],  dtype=np.uint64),
              np.array([int(d_sink)], dtype=np.uint64)]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn, 1,1,1, THREADS,1,1, 0, 0, pp.ctypes.data, 0), "warm")
    CHECK(drv.cuCtxSynchronize(), "sync warm")

    best_cycles = None
    for _ in range(10):
        CHECK(drv.cuMemsetD8(d_cyc, 0, 8 * NUM_WARPS), "zc")
        CHECK(drv.cuLaunchKernel(fn, 1,1,1, THREADS,1,1, 0, 0, pp.ctypes.data, 0), "l")
        CHECK(drv.cuCtxSynchronize(), "s")
        ch = np.empty(NUM_WARPS, dtype=np.uint64)
        CHECK(drv.cuMemcpyDtoH(ch.ctypes.data, d_cyc, 8 * NUM_WARPS), "d2h")
        # Use the max across warps (longest-running warp sets the block time)
        cyc = int(ch.max())
        if best_cycles is None or cyc < best_cycles:
            best_cycles = cyc

    cyc_per_mma = best_cycles / ITERS
    flops_per_mma = 16 * 8 * k_dim * 2  # m × n × k × 2 (mul+add)
    # Per-warp TFLOPS = flops_per_sec / 1e12, where flops_per_sec = flops_per_mma × (clk_hz / cyc_per_mma).
    warp_flops_per_sec = flops_per_mma * (clk_hz / cyc_per_mma)
    # Full device estimate: 4 warp-schedulers per SM × num_sms, assuming
    # this issue rate scales (real workload may differ).
    device_tflops = warp_flops_per_sec * 4 * num_sms / 1e12
    print(f"  {name}:")
    print(f"    cycles per MMA (best of 10, max-warp)  : {cyc_per_mma:.3f}")
    print(f"    flops per MMA                          : {flops_per_mma}")
    print(f"    per-warp TFLOPS                        : {warp_flops_per_sec/1e12:.3f}")
    print(f"    device-wide estimate (4 sched × {num_sms} SM): {device_tflops:.1f} TFLOPS")
    return cyc_per_mma, warp_flops_per_sec

c_f16, tf_f16     = run("f16.f16 m16n8k16",
                        b"mma_bench_indep_f16_k16_kernel", 16)
print()
c_f8f6f4, tf_f8f6 = run("f8f6f4.e4m3.e2m1 m16n8k32",
                        b"mma_bench_indep_f8f6f4_e4m3_e2m1_k32_kernel", 32)
print()
c_mxf4, tf_mxf4   = run("mxf4nvf4.e2m1.e2m1 m16n8k64",
                        b"mma_bench_indep_mxf4nvf4_k64_kernel", 64)
print()

print("=" * 64)
print("Summary — independent accumulators (real throughput, not latency):")
print(f"  f16 k16      : {c_f16:.3f} cycles/MMA  ({tf_f16/1e12:.3f} TFLOPS/warp)")
print(f"  f8f6f4 k32   : {c_f8f6f4:.3f} cycles/MMA  ({tf_f8f6/1e12:.3f} TFLOPS/warp)"
      f"  = {tf_f8f6/tf_f16:.2f}× f16")
print(f"  mxf4nvf4 k64 : {c_mxf4:.3f} cycles/MMA  ({tf_mxf4/1e12:.3f} TFLOPS/warp)"
      f"  = {tf_mxf4/tf_f16:.2f}× f16")
print()
print("Path A (2× f8f6f4 k32 MMAs per K=32 tile, half masked) vs baseline "
      "(2× f16 k16 MMAs per K=32 tile, both live):")
path_a_speedup = tf_f8f6 / tf_f16
print(f"  raw MMA rate ratio: {path_a_speedup:.2f}×")
print(f"  but Path A's 2 MMAs do K=32 each with K=16 masked → effective K/cycle")
print(f"  the same as baseline (both produce 32 K of useful work per 2 MMAs)")
print(f"  net Path A MMA throughput advantage: ~1.00× (smem/BW wins stack)")
print()
print("Path B (1 mxf4nvf4 k64 MMA per K=64 tile) vs baseline (4 f16 k16 MMAs):")
path_b_mma_speedup = (4.0 * c_f16) / (1.0 * c_mxf4)
print(f"  MMA-bound speedup: {path_b_mma_speedup:.2f}× "
      f"(before accounting for Q→e2m1 requant accuracy cost)")
