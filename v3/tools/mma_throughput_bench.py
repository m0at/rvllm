#!/usr/bin/env python3
# Warp-level MMA throughput microbench driver. Runs each of three MMA
# variants (f16 k16, f8f6f4 k32 e4m3.e2m1, mxf4nvf4 k64) in a tight
# single-warp loop, reads clock64() deltas, reports cycles/MMA and
# effective TFLOPS.

import pathlib, sys
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path("/home/r00t/workspace/upstream/rvllm-serve")
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX  = REPO / "kernels" / ARCH / "mma_throughput_bench.ptx"
ITERS = 8192  # must match BENCH_ITERS in the .cu

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

# SM clock rate (Hz) for TFLOPS conversion. Read from device props.
clk_khz = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clk")
num_sms = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "sms")
print(f"SM clock: {clk_khz / 1000:.0f} MHz   SMs: {num_sms}")
clk_hz = clk_khz * 1000.0
print()

d_cyc = CHECK(drv.cuMemAlloc(8), "cyc")
d_sink = CHECK(drv.cuMemAlloc(4), "sink")

def run(name: str, fn_name: bytes, k_dim: int):
    fn = CHECK(drv.cuModuleGetFunction(mod, fn_name), f"fn {name}")
    CHECK(drv.cuMemsetD8(d_cyc, 0, 8), "zc")
    params = [np.array([int(d_cyc)],  dtype=np.uint64),
              np.array([int(d_sink)], dtype=np.uint64)]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    # Warmup.
    CHECK(drv.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, pp.ctypes.data, 0), "warm")
    CHECK(drv.cuCtxSynchronize(), "sync warm")
    # Measured run.
    best_cycles = None
    for _ in range(10):
        CHECK(drv.cuMemsetD8(d_cyc, 0, 8), "zc")
        CHECK(drv.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, pp.ctypes.data, 0), "l")
        CHECK(drv.cuCtxSynchronize(), "s")
        ch = np.empty(1, dtype=np.uint64)
        CHECK(drv.cuMemcpyDtoH(ch.ctypes.data, d_cyc, 8), "d2h")
        cyc = int(ch[0])
        if best_cycles is None or cyc < best_cycles:
            best_cycles = cyc
    cyc_per_mma = best_cycles / ITERS
    # One MMA per warp = m16n8kK → 16*8*K*2 = 256*K fp32 flops.
    flops_per_mma = 16 * 8 * k_dim * 2
    # Per-warp throughput.
    warp_flops_per_sec = flops_per_mma * (clk_hz / cyc_per_mma)
    # Hardware has 4 warp-schedulers per SM → theoretical peak is 4× warps.
    sm_flops_per_sec = warp_flops_per_sec * 4
    device_tflops = sm_flops_per_sec * num_sms / 1e12
    print(f"  {name}:")
    print(f"    cycles per MMA (best of 10) : {cyc_per_mma:.2f}")
    print(f"    flops per MMA               : {flops_per_mma}")
    print(f"    per-warp TFLOPS             : {warp_flops_per_sec/1e12:.3f}")
    print(f"    per-SM (4 schedulers) TFLOPS: {sm_flops_per_sec/1e12:.3f}")
    print(f"    device-wide estimate        : {device_tflops:.1f} TFLOPS")
    return cyc_per_mma

c_f16   = run("f16.f16 m16n8k16",           b"mma_bench_f16_k16_kernel",              16)
print()
c_f8f6f4 = run("f8f6f4.e4m3.e2m1 m16n8k32", b"mma_bench_f8f6f4_e4m3_e2m1_k32_kernel", 32)
print()
c_mxf4   = run("mxf4nvf4.e2m1.e2m1 m16n8k64", b"mma_bench_mxf4nvf4_k64_kernel",        64)
print()

print("=" * 60)
print("Summary (single-warp issue rate, register-resident operands):")
print(f"  f16 k16   : {c_f16:.2f} cycles/MMA")
print(f"  f8f6f4 k32: {c_f8f6f4:.2f} cycles/MMA ({c_f16/c_f8f6f4:.2f}× issue rate vs f16)")
print(f"  mxf4nvf4 k64: {c_mxf4:.2f} cycles/MMA ({c_f16/c_mxf4:.2f}× issue rate vs f16)")
print()
print("Path A (2× f8f6f4 per K=32 tile) vs baseline (2× f16 k16 per K=32 tile):")
ratio = (2 * c_f16) / (2 * c_f8f6f4)
print(f"  = {ratio:.2f}× MMA-bound speedup (before counting smem-bandwidth saving)")
