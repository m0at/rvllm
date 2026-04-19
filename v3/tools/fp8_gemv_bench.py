#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_gemv_bench.py [sm_xxx] [duration_s]
#
# Microbench for the three warp-per-row fp8_gemv variants:
#   * fp8_gemv_blockwise_wpr_kernel        — baseline WPR, scalar decode
#   * fp8_gemv_blockwise_wpr_lut_kernel    — WPR + shared-mem LUT (throttle-friendly)
#   * fp8_gemv_blockwise_wpr_native_kernel — WPR + native cvt.rn.f16x2.e4m3x2 (sm_100+)
#
# For each variant:
#   1. Warm-up (100 launches, discarded)
#   2. Tight loop for `duration_s` seconds
#   3. Per-iter: cuda-event latency in µs, nvidia-smi clocks.sm + power.draw
#   4. Summary: first-second vs last-second mean latency (shows the
#      GB10 firmware clock-regime shift if it exists on this workload)
#
# Shape is Gemma-4-ish: M=1 (decode), N=2048 (a down_proj-ish output dim),
# K=5120 (hidden size). Weight bytes = 10.5 MB per variant — big enough
# to saturate LPDDR5X bandwidth, small enough to not swap.
#
# Output: text table + one JSONL record per iter to bench_<variant>.jsonl.

import os, sys, pathlib, subprocess, json, time
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
PTX = REPO / "kernels" / ARCH / "fp8_gemv.ptx"
OUT_DIR = REPO / "v3" / "tools" / "bench_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shape: M=1 decode token. N/K chosen so weight bytes > L2 cache
# (on GB10 L2 is ~100 MB, so pick 256 MB weight to force LPDDR5X reads
# every iteration).  N=32768, K=8192 → 256 MB of FP8 weight.
M, N, K = 1, 32768, 8192
assert N % 128 == 0 and K % 128 == 0 and N % 8 == 0
BN, BK = N // 128, K // 128

VARIANTS = [
    "fp8_gemv_blockwise_wpr_kernel",
    "fp8_gemv_blockwise_wpr_lut_kernel",
    "fp8_gemv_blockwise_wpr_native_kernel",
]

# -------- CUDA init ---------------------------------------------------------

def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what} failed: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None

CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
CHECK(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

cc_major = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc")
cc_minor = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc")
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name} ({ARCH})")
print(f"shape: M={M} N={N} K={K}, scales [{BN},{BK}], duration {DURATION}s/variant")

# -------- PTX load + alloc --------------------------------------------------

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")

def alloc(bytes_):
    return CHECK(drv.cuMemAlloc(bytes_), "cuMemAlloc")

d_output = alloc(M * N * 4)
d_weight = alloc(N * K)
d_scale  = alloc(BN * BK * 4)
d_input  = alloc(M * K * 4)

def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")

rng = np.random.default_rng(42)
weight = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
weight[weight == 0x7f] = 0x7e
weight[weight == 0xff] = 0xfe
scales = rng.normal(0.0, 0.1, size=(BN, BK)).astype(np.float32)
inp    = rng.normal(size=(M, K)).astype(np.float32)
h2d(d_weight, weight)
h2d(d_scale, scales)
h2d(d_input, inp)

# Kernel params (same for all variants — identical signature)
params = [
    np.array([int(d_output)], dtype=np.uint64),
    np.array([int(d_weight)], dtype=np.uint64),
    np.array([int(d_scale)],  dtype=np.uint64),
    np.array([int(d_input)],  dtype=np.uint64),
    np.array([M], dtype=np.int32),
    np.array([N], dtype=np.int32),
    np.array([K], dtype=np.int32),
    np.array([BK], dtype=np.int32),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

# Grid/block — matches the precision-check setup (8 rows per block.x)
GRID = ((N + 7) // 8, M, 1)
BLOCK = (256, 1, 1)

# -------- nvidia-smi sampler (lightweight, called once per iter) ------------

def smi_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,power.draw",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=0.5)
        if out.returncode != 0:
            return (0, 0.0)
        line = out.stdout.strip().split(",")
        return (int(line[0]), float(line[1]))
    except Exception:
        return (0, 0.0)

# -------- Bench loop --------------------------------------------------------

def bench_variant(entry_name):
    fn = CHECK(drv.cuModuleGetFunction(mod, entry_name.encode()), "cuModuleGetFunction")
    # warmup
    for _ in range(100):
        CHECK(drv.cuLaunchKernel(fn, *GRID, *BLOCK, 0, 0,
                                 param_ptrs.ctypes.data, 0), "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")

    ev_start = CHECK(drv.cuEventCreate(0), "eventCreate")
    ev_end   = CHECK(drv.cuEventCreate(0), "eventCreate")

    records = []
    t0 = time.perf_counter()
    last_smi_t = 0.0
    clock_mhz, power_w = smi_sample()
    while True:
        t_elapsed = time.perf_counter() - t0
        if t_elapsed >= DURATION:
            break
        CHECK(drv.cuEventRecord(ev_start, 0), "eventRecord")
        CHECK(drv.cuLaunchKernel(fn, *GRID, *BLOCK, 0, 0,
                                 param_ptrs.ctypes.data, 0), "launch")
        CHECK(drv.cuEventRecord(ev_end, 0), "eventRecord")
        CHECK(drv.cuEventSynchronize(ev_end), "eventSync")
        kern_ms = CHECK(drv.cuEventElapsedTime(ev_start, ev_end), "elapsedTime")
        # nvidia-smi is expensive (~10 ms) — sample once per 100 ms of
        # wall-clock to avoid dominating the bench
        if t_elapsed - last_smi_t >= 0.1:
            clock_mhz, power_w = smi_sample()
            last_smi_t = t_elapsed
        records.append({
            "t_ms": t_elapsed * 1000.0,
            "kern_us": kern_ms * 1000.0,
            "clocks_sm_mhz": clock_mhz,
            "power_draw_w": power_w,
        })

    CHECK(drv.cuEventDestroy(ev_start), "eventDestroy")
    CHECK(drv.cuEventDestroy(ev_end),   "eventDestroy")
    return records

def summary(name, recs):
    if not recs:
        return
    lats = np.array([r["kern_us"] for r in recs])
    clocks = np.array([r["clocks_sm_mhz"] for r in recs])
    powers = np.array([r["power_draw_w"] for r in recs])
    t_ms = np.array([r["t_ms"] for r in recs])

    # First second vs last second
    first_mask = t_ms < 1000.0
    last_mask = t_ms >= (DURATION - 1.0) * 1000.0
    def stats(mask, label):
        if not mask.any():
            return f"  {label}: (empty)"
        l = lats[mask]; c = clocks[mask]; p = powers[mask]
        return (f"  {label:>12}  iters={int(mask.sum()):5d}  "
                f"lat {l.mean():6.1f}±{l.std():5.1f} µs  "
                f"p50 {np.median(l):6.1f}  p99 {np.percentile(l,99):6.1f}   "
                f"clock {c.mean():6.0f} MHz   power {p.mean():5.1f} W")

    # Effective bandwidth (weight-only read): N*K bytes / latency
    weight_gb = N * K / 1e9
    gbps_p50 = weight_gb / (np.median(lats) / 1e6)

    print(f"\n== {name} == ({len(recs)} iters, {t_ms[-1]/1000.0:.2f}s)")
    print(stats(first_mask, "first 1s"))
    print(stats(last_mask, "last 1s"))
    print(f"  overall p50 {np.median(lats):6.1f} µs  →  eff BW {gbps_p50:5.1f} GB/s  "
          f"(weight = {weight_gb*1000:.1f} MB)")

# -------- Run ---------------------------------------------------------------

all_records = {}
for v in VARIANTS:
    print(f"\n...benching {v}", flush=True)
    try:
        recs = bench_variant(v)
    except SystemExit as e:
        print(f"  SKIP ({e})")
        continue
    all_records[v] = recs
    # Dump JSONL for offline analysis
    out_file = OUT_DIR / f"{v}.jsonl"
    with open(out_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {out_file}")

print("\n" + "=" * 78)
for v, recs in all_records.items():
    summary(v, recs)
print("\n" + "=" * 78)

# Throttle-onset heuristic: find the largest jump in 200ms-window mean
# latency after the first second.
print("\nclock-regime detection (200ms windows, mean latency):")
for v, recs in all_records.items():
    if len(recs) < 50:
        continue
    t_ms = np.array([r["t_ms"] for r in recs])
    lats = np.array([r["kern_us"] for r in recs])
    window_ms = 200.0
    n_windows = int(t_ms[-1] / window_ms)
    means = []
    for w in range(n_windows):
        mask = (t_ms >= w * window_ms) & (t_ms < (w + 1) * window_ms)
        if mask.any():
            means.append((w * window_ms / 1000.0, float(lats[mask].mean())))
    if len(means) >= 6:
        print(f"  {v}:")
        for t_s, m in means[::max(1, len(means)//8)]:
            print(f"    t={t_s:5.2f}s  mean lat {m:6.1f} µs")
