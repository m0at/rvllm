#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_mma_vs_scalar_check.py [sm_xxx] [prompt_len] [head_dim]
#
# The F6 bisect showed scalar vs MMA produce different outputs from
# layer 1 onward on the live Gemma 4 31B model, even though both
# paths pass `fa2_unified_prefill_check.py` to within 5e-2 of an fp64
# reference. The harness checked each path against fp64 SEPARATELY,
# so it couldn't catch "both paths OK individually but differ from
# each other by enough to flip downstream softmax". This harness runs
# the kernel TWICE with the same inputs — once with use_mma=0, once
# with use_mma=1 — and reports the max-abs / scale_rel difference
# between the two kernel outputs.
#
# If the delta is < ~1e-4 per element the per-layer drift is
# f32-rounding-order noise (mitigable with accumulator tricks).
# If ≥ 1e-3 something in the MMA operand packing / scale fold is
# wrong.

import sys, pathlib
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "flash_attention_unified_prefill.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}")


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
print(f"PTX: {PTX.name}")
ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(
    mod, b"flash_attention_2_prefill_fp8kv_unified_kernel"),
    "cuModuleGetFunction")


FP8_MAX = 448.0


def f32_to_e4m3(v):
    if v == 0.0:
        return 0
    sign = 1 if v < 0.0 else 0
    a = abs(v)
    if a > FP8_MAX: a = FP8_MAX
    e = 0; m = a
    while m >= 2.0: m /= 2.0; e += 1
    while m < 1.0 and e > -6: m *= 2.0; e -= 1
    exp_bits = e + 7
    if exp_bits < 0: return (sign << 7)
    if exp_bits > 15: exp_bits = 15
    mant = int(round((m - 1.0) * 8))
    if mant == 8: mant = 0; exp_bits += 1
    if exp_bits > 15: exp_bits, mant = 15, 7
    return (sign << 7) | ((exp_bits & 0xF) << 3) | (mant & 0x7)


prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 128
head_dim   = int(sys.argv[3]) if len(sys.argv) > 3 else 256
num_heads     = 32
num_kv_heads  = 16
block_size    = 32
tile_size     = 32 if head_dim <= 256 else 16
sliding_window = 0
scale         = 1.0 / np.sqrt(head_dim)
num_queries_per_kv = num_heads // num_kv_heads
BLOCK_M       = 16
block_q       = BLOCK_M // num_queries_per_kv
max_blocks_per_seq = (prompt_len + block_size - 1) // block_size
num_blocks = max_blocks_per_seq
print(f"shape: prompt_len={prompt_len}, head_dim={head_dim}, tile={tile_size}")

rng = np.random.default_rng(11)
Q_f32 = rng.normal(0, 1, (prompt_len, num_heads, head_dim)).astype(np.float32)
K_f32 = rng.normal(0, 1, (num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
V_f32 = rng.normal(0, 1, (num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)

q_scale = np.zeros((prompt_len, num_heads), dtype=np.float32)
Q_bytes = np.zeros((prompt_len, num_heads, head_dim), dtype=np.uint8)
for t in range(prompt_len):
    for h in range(num_heads):
        amax = max(float(np.max(np.abs(Q_f32[t, h]))), 1e-12)
        q_scale[t, h] = amax / FP8_MAX
        Q_bytes[t, h] = [f32_to_e4m3(float(v / q_scale[t, h])) for v in Q_f32[t, h]]
total_slots = num_blocks * block_size
k_scale = np.zeros((total_slots, num_kv_heads), dtype=np.float32)
v_scale = np.zeros((total_slots, num_kv_heads), dtype=np.float32)
K_bytes = np.zeros_like(K_f32, dtype=np.uint8)
V_bytes = np.zeros_like(V_f32, dtype=np.uint8)
for b in range(num_blocks):
    for o in range(block_size):
        slot = b * block_size + o
        for kh in range(num_kv_heads):
            ka = max(float(np.max(np.abs(K_f32[b, o, kh]))), 1e-12)
            va = max(float(np.max(np.abs(V_f32[b, o, kh]))), 1e-12)
            k_scale[slot, kh] = ka / FP8_MAX
            v_scale[slot, kh] = va / FP8_MAX
            K_bytes[b, o, kh] = [f32_to_e4m3(float(x / k_scale[slot, kh])) for x in K_f32[b, o, kh]]
            V_bytes[b, o, kh] = [f32_to_e4m3(float(x / v_scale[slot, kh])) for x in V_f32[b, o, kh]]

block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, max_blocks_per_seq)
context_lens = np.array([prompt_len], dtype=np.int32)
cu_seqlens_q = np.array([0, prompt_len], dtype=np.int32)


def alloc(n): return CHECK(drv.cuMemAlloc(n), "cuMemAlloc")


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


d_out = alloc(prompt_len * num_heads * head_dim * 2)
d_q = alloc(Q_bytes.nbytes); h2d(d_q, Q_bytes)
d_k = alloc(K_bytes.nbytes); h2d(d_k, K_bytes)
d_v = alloc(V_bytes.nbytes); h2d(d_v, V_bytes)
d_ks = alloc(k_scale.nbytes); h2d(d_ks, k_scale)
d_vs = alloc(v_scale.nbytes); h2d(d_vs, v_scale)
d_qs = alloc(q_scale.nbytes); h2d(d_qs, q_scale)
d_bt = alloc(block_tables.nbytes); h2d(d_bt, block_tables)
d_cu = alloc(cu_seqlens_q.nbytes); h2d(d_cu, cu_seqlens_q)
d_cl = alloc(context_lens.nbytes); h2d(d_cl, context_lens)
d_one = alloc(4); h2d(d_one, np.array([1.0], dtype=np.float32))

FA2_THREADS = 128
smem_bytes = (BLOCK_M * head_dim * 1 + BLOCK_M * 4
    + head_dim * tile_size * 1 + tile_size * head_dim * 1
    + tile_size * head_dim * 1
    + tile_size * 4 + tile_size * 4
    + BLOCK_M * tile_size * 4
    + BLOCK_M * 4 * 3
    + BLOCK_M * tile_size * 1 + BLOCK_M * 4
    + BLOCK_M * head_dim * 4
    + (FA2_THREADS // 32) * 4
    + 128)
if smem_bytes >= 48 * 1024:
    CHECK(drv.cuFuncSetAttribute(
        fn,
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes), "cuFuncSetAttribute")

total_num_q_blocks = prompt_len // block_q + 1
window_size_left = -1 if sliding_window <= 0 else sliding_window


def run(use_mma):
    # Zero output first
    CHECK(drv.cuMemsetD8(d_out, 0, prompt_len * num_heads * head_dim * 2), "memset")
    params = [
        np.array([int(d_out)], dtype=np.uint64),
        np.array([int(d_q)], dtype=np.uint64),
        np.array([int(d_k)], dtype=np.uint64),
        np.array([int(d_v)], dtype=np.uint64),
        np.array([int(d_ks)], dtype=np.uint64),
        np.array([int(d_vs)], dtype=np.uint64),
        np.array([int(d_qs)], dtype=np.uint64),
        np.array([int(d_one)], dtype=np.uint64),
        np.array([int(d_one)], dtype=np.uint64),
        np.array([int(d_bt)], dtype=np.uint64),
        np.array([int(d_cu)], dtype=np.uint64),
        np.array([int(d_cl)], dtype=np.uint64),
        np.array([int(d_one)], dtype=np.uint64),
        np.array([scale], dtype=np.float32),
        np.array([num_heads], dtype=np.int32),
        np.array([num_kv_heads], dtype=np.int32),
        np.array([head_dim], dtype=np.int32),
        np.array([block_size], dtype=np.int32),
        np.array([max_blocks_per_seq], dtype=np.int32),
        np.array([tile_size], dtype=np.int32),
        np.array([num_queries_per_kv], dtype=np.int32),
        np.array([block_q], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([window_size_left], dtype=np.int32),
        np.array([use_mma], dtype=np.int32),
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn, total_num_q_blocks, num_kv_heads, 1,
        FA2_THREADS, 1, 1, smem_bytes, 0, pp.ctypes.data, 0), "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")
    out = np.empty((prompt_len, num_heads, head_dim), dtype=np.float16)
    CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "DtoH")
    return out.astype(np.float32)


scalar = run(0)
mma = run(1)
diff = np.abs(mma - scalar)
scale_rel = diff / np.maximum(np.abs(scalar), 1e-30)

# Also compute per-(tok, head) max diff
per_tok_head_max = diff.reshape(prompt_len * num_heads, head_dim).max(axis=1)

print(f"scalar range: [{scalar.min():+.3e}, {scalar.max():+.3e}]")
print(f"mma    range: [{mma.min():+.3e}, {mma.max():+.3e}]")
print(f"|MMA - SCALAR|: max {diff.max():.3e}  mean {diff.mean():.3e}")
print(f"|MMA - SCALAR| / |scalar|: max {scale_rel.max():.3e}  mean {scale_rel.mean():.3e}")
print(f"non-zero diff entries: {(diff > 0).sum()} / {diff.size} "
      f"({100*(diff > 0).sum() / diff.size:.1f}%)")
if diff.max() < 1e-5:
    print("\n  OK — MMA matches scalar within f32-ULP. Per-layer drift should be tiny.")
elif diff.max() < 1e-3:
    print("\n  ACCUMULATION-ORDER noise. Small enough to potentially compound safely.")
else:
    print("\n  STRUCTURAL — MMA differs from scalar more than rounding.")
    print("  Check operand packing, scale fold, and mask semantics.")
