#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fa2_precision_check.py \
#       [sm_xxx] [head_dim] [context_len]
#
# Validates `flash_attention_2_decode_kernel` in
# kernels/<sm_xxx>/flash_attention.ptx against a naive fp64 attention
# reference. Closes the numerical validation gap for the sm_121
# `FA2_BC = 32` change: the arch-conditional tile-width was only
# compile-verified so far; this harness runs the kernel on GB10 and
# checks the output against an independent reference.
#
# Layout (one decode token per sequence):
#   query:          [num_seqs, num_heads, head_dim]  f32
#   key_cache:      [num_blocks, block_size, num_kv_heads, head_dim]  f32
#   value_cache:    [num_blocks, block_size, num_kv_heads, head_dim]  f32
#   block_tables:   [num_seqs, max_blocks_per_seq]   i32
#   context_lens:   [num_seqs]                        i32
#
# Reference per (seq, head):
#   scores[t] = (Q . K[t]) * scale
#   soft[t]   = softmax(scores)                       (no causal mask needed —
#                                                      decode token attends to
#                                                      the entire context)
#   out[]     = Σ_t soft[t] * V[t]
#
# Pass criteria:
#   max |out_kernel - out_ref| / mean(|out_ref|)  <= 1e-3

import sys, pathlib
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PTX = REPO / "kernels" / ARCH / "flash_attention.ptx"
if not PTX.exists():
    sys.exit(f"missing PTX: {PTX}  (build with: kernels/build.sh {ARCH})")

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

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
fn = CHECK(drv.cuModuleGetFunction(mod, b"flash_attention_2_decode_kernel"),
           "cuModuleGetFunction")

# -------- Test shape --------------------------------------------------------

num_seqs      = 1
num_heads     = 8
num_kv_heads  = 8      # no GQA; pure MHA
head_dim      = int(sys.argv[2]) if len(sys.argv) > 2 else 128
block_size    = 16
context_len   = int(sys.argv[3]) if len(sys.argv) > 3 else 64
max_blocks_per_seq = (context_len + block_size - 1) // block_size
num_blocks    = max_blocks_per_seq * num_seqs
scale         = 1.0 / np.sqrt(head_dim)
print(f"shape: seqs={num_seqs}, heads={num_heads}, head_dim={head_dim}, "
      f"ctx_len={context_len}, blocks={num_blocks}x{block_size}")

rng = np.random.default_rng(42)
Q = rng.normal(0.0, 1.0, size=(num_seqs, num_heads, head_dim)).astype(np.float32)
K = rng.normal(0.0, 1.0, size=(num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
V = rng.normal(0.0, 1.0, size=(num_blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)

block_tables = np.arange(num_blocks, dtype=np.int32).reshape(num_seqs, max_blocks_per_seq)
context_lens = np.full((num_seqs,), context_len, dtype=np.int32)

# -------- Reference ---------------------------------------------------------
# Naive attention in fp64 for the sole decode query per (seq, head).
out_ref = np.zeros((num_seqs, num_heads, head_dim), dtype=np.float64)
for s in range(num_seqs):
    for h in range(num_heads):
        kh = h if num_kv_heads == num_heads else h // (num_heads // num_kv_heads)
        # Gather [context_len, head_dim] K and V via block_tables.
        k_gathered = np.zeros((context_len, head_dim), dtype=np.float64)
        v_gathered = np.zeros((context_len, head_dim), dtype=np.float64)
        for t in range(context_len):
            b = block_tables[s, t // block_size]
            off = t % block_size
            k_gathered[t] = K[b, off, kh]
            v_gathered[t] = V[b, off, kh]
        q = Q[s, h].astype(np.float64)
        scores = (k_gathered @ q) * scale
        m = scores.max()
        p = np.exp(scores - m)
        p /= p.sum()
        out_ref[s, h] = p @ v_gathered
out_ref = out_ref.astype(np.float32)

# -------- GPU launch --------------------------------------------------------

def alloc(bytes_):
    return CHECK(drv.cuMemAlloc(bytes_), "cuMemAlloc")

def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")

d_out = alloc(num_seqs * num_heads * head_dim * 4)
d_q   = alloc(Q.nbytes); h2d(d_q, Q)
d_k   = alloc(K.nbytes); h2d(d_k, K)
d_v   = alloc(V.nbytes); h2d(d_v, V)
d_bt  = alloc(block_tables.nbytes); h2d(d_bt, block_tables)
d_cl  = alloc(context_lens.nbytes); h2d(d_cl, context_lens)

# Shared-memory size (must match the kernel's layout). Under sm_121 the
# `FA2_BC = 32` arch gate applies, so compute accordingly for the
# expected allocation; the kernel's fixed `extern __shared__` block
# only reads this pointer, not the size constant.
FA2_BC      = 32 if ARCH in ("sm_100", "sm_121", "sm_122") else 64
FA2_THREADS = 128
smem_bytes = (
    2 * FA2_BC * head_dim * 4          # K tile + V tile
    + FA2_BC * 4                       # scores
    + (FA2_THREADS // 32) * 4          # block-level reduce
)
print(f"smem request: {smem_bytes} bytes (FA2_BC={FA2_BC}, head_dim={head_dim})")

# Smem requests above the static ceiling (48 KB on most arches, lower on
# some Blackwell consumer variants) require opt-in via
# cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES). Always set it —
# the driver treats it as a max, not a fixed size. The runtime does
# this too before launching FA; mirroring keeps the harness honest.
if smem_bytes >= 48 * 1024:
    CHECK(drv.cuFuncSetAttribute(
        fn,
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes,
    ), "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)")

# Kernel params (match the __global__ signature order exactly)
params = [
    np.array([int(d_out)], dtype=np.uint64),
    np.array([int(d_q)],   dtype=np.uint64),
    np.array([int(d_k)],   dtype=np.uint64),
    np.array([int(d_v)],   dtype=np.uint64),
    np.array([int(d_bt)],  dtype=np.uint64),
    np.array([int(d_cl)],  dtype=np.uint64),
    np.array([scale],            dtype=np.float32),
    np.array([num_heads],        dtype=np.int32),
    np.array([num_kv_heads],     dtype=np.int32),
    np.array([head_dim],         dtype=np.int32),
    np.array([block_size],       dtype=np.int32),
    np.array([max_blocks_per_seq], dtype=np.int32),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(
    fn,
    num_seqs, num_heads, 1,
    FA2_THREADS, 1, 1,
    smem_bytes,                # shared mem
    0,                         # default stream
    param_ptrs.ctypes.data,
    0,
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

out = np.empty((num_seqs, num_heads, head_dim), dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "DtoH")

# -------- Compare -----------------------------------------------------------

abs_err = np.abs(out - out_ref)
ref_mean_abs = float(np.abs(out_ref).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

print(f"ref    range: [{out_ref.min():+.4e}, {out_ref.max():+.4e}], "
      f"|ref| mean {ref_mean_abs:.4e}")
print(f"kernel range: [{out.min():+.4e}, {out.max():+.4e}]")
print(f"abs_err:   max {abs_err.max():.4e}  mean {abs_err.mean():.4e}")
print(f"scale_rel: max {scale_rel.max():.4e}  mean {scale_rel.mean():.4e}   "
      f"(abs_err / mean|ref|)")

# Worst 3 (seq, head) pairs
worst = np.argsort(abs_err.reshape(num_seqs * num_heads, -1).max(axis=1))[-3:][::-1]
print("worst (seq, head): " +
      ", ".join(f"({w // num_heads},{w % num_heads}): "
                f"{abs_err.reshape(num_seqs*num_heads,-1)[w].max():.3e}"
                for w in worst))

THRESHOLD = 1e-3
if scale_rel.max() > THRESHOLD:
    print(f"\nFAIL: max scale_rel {scale_rel.max():.4e} > {THRESHOLD:.0e}")
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.4e} <= {THRESHOLD:.0e}")
