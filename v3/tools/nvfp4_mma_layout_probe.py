#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/nvfp4_mma_layout_probe.py [sm_xxx]
#
# Empirically derives the lane-and-nibble → (row, col) mapping for the
# Blackwell native E2M1 tensor-core MMA
#
#     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
#
# by sweeping A one-hot (and separately B one-hot) against an all-ones
# opposite operand and observing which D[row, col] output lights up.
#
# Register layout:
#   A: 4 × u32 per lane = 16 bytes = 32 e2m1 nibbles per lane
#   B: 2 × u32 per lane =  8 bytes = 16 e2m1 nibbles per lane
#   D: 4 × f32 per lane, standard Ampere-era m16n8k32 D layout:
#       d[0] = D[lane/4,     (lane%4)*2 + 0]
#       d[1] = D[lane/4,     (lane%4)*2 + 1]
#       d[2] = D[lane/4 + 8, (lane%4)*2 + 0]
#       d[3] = D[lane/4 + 8, (lane%4)*2 + 1]
#
# NVFP4 (e2m1) encoding (4 bits): sign(1) + exp(2) + mantissa(1)
#   0.0 → 0b0000 = 0x0
#  +1.0 → 0b0010 = 0x2
#
# For the A-one-hot sweep:
#   A[lane * 4 + reg] has a 0x2 at (byte_pos, nibble_pos); rest zero.
#   B[lane * 2 + *]  = 0x22222222 everywhere (all +1.0).
# ⇒  D[m, n] = sum_k A[m, k] * B[n, k] = 1 for the one m,
#    for every n (independent of n, since B is uniform).
# We then find which row of D is uniformly 1.0 → that's the m_row
# mapped from (lane, reg, byte_pos, nibble_pos).
#
# For the B-one-hot sweep: symmetric — A = all ones, B = one-hot.
# D[m, n] uniform along m, pinned to one n column.

import pathlib
import sys
from collections import defaultdict

import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_121"
PROBE_PTX = REPO / "kernels" / ARCH / "nvfp4_mma_probe.ptx"

if not PROBE_PTX.exists():
    sys.exit(f"missing PTX: {PROBE_PTX}\n  build: kernels/build.sh {ARCH}")


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

mod = CHECK(drv.cuModuleLoadData(PROBE_PTX.read_bytes() + b"\0"), "load probe")
fn  = CHECK(drv.cuModuleGetFunction(mod, b"nvfp4_mma_v2_probe_kernel"),
            "get v2 probe fn")

# E4M3 +1.0 = 0x38 (sign=0, exp=0111, mant=000 — value = 2^0 * 1). Four
# packed into a u32 gives scale=1.0 across all 4 per-16 scale slots.
SF_ONE_U32 = 0x38383838


# Allocate device buffers — reuse across all sweeps.
A_BYTES  = 32 * 4 * 4   # 32 lanes × 4 u32
B_BYTES  = 32 * 2 * 4   # 32 lanes × 2 u32
SF_BYTES = 32 * 4       # 32 lanes × 1 u32 (= 4 × E4M3 scales)
D_BYTES  = 32 * 4 * 4   # 32 lanes × 4 f32
d_a   = CHECK(drv.cuMemAlloc(A_BYTES),  "alloc a")
d_b   = CHECK(drv.cuMemAlloc(B_BYTES),  "alloc b")
d_sfa = CHECK(drv.cuMemAlloc(SF_BYTES), "alloc sfa")
d_sfb = CHECK(drv.cuMemAlloc(SF_BYTES), "alloc sfb")
d_d   = CHECK(drv.cuMemAlloc(D_BYTES),  "alloc d")

# Scales stay = 1.0 for the layout sweep — we're probing the e2m1
# data layout, not the scale layout.
sfa_ones = np.full(32, SF_ONE_U32, dtype=np.uint32)
sfb_ones = np.full(32, SF_ONE_U32, dtype=np.uint32)
CHECK(drv.cuMemcpyHtoD(d_sfa, sfa_ones.ctypes.data, SF_BYTES), "sfa H2D")
CHECK(drv.cuMemcpyHtoD(d_sfb, sfb_ones.ctypes.data, SF_BYTES), "sfb H2D")


def run_probe(a_u32: np.ndarray, b_u32: np.ndarray) -> np.ndarray:
    """Returns the full D [16, 8] f32 tile via the standard m16n8 D
    lane-mapping (same as fp16/fp8 m16n8kXX)."""
    CHECK(drv.cuMemcpyHtoD(d_a, a_u32.ctypes.data, A_BYTES), "a H2D")
    CHECK(drv.cuMemcpyHtoD(d_b, b_u32.ctypes.data, B_BYTES), "b H2D")
    CHECK(drv.cuMemsetD8(d_d, 0, D_BYTES), "zero d")
    params = [
        np.array([int(d_a)],   dtype=np.uint64),
        np.array([int(d_b)],   dtype=np.uint64),
        np.array([int(d_sfa)], dtype=np.uint64),
        np.array([int(d_sfb)], dtype=np.uint64),
        np.array([int(d_d)],   dtype=np.uint64),
    ]
    pp = np.array([p.ctypes.data for p in params], dtype=np.uint64)
    CHECK(drv.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, pp.ctypes.data, 0),
          "probe launch")
    CHECK(drv.cuCtxSynchronize(), "sync")
    d_host = np.empty(32 * 4, dtype=np.float32)
    CHECK(drv.cuMemcpyDtoH(d_host.ctypes.data, d_d, D_BYTES), "d D2H")
    # Decode to [16, 8] via the standard layout.
    D = np.zeros((16, 8), dtype=np.float32)
    for lane in range(32):
        m_top = lane // 4
        col0  = (lane % 4) * 2
        D[m_top,     col0    ] = d_host[lane * 4 + 0]
        D[m_top,     col0 + 1] = d_host[lane * 4 + 1]
        D[m_top + 8, col0    ] = d_host[lane * 4 + 2]
        D[m_top + 8, col0 + 1] = d_host[lane * 4 + 3]
    return D


def make_b_all_ones_fp4() -> np.ndarray:
    """B = 16 × 2 u32 per lane, all bytes 0x22 (every e2m1 = +1.0)."""
    return np.full(32 * 2, 0x22222222, dtype=np.uint32)


def make_a_all_ones_fp4() -> np.ndarray:
    return np.full(32 * 4, 0x22222222, dtype=np.uint32)


def sanity_check():
    """MMA(all-ones, all-ones, scale=1) should produce K=64 in every
    D cell (m16n8k64 with e2m1 values all = 1.0 and ue4m3 scales = 1.0
    → sum_{k=0..63} 1 * 1 * 1 * 1 = 64)."""
    a = make_a_all_ones_fp4()
    b = make_b_all_ones_fp4()
    D = run_probe(a, b)
    uniq = np.unique(D)
    print(f"[sanity] all-ones × all-ones → D unique values = {uniq}")
    if len(uniq) == 1 and abs(uniq[0] - 64.0) < 1e-3:
        print(f"[sanity]   OK — effective K = 64")
        return 64
    print(f"[sanity]   UNEXPECTED — expected 64 uniformly")
    return None


# Build one-hot A: lane l gets reg_index r with 0x2 at (byte, nibble).
def build_a_one_hot(lane: int, reg: int, byte_pos: int, nibble: int) -> np.ndarray:
    a = np.zeros(32 * 4, dtype=np.uint32)
    bit_offset = byte_pos * 8 + nibble * 4
    a[lane * 4 + reg] = 0x2 << bit_offset
    return a


def build_b_one_hot(lane: int, reg: int, byte_pos: int, nibble: int) -> np.ndarray:
    b = np.zeros(32 * 2, dtype=np.uint32)
    bit_offset = byte_pos * 8 + nibble * 4
    b[lane * 2 + reg] = 0x2 << bit_offset
    return b


def classify_d(D: np.ndarray, tol: float = 1e-3):
    """For a one-hot input, D should have either (a) one row of 1.0s
    (A-sweep: D[m_row, :] = 1.0 for the matched row, else 0), or
    (b) one column of 1.0s (B-sweep: D[:, n_col] = 1.0). We also
    allow partial matches for unusual layouts."""
    nonzero = np.abs(D) > tol
    row_sums = nonzero.sum(axis=1)  # per-row hit count
    col_sums = nonzero.sum(axis=0)  # per-col hit count
    return {
        "total_nonzero": int(nonzero.sum()),
        "hit_rows":      [int(m) for m in np.flatnonzero(row_sums > 0)],
        "hit_cols":      [int(n) for n in np.flatnonzero(col_sums > 0)],
        "row_sums":      row_sums.tolist(),
        "col_sums":      col_sums.tolist(),
        "row_max_sum":   int(row_sums.max()),
        "col_max_sum":   int(col_sums.max()),
    }


def sweep_a_one_hot(effective_K: int):
    """Sweep A-nibble one-hot across (lane, reg, byte_pos, nibble). For
    each, expect D to have exactly one row with every column = 1 (sum
    over K is 1 since only one A element is nonzero and B = all ones).
    Record (lane, reg, byte, nibble) → m_row."""
    b_all = make_b_all_ones_fp4()
    mapping = {}  # (lane, reg, byte, nibble) → m_row (int) or None
    anomalies = []
    for lane in range(32):
        for reg in range(4):
            for byte_pos in range(4):
                for nibble in range(2):
                    a = build_a_one_hot(lane, reg, byte_pos, nibble)
                    D = run_probe(a, b_all)
                    info = classify_d(D)
                    key = (lane, reg, byte_pos, nibble)
                    # Expected: exactly ONE row has 8 columns equal to 1.0
                    # (D[m, n] = 1 for the matched m, all n).
                    if (info["total_nonzero"] == 8
                        and len(info["hit_rows"]) == 1
                        and info["hit_cols"] == list(range(8))):
                        m_row = info["hit_rows"][0]
                        val = float(D[m_row, 0])
                        mapping[key] = (m_row, val)
                    else:
                        anomalies.append((key, info))
                        mapping[key] = None
    return mapping, anomalies


def sweep_b_one_hot():
    a_all = make_a_all_ones_fp4()
    mapping = {}
    anomalies = []
    for lane in range(32):
        for reg in range(2):
            for byte_pos in range(4):
                for nibble in range(2):
                    b = build_b_one_hot(lane, reg, byte_pos, nibble)
                    D = run_probe(a_all, b)
                    info = classify_d(D)
                    key = (lane, reg, byte_pos, nibble)
                    # Expected: exactly ONE col has 16 rows equal to 1.0.
                    if (info["total_nonzero"] == 16
                        and len(info["hit_cols"]) == 1
                        and info["hit_rows"] == list(range(16))):
                        n_col = info["hit_cols"][0]
                        val = float(D[0, n_col])
                        mapping[key] = (n_col, val)
                    else:
                        anomalies.append((key, info))
                        mapping[key] = None
    return mapping, anomalies


# --- main ---

print(f"device PTX: {PROBE_PTX.name}")
print()

K_eff = sanity_check()
print()

print("=== A one-hot sweep (1024 positions) ===")
a_map, a_anom = sweep_a_one_hot(K_eff or 64)
print(f"  mapped: {sum(1 for v in a_map.values() if v is not None)}/1024")
print(f"  anomalies: {len(a_anom)}")
if a_anom[:5]:
    print("  first 5 anomalies (key, total_nonzero, hit_rows, hit_cols):")
    for key, info in a_anom[:5]:
        print(f"    {key} -> nz={info['total_nonzero']} "
              f"rows={info['hit_rows']} cols={info['hit_cols']}")
print()

print("=== B one-hot sweep (512 positions) ===")
b_map, b_anom = sweep_b_one_hot()
print(f"  mapped: {sum(1 for v in b_map.values() if v is not None)}/512")
print(f"  anomalies: {len(b_anom)}")
if b_anom[:5]:
    print("  first 5 anomalies (key, total_nonzero, hit_rows, hit_cols):")
    for key, info in b_anom[:5]:
        print(f"    {key} -> nz={info['total_nonzero']} "
              f"rows={info['hit_rows']} cols={info['hit_cols']}")
print()

# Summarize A mapping by m_row. How many (lane, reg, byte, nibble)
# positions hit each m? Expected: each row m has 1024/16 = 64 positions.
if all(v is not None for v in a_map.values()):
    per_row_a = defaultdict(list)
    for key, (m, v) in a_map.items():
        per_row_a[m].append((key, v))
    print("A → m_row distribution (should be 64 positions per row):")
    for m in sorted(per_row_a):
        entries = per_row_a[m]
        vals = set(round(v, 2) for _, v in entries)
        print(f"  m={m:2d}: {len(entries):3d} positions, values={sorted(vals)}")
    print()

if all(v is not None for v in b_map.values()):
    per_col_b = defaultdict(list)
    for key, (n, v) in b_map.items():
        per_col_b[n].append((key, v))
    print("B → n_col distribution (should be 64 positions per col):")
    for n in sorted(per_col_b):
        entries = per_col_b[n]
        vals = set(round(v, 2) for _, v in entries)
        print(f"  n={n:2d}: {len(entries):3d} positions, values={sorted(vals)}")
    print()

# Dump the A mapping, grouped by lane, as a compact table — this is
# what the packer rewrite will key off.
if all(v is not None for v in a_map.values()):
    print("=== A lane-layout table (m_row per nibble-position) ===")
    print("      reg0            reg1            reg2            reg3")
    print("      b0l b0h b1l b1h b2l b2h b3l b3h ...")
    for lane in range(32):
        cells = []
        for reg in range(4):
            row = []
            for byte_pos in range(4):
                for nibble in range(2):
                    mv = a_map[(lane, reg, byte_pos, nibble)]
                    row.append(str(mv[0]) if mv else "?")
            cells.append(" ".join(f"{r:>2}" for r in row))
        print(f"  L{lane:2d}: " + "  ".join(cells))
    print()

if all(v is not None for v in b_map.values()):
    print("=== B lane-layout table (n_col per nibble-position) ===")
    for lane in range(32):
        cells = []
        for reg in range(2):
            row = []
            for byte_pos in range(4):
                for nibble in range(2):
                    nv = b_map[(lane, reg, byte_pos, nibble)]
                    row.append(str(nv[0]) if nv else "?")
            cells.append(" ".join(f"{r:>2}" for r in row))
        print(f"  L{lane:2d}: " + "  ".join(cells))

print()
print("sweep complete")
