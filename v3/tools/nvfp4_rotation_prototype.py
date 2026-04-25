#!/usr/bin/env python3
"""
NVFP4 Hadamard-rotation prototype (offline analysis on shadow dump).

Reads /tmp/nvfp4_shadow/, replays NVFP4 quant in pure Python, then
applies QuaRot-style orthogonal rotations along head_dim and measures
whether the resulting per-(layer, kv_head) error / attention quality
improves enough to justify a CUDA kernel implementation.

Pure numpy + scipy.linalg.hadamard. No torch, no GPU.

Usage:
    /home/r00t/.venv/bin/python3 v3/tools/nvfp4_rotation_prototype.py

Outputs:
    /tmp/rotation_prototype_report.md
    + one-screen summary on stdout.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard as scipy_hadamard

# Re-use helpers from the existing analyzer (no copy/paste).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_shadow_analyze import (  # type: ignore
    E2M1_LUT,
    E2M1_MAGS,
    e4m3_to_f32,
    load_layer_q,
    rel_err,
    softmax,
    topk_overlap,
    unpack_e2m1_u8,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DUMP_DIR = Path("/tmp/nvfp4_shadow")
REPORT_PATH = Path("/tmp/rotation_prototype_report.md")
NUM_RANDOM_ORTH_SEEDS = 4
NUM_SIGNED_HADAMARD_SEEDS = 8
RNG_BASE_SEED = 0xC0FFEE
ORACLE_SCALE_GRID = 16  # candidates between [amax/12, amax/3]

# E4M3 representable scale set, precomputed once for "round to E4M3".
def _e4m3_grid() -> np.ndarray:
    bytes_ = np.arange(256, dtype=np.uint8)
    vals = e4m3_to_f32(bytes_)
    pos = vals[vals > 0]
    return np.unique(pos)

E4M3_POS_VALUES = _e4m3_grid()


def round_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Snap positive scalars to nearest E4M3 representable value
    (in log space — closest fp8 magnitude). x.shape arbitrary."""
    out = np.zeros_like(x, dtype=np.float32)
    mask = x > 0
    if not np.any(mask):
        return out
    pos = x[mask].astype(np.float32)
    # nearest in linear space matches kernel's __nv_fp8_e4m3(s) cast
    # well enough for our purposes (kernel uses RTE in linear space).
    idx = np.searchsorted(E4M3_POS_VALUES, pos)
    idx = np.clip(idx, 1, len(E4M3_POS_VALUES) - 1)
    left = E4M3_POS_VALUES[idx - 1]
    right = E4M3_POS_VALUES[idx]
    pick_right = (right - pos) < (pos - left)
    out_pos = np.where(pick_right, right, left)
    out[mask] = out_pos
    return out


# ----------------------------------------------------------------------
# Python NVFP4 clone (matches kernel: scale = amax/6 → E4M3, encode E2M1)
# ----------------------------------------------------------------------

def _nearest_e2m1_idx(scaled_abs: np.ndarray) -> np.ndarray:
    """Pick nearest LUT magnitude (0,0.5,1,1.5,2,3,4,6) for |scaled|.
    Mirrors the kernel's strict-less-than threshold ladder (fp4_encode):
        mag < 0.25 -> 0   (val 0.0)
        mag < 0.75 -> 1   (val 0.5)
        mag < 1.25 -> 2   (val 1.0)
        mag < 1.75 -> 3   (val 1.5)
        mag < 2.50 -> 4   (val 2.0)
        mag < 3.50 -> 5   (val 3.0)
        mag < 5.00 -> 6   (val 4.0)
        else        -> 7   (val 6.0)
    Ties round up (e.g. 0.25 -> 0.5). Matters for byte/scale agreement
    with the on-disk kernel quant — argmin breaks ties toward the
    smaller LUT value, which disagrees on every halfway case.
    """
    thresholds = np.array([0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00],
                          dtype=np.float32)
    return np.searchsorted(thresholds, scaled_abs, side="right").astype(np.uint8)


def quant_block16_kernel(block16: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay kernel's per-16 block quantization.
    Returns (nibbles[16] u8 with sign bit, e4m3_scale_byte u8, scale_f32).
    """
    peak = float(np.max(np.abs(block16)))
    s_raw = peak / 6.0 if peak > 0.0 else 0.0
    # E4M3 round-trip via grid (mirrors __nv_fp8_e4m3(s))
    if s_raw == 0.0:
        scale_f32 = 0.0
    else:
        scale_f32 = float(round_to_e4m3(np.array([s_raw]))[0])
    if scale_f32 == 0.0:
        return np.zeros(16, dtype=np.uint8), np.array([0], dtype=np.uint8), 0.0
    inv = 1.0 / scale_f32
    scaled = block16 * inv
    sign_bits = (scaled < 0).astype(np.uint8)
    mags = np.abs(scaled)
    mag_idx = _nearest_e2m1_idx(mags)
    nibs = ((sign_bits << 3) | mag_idx).astype(np.uint8)
    # Encode E4M3 byte (find the byte whose decoded value == scale_f32)
    # Build once + cache.
    e4m3_byte = _e4m3_byte_from_value(scale_f32)
    return nibs, np.array([e4m3_byte], dtype=np.uint8), scale_f32


_E4M3_VAL_TO_BYTE: dict[float, int] | None = None


def _e4m3_byte_from_value(v: float) -> int:
    global _E4M3_VAL_TO_BYTE
    if _E4M3_VAL_TO_BYTE is None:
        _E4M3_VAL_TO_BYTE = {}
        bytes_ = np.arange(256, dtype=np.uint8)
        vals = e4m3_to_f32(bytes_)
        for b, val in zip(bytes_.tolist(), vals.tolist()):
            if val > 0 and val not in _E4M3_VAL_TO_BYTE:
                _E4M3_VAL_TO_BYTE[val] = b
    # nearest match
    keys = np.array(sorted(_E4M3_VAL_TO_BYTE.keys()))
    i = int(np.argmin(np.abs(keys - v)))
    return _E4M3_VAL_TO_BYTE[float(keys[i])]


def quant_array_kernel(x_flat: np.ndarray, policy: str = "mse") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a 1-D fp32 array (length multiple of 16) using the kernel
    policy. Returns (packed_u8[N/2], scale_e4m3_u8[N/16], dequant_f32[N]).
    Vectorised across 16-blocks for speed.

    `policy`:
      - "amax6": scale = peak/6, E4M3-rounded (OCP baseline).
      - "mse" (default): mirrors MSE policy in fused_rope_partial_nvfp4kv.cu.
        Searches 4 candidates {peak/6, peak/4, second/6, second/4} (each
        E4M3-rounded), picks the one minimising sum-squared block error.
        Prefers c0 on ties (range-preserving baseline). The on-disk shadow
        dump was produced with the MSE policy.
    """
    n = x_flat.size
    assert n % 16 == 0
    blocks = x_flat.reshape(-1, 16).astype(np.float32)
    abs_blocks = np.abs(blocks)
    peaks = np.max(abs_blocks, axis=1)

    if policy == "mse":
        # second-largest abs per row
        argmax_pos = np.argmax(abs_blocks, axis=1)
        ab2 = abs_blocks.copy()
        ab2[np.arange(blocks.shape[0]), argmax_pos] = 0.0
        seconds = np.max(ab2, axis=1)
        c0 = round_to_e4m3(np.where(peaks > 0, peaks / 6.0, 0.0))
        c1 = round_to_e4m3(np.where(peaks > 0, peaks / 4.0, 0.0))
        c2 = round_to_e4m3(np.where(seconds > 0, seconds / 6.0, 0.0))
        c3 = round_to_e4m3(np.where(seconds > 0, seconds / 4.0, 0.0))
        cand = np.stack([c0, c1, c2, c3], axis=1)        # [B, 4]
        scale_f32 = np.zeros(blocks.shape[0], dtype=np.float32)

        # Chunked over blocks to keep [chunk, 4, 16] tensor footprint small.
        chunk = 65536
        for s in range(0, blocks.shape[0], chunk):
            e = min(s + chunk, blocks.shape[0])
            blk_c = blocks[s:e]                          # [c, 16]
            cand_c = cand[s:e]                           # [c, 4]
            safe = np.where(cand_c > 0, cand_c, 1.0)
            scaled = blk_c[:, None, :] / safe[:, :, None]   # [c, 4, 16]
            signs = (scaled < 0).astype(np.float32)
            mags = np.abs(scaled)
            idx = _nearest_e2m1_idx(mags)
            chosen = E2M1_MAGS[idx]
            deq_cand = chosen * (1 - 2 * signs) * cand_c[:, :, None]
            deq_cand = np.where(cand_c[:, :, None] > 0, deq_cand, 0.0)
            sse = np.sum((deq_cand - blk_c[:, None, :]) ** 2, axis=-1)  # [c, 4]
            best = np.argmin(sse, axis=1)                # [c]
            rows = np.arange(e - s)
            scale_f32[s:e] = cand_c[rows, best]
    else:
        s_raw = np.where(peaks > 0, peaks / 6.0, 0.0)
        scale_f32 = round_to_e4m3(s_raw)
    inv = np.where(scale_f32 > 0, 1.0 / np.maximum(scale_f32, 1e-30), 0.0)
    scaled = blocks * inv[:, None]
    signs = (scaled < 0).astype(np.uint8)
    mags = np.abs(scaled)
    # nearest LUT idx per element (kernel-faithful threshold ladder)
    mag_idx = _nearest_e2m1_idx(mags)
    nibs = ((signs << 3) | mag_idx).astype(np.uint8)  # [B, 16]

    # Pack 2 nibbles per byte: low = even, high = odd.
    lo = nibs[:, 0::2]  # [B, 8]
    hi = nibs[:, 1::2]  # [B, 8]
    packed_blocks = (lo | (hi << 4)).astype(np.uint8)  # [B, 8]
    packed = packed_blocks.reshape(-1)

    # E4M3 scale bytes — vectorised: nearest E4M3 magnitude byte per block
    e4m3_vals = e4m3_to_f32(np.arange(256, dtype=np.uint8))
    pos_mask = e4m3_vals > 0
    pos_idx = np.nonzero(pos_mask)[0]      # byte indices that decode positive
    pos_vals_only = e4m3_vals[pos_mask]    # corresponding positive magnitudes
    # For each scale_f32 value, find nearest pos_vals_only entry.
    diffs = np.abs(scale_f32[:, None] - pos_vals_only[None, :])
    nearest = np.argmin(diffs, axis=1)
    scale_bytes = pos_idx[nearest].astype(np.uint8)
    scale_bytes[scale_f32 <= 0] = 0

    # Dequant
    deq = (E2M1_LUT[nibs] * scale_f32[:, None]).reshape(-1)
    return packed, scale_bytes, deq.astype(np.float32)


# ----------------------------------------------------------------------
# Oracle scale: best per-block scale that minimizes block L2.
# ----------------------------------------------------------------------

def quant_with_scale(block: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(block)
    inv = 1.0 / scale
    scaled = block * inv
    signs = (scaled < 0).astype(np.float32)
    mags = np.abs(scaled)
    idx = _nearest_e2m1_idx(mags)
    chosen_mag = E2M1_MAGS[idx]
    return (chosen_mag * (1 - 2 * signs) * scale).astype(np.float32)


def oracle_quant_array(x_flat: np.ndarray, grid_n: int = ORACLE_SCALE_GRID) -> np.ndarray:
    """Per-16 block, exhaustive search of scale in [amax/12, amax/3]
    minimizing block L2 reconstruction error. Returns dequant fp32.
    Fully vectorised across all blocks × candidates × 16 channels — uses
    a chunked loop over blocks to keep peak memory bounded.
    """
    blocks = x_flat.reshape(-1, 16).astype(np.float32)
    peaks = np.max(np.abs(blocks), axis=1)
    out = np.zeros_like(blocks)
    grid = np.linspace(1.0 / 12.0, 1.0 / 3.0, grid_n).astype(np.float32)

    nonzero = peaks > 0
    if not np.any(nonzero):
        return out.reshape(-1)

    nz_idx = np.nonzero(nonzero)[0]
    blocks_nz = blocks[nz_idx]              # [B', 16]
    peaks_nz = peaks[nz_idx]                # [B']

    # Chunk to bound memory: per-chunk tensor is [chunk, grid, 16] f32
    chunk = max(1, int(2_000_000 / (grid_n * 16)))  # ~120 MB upper bound
    for start in range(0, blocks_nz.shape[0], chunk):
        end = min(start + chunk, blocks_nz.shape[0])
        blk_c = blocks_nz[start:end]            # [c, 16]
        pk_c = peaks_nz[start:end]              # [c]
        # candidate scales [c, grid_n] then snap to E4M3
        cand = pk_c[:, None] * grid[None, :]    # [c, grid_n]
        cand_e4m3 = round_to_e4m3(cand)         # [c, grid_n]
        # Quant: scaled = blk[..., None, :] / cand[..., :, None]
        # but we need to handle scale==0
        safe = np.where(cand_e4m3 > 0, cand_e4m3, 1.0)
        # scaled[c, g, 16]
        scaled = blk_c[:, None, :] / safe[:, :, None]
        signs = (scaled < 0).astype(np.float32)
        mags = np.abs(scaled)
        # nearest LUT mag (kernel-faithful threshold ladder)
        idx = _nearest_e2m1_idx(mags)
        chosen_mag = E2M1_MAGS[idx]
        deq = chosen_mag * (1 - 2 * signs) * cand_e4m3[:, :, None]
        # zero out invalid scale candidates
        deq = np.where(cand_e4m3[:, :, None] > 0, deq, 0.0)
        # block L2 err per candidate: sum over last axis
        err = np.sum((deq - blk_c[:, None, :]) ** 2, axis=-1)  # [c, grid_n]
        best_g = np.argmin(err, axis=1)                        # [c]
        rows = np.arange(end - start)
        best_deq = deq[rows, best_g]                           # [c, 16]
        out[nz_idx[start:end]] = best_deq

    return out.reshape(-1)


# ----------------------------------------------------------------------
# Rotation matrices
# ----------------------------------------------------------------------

def random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def walsh_hadamard(d: int) -> np.ndarray:
    H = scipy_hadamard(d).astype(np.float32)
    return H / math.sqrt(d)


def signed_hadamard(d: int, seed: int) -> np.ndarray:
    H = walsh_hadamard(d)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    return H * signs[None, :]  # H @ diag(signs)


# ----------------------------------------------------------------------
# Per-layer evaluation
# ----------------------------------------------------------------------

def fro_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return rel_err(a, b)


def kl_softmax(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(softmax(p) || softmax(q)) along last axis, returned mean."""
    p = softmax(p_logits)
    q = softmax(q_logits)
    eps = 1e-30
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return float(np.mean(kl))


def top1_mass_retain(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = softmax(p_logits)
    q = softmax(q_logits)
    # argmax of GT for each row, mass that test assigns
    gt_idx = np.argmax(p_logits, axis=-1)
    rows = np.arange(p_logits.shape[0])
    return float(np.mean(q[rows, gt_idx]))


def position_mask(ctx: int) -> tuple[slice, slice, slice]:
    a = ctx // 3
    b = 2 * ctx // 3
    return slice(0, a), slice(a, b), slice(b, ctx)


def evaluate_layer(
    layer_info: dict,
    meta: dict,
    k_truth_full: np.ndarray,  # [num_blocks, block_size, nkvh, hd] f32
    v_truth_full: np.ndarray,
    k_kernel_deq_full: np.ndarray,  # baseline NVFP4 dequant from disk
    v_kernel_deq_full: np.ndarray,
    q_layer: np.ndarray | None,    # [num_heads, hd] or None
) -> dict:
    """Compute baseline + oracle + rotation conditions for one layer."""
    hd = layer_info["head_dim"]
    nkvh = layer_info["num_kv_heads"]
    ctx = meta["context_len"]
    num_heads = meta["num_heads"]
    group = num_heads // nkvh
    inv_sqrt_hd = 1.0 / math.sqrt(hd)

    # Flatten to per-kv_head 2-D arrays once: [ctx, hd]
    def _per_head(arr_full: np.ndarray, h: int) -> np.ndarray:
        return arr_full[:, :, h, :].reshape(-1, hd)[:ctx].astype(np.float32)

    k_per_head = [_per_head(k_truth_full, h) for h in range(nkvh)]
    v_per_head = [_per_head(v_truth_full, h) for h in range(nkvh)]
    k_kernel_per_head = [_per_head(k_kernel_deq_full, h) for h in range(nkvh)]
    v_kernel_per_head = [_per_head(v_kernel_deq_full, h) for h in range(nkvh)]

    # Build conditions: each condition is a function (k_truth_per_head, v_truth_per_head)
    # that returns (k_test_per_head, v_test_per_head) of same shape.
    # When the condition is "rotated", we quantize K·H, dequant, and compare
    # against K·H (Frobenius is rotation-invariant). Q is right-multiplied
    # by H so logits Q' (KH)^T = Q H H^T K^T = Q K^T (Q exact would match).
    # But Q is NOT rotated in storage — only KV. We still rotate Q for the
    # comparison so logit_truth uses (Q, K_truth) and logit_test uses
    # (Q·H, dequant(quant(K_truth·H))). The "truth" logits remain unrotated.

    conditions: dict[str, dict] = {}

    # Baseline: kernel dequant directly from disk
    conditions["baseline"] = {
        "k_test": k_kernel_per_head,
        "v_test": v_kernel_per_head,
        "rot": None,
    }

    # Sanity-check Python clone vs kernel: quant K_truth ourselves.
    py_k_per_head = []
    py_v_per_head = []
    for h in range(nkvh):
        _, _, kdeq = quant_array_kernel(k_per_head[h].reshape(-1))
        _, _, vdeq = quant_array_kernel(v_per_head[h].reshape(-1))
        py_k_per_head.append(kdeq.reshape(-1, hd))
        py_v_per_head.append(vdeq.reshape(-1, hd))
    conditions["python_clone"] = {
        "k_test": py_k_per_head,
        "v_test": py_v_per_head,
        "rot": None,
    }

    # Oracle scale
    oracle_k = []
    oracle_v = []
    for h in range(nkvh):
        oracle_k.append(oracle_quant_array(k_per_head[h].reshape(-1)).reshape(-1, hd))
        oracle_v.append(oracle_quant_array(v_per_head[h].reshape(-1)).reshape(-1, hd))
    conditions["oracle_scale"] = {
        "k_test": oracle_k,
        "v_test": oracle_v,
        "rot": None,
    }

    # Rotations: random_orth (multiple seeds), walsh, signed_walsh (multiple seeds)
    rot_specs: list[tuple[str, np.ndarray]] = []
    for s in range(NUM_RANDOM_ORTH_SEEDS):
        rot_specs.append((f"rot_random_orth_seed{s}", random_orthogonal(hd, RNG_BASE_SEED + s)))
    rot_specs.append(("rot_walsh", walsh_hadamard(hd)))
    for s in range(NUM_SIGNED_HADAMARD_SEEDS):
        rot_specs.append((f"rot_signed_walsh_seed{s}", signed_hadamard(hd, RNG_BASE_SEED + 1000 + s)))

    for name, R in rot_specs:
        rk = []
        rv = []
        for h in range(nkvh):
            kr = k_per_head[h] @ R  # [ctx, hd]
            vr = v_per_head[h] @ R
            _, _, kdeq = quant_array_kernel(kr.reshape(-1))
            _, _, vdeq = quant_array_kernel(vr.reshape(-1))
            rk.append(kdeq.reshape(-1, hd))
            rv.append(vdeq.reshape(-1, hd))
        conditions[name] = {"k_test": rk, "v_test": rv, "rot": R}

    # Rotation + oracle scale (use best signed_walsh seed deterministically)
    # We'll pick best after the fact; here just do walsh+oracle as a probe.
    walsh = walsh_hadamard(hd)
    rk_oracle = []
    rv_oracle = []
    for h in range(nkvh):
        kr = k_per_head[h] @ walsh
        vr = v_per_head[h] @ walsh
        rk_oracle.append(oracle_quant_array(kr.reshape(-1)).reshape(-1, hd))
        rv_oracle.append(oracle_quant_array(vr.reshape(-1)).reshape(-1, hd))
    conditions["rot_walsh_plus_oracle"] = {"k_test": rk_oracle, "v_test": rv_oracle, "rot": walsh}

    # Build truth logits/attention once per kv_head
    has_q = q_layer is not None
    truth_logits = None
    truth_attn_out = None
    if has_q:
        truth_logits = []  # per kv_head: [group, ctx]
        truth_attn_out = []  # per kv_head: [group, hd]
        for h in range(nkvh):
            q_group = q_layer[h * group:(h + 1) * group]  # [group, hd]
            lg = (q_group @ k_per_head[h].T) * inv_sqrt_hd
            truth_logits.append(lg)
            truth_attn_out.append(softmax(lg) @ v_per_head[h])

    pos_slices = position_mask(ctx)
    pos_names = ("early", "mid", "late")

    # Score each condition
    results = {}
    for cname, cdat in conditions.items():
        k_test_list = cdat["k_test"]
        v_test_list = cdat["v_test"]
        R = cdat["rot"]

        per_head_metrics = []
        for h in range(nkvh):
            k_truth = k_per_head[h]
            v_truth = v_per_head[h]
            k_test = k_test_list[h]
            v_test = v_test_list[h]
            # K rel_err: when rotated, k_test is dequant(quant(K@R)), and we
            # compare to K@R (Frobenius invariant under R).
            k_target = k_truth @ R if R is not None else k_truth
            v_target = v_truth @ R if R is not None else v_truth
            entry = {
                "rel_err_K": fro_rel_err(k_target, k_test),
                "rel_err_V": fro_rel_err(v_target, v_test),
            }
            if has_q:
                q_group = q_layer[h * group:(h + 1) * group]
                if R is not None:
                    q_eff = q_group @ R
                else:
                    q_eff = q_group
                lg_test = (q_eff @ k_test.T) * inv_sqrt_hd
                lg_truth = truth_logits[h]
                # logit err averaged over group
                le = [fro_rel_err(lg_truth[g], lg_test[g]) for g in range(group)]
                t8 = [topk_overlap(lg_truth[g], lg_test[g], 8) for g in range(group)]
                t16 = [topk_overlap(lg_truth[g], lg_test[g], 16) for g in range(group)]
                kl = kl_softmax(lg_truth, lg_test)
                top1 = top1_mass_retain(lg_truth, lg_test)
                # out_err
                attn_test = softmax(lg_test) @ v_test
                attn_truth = truth_attn_out[h]
                oe = [fro_rel_err(attn_truth[g], attn_test[g]) for g in range(group)]
                entry["logit_err"] = float(np.mean(le))
                entry["topk8"] = float(np.mean(t8))
                entry["topk16"] = float(np.mean(t16))
                entry["kl"] = kl
                entry["top1_mass"] = top1
                entry["out_err"] = float(np.mean(oe))

                # Position-conditioned (only on logit_err / topk8 / out_err)
                pos = {}
                for sl, pn in zip(pos_slices, pos_names):
                    if sl.stop - sl.start < 8:
                        continue
                    le_p = fro_rel_err(lg_truth[:, sl], lg_test[:, sl])
                    t8_p = float(np.mean([topk_overlap(lg_truth[g, sl], lg_test[g, sl], 8)
                                          for g in range(group)]))
                    pos[pn] = {"logit_err": le_p, "topk8": t8_p}
                entry["pos"] = pos
            per_head_metrics.append(entry)

        # Aggregate
        agg = {
            "mean_rel_err_K": float(np.mean([e["rel_err_K"] for e in per_head_metrics])),
            "max_rel_err_K": float(np.max([e["rel_err_K"] for e in per_head_metrics])),
            "mean_rel_err_V": float(np.mean([e["rel_err_V"] for e in per_head_metrics])),
            "max_rel_err_V": float(np.max([e["rel_err_V"] for e in per_head_metrics])),
        }
        if has_q:
            agg["mean_logit_err"] = float(np.mean([e["logit_err"] for e in per_head_metrics]))
            agg["max_logit_err"] = float(np.max([e["logit_err"] for e in per_head_metrics]))
            agg["mean_topk8"] = float(np.mean([e["topk8"] for e in per_head_metrics]))
            agg["min_topk8"] = float(np.min([e["topk8"] for e in per_head_metrics]))
            agg["mean_topk16"] = float(np.mean([e["topk16"] for e in per_head_metrics]))
            agg["mean_kl"] = float(np.mean([e["kl"] for e in per_head_metrics]))
            agg["mean_top1_mass"] = float(np.mean([e["top1_mass"] for e in per_head_metrics]))
            agg["min_top1_mass"] = float(np.min([e["top1_mass"] for e in per_head_metrics]))
            agg["mean_out_err"] = float(np.mean([e["out_err"] for e in per_head_metrics]))
            # position aggregate
            pos_agg = {}
            for pn in pos_names:
                vals_le = [e["pos"].get(pn, {}).get("logit_err") for e in per_head_metrics
                           if "pos" in e and pn in e["pos"]]
                vals_t8 = [e["pos"].get(pn, {}).get("topk8") for e in per_head_metrics
                           if "pos" in e and pn in e["pos"]]
                if vals_le:
                    pos_agg[pn] = {
                        "mean_logit_err": float(np.mean(vals_le)),
                        "mean_topk8": float(np.mean(vals_t8)),
                    }
            agg["pos"] = pos_agg
        results[cname] = agg

    return results


# ----------------------------------------------------------------------
# Sanity check: Python clone vs on-disk kernel quant
# ----------------------------------------------------------------------

def sanity_check(
    layer_info: dict, meta: dict,
    k_truth_full: np.ndarray, k_kernel_packed: np.ndarray,
    k_kernel_scale_e4m3: np.ndarray, k_kernel_deq_full: np.ndarray,
) -> dict:
    """Quantize the f16 K shadow with our Python clone, compare bytes
    + scale picks against the kernel's on-disk quant.
    """
    hd = layer_info["head_dim"]
    nkvh = layer_info["num_kv_heads"]
    nblocks = layer_info["num_blocks"]
    block_size = meta["block_size"]
    total = nblocks * block_size * nkvh * hd

    # Use only the populated portion (first ctx slots in flat layout),
    # but since unused slots are zeros, full-array compare is fine and
    # still informative.
    flat_truth = k_truth_full.reshape(-1)[:total]
    py_packed, py_scale_bytes, py_deq = quant_array_kernel(flat_truth)

    # Byte-match
    n_match_pack = int(np.sum(py_packed == k_kernel_packed[:py_packed.size]))
    pack_match = n_match_pack / py_packed.size

    # Scale-pick agreement (compare E4M3 byte values)
    n_match_scale = int(np.sum(py_scale_bytes == k_kernel_scale_e4m3[:py_scale_bytes.size]))
    scale_match = n_match_scale / py_scale_bytes.size

    # Rel_err comparison: kernel-deq vs truth, py-deq vs truth
    rel_err_kernel = float(np.linalg.norm(k_kernel_deq_full[:total] - flat_truth) /
                           (np.linalg.norm(flat_truth) + 1e-30))
    rel_err_python = float(np.linalg.norm(py_deq - flat_truth) /
                           (np.linalg.norm(flat_truth) + 1e-30))

    return {
        "byte_match_rate": pack_match,
        "scale_pick_agreement": scale_match,
        "rel_err_kernel_pct": rel_err_kernel * 100.0,
        "rel_err_python_pct": rel_err_python * 100.0,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    if not DUMP_DIR.exists():
        print(f"FATAL: dump dir {DUMP_DIR} not found", file=sys.stderr)
        sys.exit(1)
    meta = json.loads((DUMP_DIR / "meta.json").read_text())
    block_size = meta["block_size"]

    print(f"[rotate-proto] dump = {DUMP_DIR}")
    print(f"[rotate-proto] ctx = {meta['context_len']}, layers = {meta['shadow_layer_indices']}")
    print(f"[rotate-proto] seeds: random_orth={NUM_RANDOM_ORTH_SEEDS}, signed_walsh={NUM_SIGNED_HADAMARD_SEEDS}")
    print()

    sanity_results: list[dict] = []
    layer_results: dict[int, dict] = {}

    for layer_info in meta["layers"]:
        l = layer_info["layer"]
        hd = layer_info["head_dim"]
        nkvh = layer_info["num_kv_heads"]
        nblocks = layer_info["num_blocks"]
        print(f"[rotate-proto] layer {l}  hd={hd}  nkvh={nkvh}  nblocks={nblocks} ...", flush=True)

        # Load
        k_packed = np.fromfile(DUMP_DIR / f"layer_{l}_k.bin", dtype=np.uint8)
        v_packed = np.fromfile(DUMP_DIR / f"layer_{l}_v.bin", dtype=np.uint8)
        k_scale = np.fromfile(DUMP_DIR / f"layer_{l}_k_scale.bin", dtype=np.uint8)
        v_scale = np.fromfile(DUMP_DIR / f"layer_{l}_v_scale.bin", dtype=np.uint8)
        k_shadow = np.fromfile(DUMP_DIR / f"layer_{l}_k_shadow.bin", dtype=np.float16).astype(np.float32)
        v_shadow = np.fromfile(DUMP_DIR / f"layer_{l}_v_shadow.bin", dtype=np.float16).astype(np.float32)

        # Dequant kernel cache to fp32 [num_blocks, block_size, nkvh, hd]
        total = nblocks * block_size * nkvh * hd
        # use unpack_e2m1_u8 + scales
        unpacked = unpack_e2m1_u8(k_packed)[:total]
        scales_f32 = e4m3_to_f32(k_scale)
        scales_per_elem = np.repeat(scales_f32, 16)[:total]
        k_kernel_deq = unpacked * scales_per_elem

        unpacked_v = unpack_e2m1_u8(v_packed)[:total]
        scales_v_f32 = e4m3_to_f32(v_scale)
        scales_v_per_elem = np.repeat(scales_v_f32, 16)[:total]
        v_kernel_deq = unpacked_v * scales_v_per_elem

        shape = (nblocks, block_size, nkvh, hd)
        k_truth_full = k_shadow.reshape(shape)
        v_truth_full = v_shadow.reshape(shape)
        k_kernel_full = k_kernel_deq.reshape(shape)
        v_kernel_full = v_kernel_deq.reshape(shape)

        # Sanity-check Python clone vs kernel
        sc = sanity_check(layer_info, meta, k_truth_full, k_packed, k_scale, k_kernel_deq)
        sc["layer"] = l
        sanity_results.append(sc)
        print(f"  sanity: byte_match={sc['byte_match_rate']*100:.2f}%  "
              f"scale_pick={sc['scale_pick_agreement']*100:.2f}%  "
              f"rel_err kernel={sc['rel_err_kernel_pct']:.2f}%  python={sc['rel_err_python_pct']:.2f}%",
              flush=True)

        # Q
        q = load_layer_q(DUMP_DIR, meta, layer_info)

        # Evaluate all conditions
        res = evaluate_layer(layer_info, meta, k_truth_full, v_truth_full,
                             k_kernel_full, v_kernel_full, q)
        layer_results[l] = res

        # quick log
        b = res["baseline"]
        print(f"  baseline: rel_err_K={b['mean_rel_err_K']*100:.2f}%  "
              f"top1_mass={b.get('mean_top1_mass', float('nan')):.4f}", flush=True)
        # find best signed_walsh
        best_seed_name = None
        best_K = float("inf")
        for cname, agg in res.items():
            if cname.startswith("rot_signed_walsh_seed"):
                if agg["mean_rel_err_K"] < best_K:
                    best_K = agg["mean_rel_err_K"]
                    best_seed_name = cname
        if best_seed_name:
            br = res[best_seed_name]
            print(f"  best signed_walsh: {best_seed_name}  rel_err_K={br['mean_rel_err_K']*100:.2f}%  "
                  f"top1_mass={br.get('mean_top1_mass', float('nan')):.4f}", flush=True)

    # ----------------------------------------------------------
    # Aggregations across layers / seed-regime selection
    # ----------------------------------------------------------

    # Per rotation flavor, list of seeds
    rotation_flavors = {
        "random_orth": [f"rot_random_orth_seed{s}" for s in range(NUM_RANDOM_ORTH_SEEDS)],
        "walsh": ["rot_walsh"],
        "signed_walsh": [f"rot_signed_walsh_seed{s}" for s in range(NUM_SIGNED_HADAMARD_SEEDS)],
    }

    # best-per-layer / fixed-global / fixed-per-head_dim
    layers = list(layer_results.keys())
    head_dims = {l: meta["layers"][i]["head_dim"]
                 for i, l in enumerate([li["layer"] for li in meta["layers"]])}

    def best_per_layer(metric_key: str, flavor: str) -> dict[int, tuple[str, float]]:
        out = {}
        for l in layers:
            best_name, best_v = None, float("inf")
            for seed_name in rotation_flavors[flavor]:
                v = layer_results[l][seed_name][metric_key]
                if v < best_v:
                    best_v = v
                    best_name = seed_name
            out[l] = (best_name, best_v)
        return out

    def fixed_global(metric_key: str, flavor: str) -> tuple[str, dict[int, float]]:
        # pick seed that minimizes mean(metric) over all layers
        best_name, best_mean = None, float("inf")
        for seed_name in rotation_flavors[flavor]:
            m = float(np.mean([layer_results[l][seed_name][metric_key] for l in layers]))
            if m < best_mean:
                best_mean = m
                best_name = seed_name
        per_layer = {l: layer_results[l][best_name][metric_key] for l in layers}
        return best_name, per_layer

    def fixed_per_hd(metric_key: str, flavor: str) -> dict[int, tuple[str, float]]:
        # one seed per (head_dim) value, picking by mean over layers of that hd
        hd_seeds: dict[int, str] = {}
        for hd_val in set(head_dims.values()):
            ls = [l for l, h in head_dims.items() if h == hd_val]
            best_name, best_mean = None, float("inf")
            for seed_name in rotation_flavors[flavor]:
                m = float(np.mean([layer_results[l][seed_name][metric_key] for l in ls]))
                if m < best_mean:
                    best_mean = m
                    best_name = seed_name
            hd_seeds[hd_val] = best_name
        out = {}
        for l in layers:
            sn = hd_seeds[head_dims[l]]
            out[l] = (sn, layer_results[l][sn][metric_key])
        return out

    # Compute regimes for signed_walsh on metric mean_rel_err_K
    metric = "mean_rel_err_K"
    bpl_signed = best_per_layer(metric, "signed_walsh")
    fg_signed_name, fg_signed_perlayer = fixed_global(metric, "signed_walsh")
    fphd_signed = fixed_per_hd(metric, "signed_walsh")

    bpl_rand = best_per_layer(metric, "random_orth")
    fg_rand_name, fg_rand_perlayer = fixed_global(metric, "random_orth")
    fphd_rand = fixed_per_hd(metric, "random_orth")

    # ----------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------

    baseline_K = {l: layer_results[l]["baseline"]["mean_rel_err_K"] for l in layers}
    oracle_K = {l: layer_results[l]["oracle_scale"]["mean_rel_err_K"] for l in layers}
    # Use fixed-per-hd signed_walsh as the realistic regime
    rot_K = {l: fphd_signed[l][1] for l in layers}

    # Relative drops
    rel_drop_rotation = float(np.mean([(baseline_K[l] - rot_K[l]) / baseline_K[l] for l in layers]))
    rel_drop_oracle = float(np.mean([(baseline_K[l] - oracle_K[l]) / baseline_K[l] for l in layers]))

    # Top-k / top-1 deltas (best regime vs baseline) — use fixed-per-hd
    base_top1 = float(np.mean([layer_results[l]["baseline"].get("mean_top1_mass", 0) for l in layers
                               if layer_results[l]["baseline"].get("mean_top1_mass") is not None]))
    rot_top1_vals = []
    for l in layers:
        seed_name = fphd_signed[l][0]
        rot_top1_vals.append(layer_results[l][seed_name].get("mean_top1_mass", 0))
    rot_top1 = float(np.mean(rot_top1_vals))

    base_top8 = float(np.mean([layer_results[l]["baseline"].get("mean_topk8", 0) for l in layers]))
    rot_top8 = float(np.mean([layer_results[l][fphd_signed[l][0]].get("mean_topk8", 0) for l in layers]))

    # Worst-case across (layers, position bucket) for the rot regime
    pos_worst_logit_err = 0.0
    pos_worst_layer = None
    pos_worst_bucket = None
    for l in layers:
        seed_name = fphd_signed[l][0]
        pos = layer_results[l][seed_name].get("pos", {})
        for bn, bd in pos.items():
            v = bd["mean_logit_err"]
            if v > pos_worst_logit_err:
                pos_worst_logit_err = v
                pos_worst_layer = l
                pos_worst_bucket = bn

    if rel_drop_rotation >= 0.20 and rot_top8 > base_top8 and rot_top1 > base_top1:
        verdict = "GO_KERNEL_IMPL"
    elif rel_drop_rotation < 0.10:
        verdict = "NO_GO"
    else:
        verdict = "NEEDS_INVESTIGATION"

    # Sanity-check gate
    sanity_pass = True
    sanity_notes = []
    for sc in sanity_results:
        py = sc["rel_err_python_pct"]
        if not (8.0 <= py <= 8.8):
            sanity_pass = False
            sanity_notes.append(f"layer {sc['layer']}: python rel_err {py:.2f}% outside [8.0, 8.8]")
        if sc["scale_pick_agreement"] < 0.85:
            sanity_pass = False
            sanity_notes.append(
                f"layer {sc['layer']}: scale-pick agreement {sc['scale_pick_agreement']*100:.1f}% < 85%")

    elapsed = time.time() - t0

    # ----------------------------------------------------------
    # Write report
    # ----------------------------------------------------------

    lines = []
    lines.append("# NVFP4 Hadamard-Rotation Prototype Report")
    lines.append("")
    lines.append(f"- dump: `{DUMP_DIR}`")
    lines.append(f"- ctx: {meta['context_len']}, prompt_len: {meta['prompt_len']}")
    lines.append(f"- layers analyzed: {meta['shadow_layer_indices']}")
    lines.append(f"- random-orth seeds: {NUM_RANDOM_ORTH_SEEDS}, "
                 f"signed-walsh seeds: {NUM_SIGNED_HADAMARD_SEEDS}")
    lines.append(f"- elapsed: {elapsed:.1f} s")
    lines.append("")
    lines.append("## 1. Sanity check (Python NVFP4 clone vs on-disk kernel quant)")
    lines.append("")
    lines.append("| layer | byte_match | scale_pick | rel_err kernel | rel_err python |")
    lines.append("|---:|---:|---:|---:|---:|")
    for sc in sanity_results:
        lines.append(f"| {sc['layer']} | {sc['byte_match_rate']*100:.2f}% | "
                     f"{sc['scale_pick_agreement']*100:.2f}% | "
                     f"{sc['rel_err_kernel_pct']:.2f}% | "
                     f"{sc['rel_err_python_pct']:.2f}% |")
    lines.append("")
    lines.append(f"- Validation gate: Python rel_err must land in [8.0, 8.8]%")
    lines.append(f"- **Sanity gate: {'PASS' if sanity_pass else 'FAIL'}**")
    if sanity_notes:
        for n in sanity_notes:
            lines.append(f"  - {n}")
    lines.append("")

    lines.append("## 2. Per-layer baseline + oracle_scale")
    lines.append("")
    lines.append("| layer | hd | base mean_K | oracle mean_K | Δ (rel drop) | base top1 | base topk8 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for l in layers:
        b = layer_results[l]["baseline"]
        o = layer_results[l]["oracle_scale"]
        drop = (b["mean_rel_err_K"] - o["mean_rel_err_K"]) / b["mean_rel_err_K"]
        top1 = b.get("mean_top1_mass", float("nan"))
        top8 = b.get("mean_topk8", float("nan"))
        lines.append(f"| {l} | {head_dims[l]} | {b['mean_rel_err_K']*100:.2f}% | "
                     f"{o['mean_rel_err_K']*100:.2f}% | {drop*100:.1f}% | "
                     f"{top1:.4f} | {top8:.3f} |")
    lines.append("")

    lines.append("## 3. Rotation gain by flavor (mean over layers, mean across heads)")
    lines.append("")
    for flavor in ["random_orth", "walsh", "signed_walsh"]:
        lines.append(f"### {flavor}")
        lines.append("")
        lines.append("| seed | mean K rel_err | mean logit_err | mean topk8 | mean top1 mass | mean kl | mean out_err |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for seed_name in rotation_flavors[flavor]:
            mk = float(np.mean([layer_results[l][seed_name]["mean_rel_err_K"] for l in layers]))
            mle = float(np.mean([layer_results[l][seed_name].get("mean_logit_err", 0) for l in layers]))
            mt8 = float(np.mean([layer_results[l][seed_name].get("mean_topk8", 0) for l in layers]))
            mt1 = float(np.mean([layer_results[l][seed_name].get("mean_top1_mass", 0) for l in layers]))
            mkl = float(np.mean([layer_results[l][seed_name].get("mean_kl", 0) for l in layers]))
            moe = float(np.mean([layer_results[l][seed_name].get("mean_out_err", 0) for l in layers]))
            lines.append(f"| {seed_name} | {mk*100:.2f}% | {mle*100:.2f}% | {mt8:.3f} | "
                         f"{mt1:.4f} | {mkl:.4f} | {moe*100:.2f}% |")
        lines.append("")

    lines.append("## 4. Three seed-selection regimes (signed_walsh, mean_rel_err_K)")
    lines.append("")
    lines.append("| layer | hd | baseline | best-per-layer | fixed-global | fixed-per-hd |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for l in layers:
        bpl_name, bpl_v = bpl_signed[l]
        fphd_name, fphd_v = fphd_signed[l]
        lines.append(f"| {l} | {head_dims[l]} | {baseline_K[l]*100:.2f}% | "
                     f"{bpl_v*100:.2f}% ({bpl_name.split('seed')[-1]}) | "
                     f"{fg_signed_perlayer[l]*100:.2f}% | "
                     f"{fphd_v*100:.2f}% ({fphd_name.split('seed')[-1]}) |")
    lines.append("")
    lines.append(f"- fixed-global signed_walsh seed: {fg_signed_name}")
    lines.append(f"- fixed-per-hd seed map: " + ", ".join(
        f"hd{hd}={sn}" for hd, sn in
        {head_dims[l]: fphd_signed[l][0] for l in layers}.items()))
    lines.append("")

    lines.append("## 5. Position-conditioned breakdown (best-per-layer signed_walsh)")
    lines.append("")
    lines.append("| layer | bucket | mean logit_err | mean topk8 |")
    lines.append("|---:|---|---:|---:|")
    for l in layers:
        seed_name = bpl_signed[l][0]
        pos = layer_results[l][seed_name].get("pos", {})
        for bn in ("early", "mid", "late"):
            if bn in pos:
                bd = pos[bn]
                lines.append(f"| {l} | {bn} | {bd['mean_logit_err']*100:.2f}% | "
                             f"{bd['mean_topk8']:.3f} |")
    lines.append("")

    lines.append("## 6. Rotation + oracle scale (walsh × oracle)")
    lines.append("")
    lines.append("| layer | base K | oracle K | walsh+oracle K |")
    lines.append("|---:|---:|---:|---:|")
    for l in layers:
        b = layer_results[l]["baseline"]["mean_rel_err_K"]
        o = layer_results[l]["oracle_scale"]["mean_rel_err_K"]
        wo = layer_results[l]["rot_walsh_plus_oracle"]["mean_rel_err_K"]
        lines.append(f"| {l} | {b*100:.2f}% | {o*100:.2f}% | {wo*100:.2f}% |")
    lines.append("")

    lines.append("## 7. Verdict")
    lines.append("")
    lines.append(f"- Mean K rel_err relative drop (rotation, fixed-per-hd): **{rel_drop_rotation*100:.1f}%**")
    lines.append(f"- Mean K rel_err relative drop (oracle scale alone): {rel_drop_oracle*100:.1f}%")
    lines.append(f"- Top-1 mass retain: baseline={base_top1:.4f}  rotation={rot_top1:.4f}")
    lines.append(f"- Top-8 overlap:    baseline={base_top8:.3f}  rotation={rot_top8:.3f}")
    if pos_worst_layer is not None:
        lines.append(f"- Worst position bucket (rotation regime): layer {pos_worst_layer} "
                     f"{pos_worst_bucket}, logit_err={pos_worst_logit_err*100:.2f}%")
    lines.append("")
    lines.append(f"### **VERDICT: {verdict}**")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("- GO_KERNEL_IMPL  if rel_drop_K ≥ 20% AND topk8 improves AND top1_mass improves")
    lines.append("- NO_GO          if rel_drop_K < 10%")
    lines.append("- NEEDS_INVESTIGATION otherwise (e.g. oracle dominates rotation gain)")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))

    # ----------------------------------------------------------
    # Stdout one-screen summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print(f"  NVFP4 ROTATION PROTOTYPE — {verdict}")
    print("=" * 70)
    print(f"  report: {REPORT_PATH}")
    print(f"  elapsed: {elapsed:.1f} s")
    print()
    print(f"  sanity gate:        {'PASS' if sanity_pass else 'FAIL'}")
    print(f"  rel_drop K (rot):   {rel_drop_rotation*100:+.1f}%   "
          f"(oracle-only: {rel_drop_oracle*100:+.1f}%)")
    print(f"  top1 mass: base {base_top1:.4f}  rot {rot_top1:.4f}  "
          f"Δ {rot_top1-base_top1:+.4f}")
    print(f"  topk8:     base {base_top8:.3f}  rot {rot_top8:.3f}  "
          f"Δ {rot_top8-base_top8:+.3f}")
    if pos_worst_layer is not None:
        print(f"  worst pos bucket (rot): layer {pos_worst_layer} "
              f"{pos_worst_bucket}  logit_err={pos_worst_logit_err*100:.2f}%")
    print()
    print(f"  fixed-per-hd seeds: " + ", ".join(
        f"hd{hd}={sn.split('seed')[-1]}"
        for hd, sn in
        {head_dims[l]: fphd_signed[l][0] for l in layers}.items()))
    print("=" * 70)


if __name__ == "__main__":
    main()
