#!/usr/bin/env python3
"""
Fast single-layer rotation quickcheck.

Given the full prototype is slow on a 15.5k-token context, this script
runs ONE layer end-to-end with a reduced seed budget (4 random_orth +
walsh + 4 signed_walsh) and prints the headline numbers — used to
back-stop the verdict in the report when the full prototype timed out.

Same conventions and helpers as nvfp4_rotation_prototype.py.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard as scipy_hadamard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_rotation_prototype import (  # type: ignore
    quant_array_kernel,
    random_orthogonal,
    walsh_hadamard,
    signed_hadamard,
    oracle_quant_array,
)
from nvfp4_shadow_analyze import (  # type: ignore
    unpack_e2m1_u8, e4m3_to_f32, load_layer_q, rel_err, softmax, topk_overlap,
)


DUMP_DIR = Path("/tmp/nvfp4_shadow")
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NUM_RAND = 4
NUM_SIGNED = 4


def run():
    t0 = time.time()
    meta = json.loads((DUMP_DIR / "meta.json").read_text())
    li = next(li for li in meta["layers"] if li["layer"] == LAYER)
    hd = li["head_dim"]; nkvh = li["num_kv_heads"]; nb = li["num_blocks"]
    bs = meta["block_size"]; ctx = meta["context_len"]
    num_heads = meta["num_heads"]; group = num_heads // nkvh
    inv_sqrt = 1.0 / math.sqrt(hd)
    print(f"layer {LAYER}  hd={hd}  nkvh={nkvh}  ctx={ctx}")

    k_shadow = np.fromfile(DUMP_DIR / f"layer_{LAYER}_k_shadow.bin",
                           dtype=np.float16).astype(np.float32)
    k_packed = np.fromfile(DUMP_DIR / f"layer_{LAYER}_k.bin", dtype=np.uint8)
    k_scale = np.fromfile(DUMP_DIR / f"layer_{LAYER}_k_scale.bin", dtype=np.uint8)
    v_shadow = np.fromfile(DUMP_DIR / f"layer_{LAYER}_v_shadow.bin",
                           dtype=np.float16).astype(np.float32)
    v_packed = np.fromfile(DUMP_DIR / f"layer_{LAYER}_v.bin", dtype=np.uint8)
    v_scale = np.fromfile(DUMP_DIR / f"layer_{LAYER}_v_scale.bin", dtype=np.uint8)

    total = nb * bs * nkvh * hd
    shape = (nb, bs, nkvh, hd)
    k_truth = k_shadow.reshape(shape)
    v_truth = v_shadow.reshape(shape)
    k_kernel = (unpack_e2m1_u8(k_packed)[:total] *
                np.repeat(e4m3_to_f32(k_scale), 16)[:total]).reshape(shape)
    v_kernel = (unpack_e2m1_u8(v_packed)[:total] *
                np.repeat(e4m3_to_f32(v_scale), 16)[:total]).reshape(shape)

    q = load_layer_q(DUMP_DIR, meta, li)
    print(f"  load done  {time.time()-t0:.1f}s")

    rot_specs = [("walsh", walsh_hadamard(hd))]
    for s in range(NUM_RAND):
        rot_specs.append((f"random_orth_{s}", random_orthogonal(hd, 0xC0FFEE + s)))
    for s in range(NUM_SIGNED):
        rot_specs.append((f"signed_walsh_{s}", signed_hadamard(hd, 0xC0FFEE + 1000 + s)))

    # Per-kv_head extraction (only K — V uses no rotation in baseline)
    k_per = [k_truth[:, :, h, :].reshape(-1, hd)[:ctx] for h in range(nkvh)]
    v_per = [v_truth[:, :, h, :].reshape(-1, hd)[:ctx] for h in range(nkvh)]
    k_kern_per = [k_kernel[:, :, h, :].reshape(-1, hd)[:ctx] for h in range(nkvh)]
    v_kern_per = [v_kernel[:, :, h, :].reshape(-1, hd)[:ctx] for h in range(nkvh)]

    # Truth attn
    truth_logits = []; truth_attn = []
    for h in range(nkvh):
        qg = q[h*group:(h+1)*group]
        lg = (qg @ k_per[h].T) * inv_sqrt
        truth_logits.append(lg)
        truth_attn.append(softmax(lg) @ v_per[h])

    def metrics_K_only(k_test_per, R):
        """K-only swap: V stays fp16 ground truth (isolates K rotation gain)."""
        rels = []; logit_errs = []; t8s = []; t1s = []
        for h in range(nkvh):
            k_target = k_per[h] @ R if R is not None else k_per[h]
            rels.append(rel_err(k_target, k_test_per[h]))
            qg = q[h*group:(h+1)*group]
            qe = qg @ R if R is not None else qg
            lg = (qe @ k_test_per[h].T) * inv_sqrt
            le = [rel_err(truth_logits[h][g], lg[g]) for g in range(group)]
            t8 = [topk_overlap(truth_logits[h][g], lg[g], 8) for g in range(group)]
            # top1 mass retain
            p_truth = softmax(truth_logits[h]); p_test = softmax(lg)
            gt_idx = np.argmax(truth_logits[h], axis=-1)
            t1 = float(np.mean([p_test[g, gt_idx[g]] for g in range(group)]))
            logit_errs.append(np.mean(le)); t8s.append(np.mean(t8)); t1s.append(t1)
        return (float(np.mean(rels)), float(np.mean(logit_errs)),
                float(np.mean(t8s)), float(np.mean(t1s)))

    # Baseline
    base = metrics_K_only(k_kern_per, None)
    print(f"  baseline           K_rel={base[0]*100:.3f}%  logit_err={base[1]*100:.3f}%  "
          f"top8={base[2]:.4f}  top1={base[3]:.6f}  ({time.time()-t0:.1f}s)")

    # Oracle scale
    oracle_per = [oracle_quant_array(k_per[h].reshape(-1)).reshape(-1, hd)
                  for h in range(nkvh)]
    o = metrics_K_only(oracle_per, None)
    print(f"  oracle_scale       K_rel={o[0]*100:.3f}%  logit_err={o[1]*100:.3f}%  "
          f"top8={o[2]:.4f}  top1={o[3]:.6f}  ({time.time()-t0:.1f}s)")

    # Rotations
    results = {}
    for name, R in rot_specs:
        rk = []
        for h in range(nkvh):
            kr = k_per[h] @ R
            _, _, deq = quant_array_kernel(kr.reshape(-1))
            rk.append(deq.reshape(-1, hd))
        m = metrics_K_only(rk, R)
        results[name] = m
        print(f"  {name:20s} K_rel={m[0]*100:.3f}%  logit_err={m[1]*100:.3f}%  "
              f"top8={m[2]:.4f}  top1={m[3]:.6f}  ({time.time()-t0:.1f}s)")

    # Best rotation
    best = min(results.items(), key=lambda kv: kv[1][0])
    print()
    print(f"BEST ROTATION: {best[0]}  K_rel={best[1][0]*100:.3f}%")
    print(f"BASELINE K_rel: {base[0]*100:.3f}%")
    drop = (base[0] - best[1][0]) / base[0] * 100
    print(f"RELATIVE DROP: {drop:.2f}%   ({'GO' if drop >= 20 else 'NO_GO' if drop < 10 else 'INVESTIGATE'})")
    print(f"ORACLE K_rel: {o[0]*100:.3f}%   relative drop: {(base[0]-o[0])/base[0]*100:.2f}%")
    print(f"\ntotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
