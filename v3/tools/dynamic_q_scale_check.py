#!/usr/bin/env python3
# === DYNAMIC NVFP4 Q SCALE ===
"""Math sanity test for per-(token, head) dynamic Q FP8 scale.

Demonstrates that:
  1. Static (per-tensor) Q FP8 scale saturates when Hadamard rotation
     redistributes Q magnitudes across head_dim (post-rotation channels
     can grow up to ~sqrt(head_dim) above the un-rotated max).
  2. Dynamic per-(token, head) scale = max(amax(|Q|)/448, 1e-12)
     keeps the FP8 round-trip within the E4M3 grid resolution.

Mirrors what the patched `fused_rope_partial_nvfp4kv.cu` does:
  - block-reduce amax across head_dim threads
  - scale = max(amax/448, 1e-12)
  - quant: q_int = sat_e4m3(v / scale); dequant: v_hat = q_int * scale.

The test is host-only (numpy) so it can run pre-flight without the
GPU. Run from the repo root:

    python3 v3/tools/dynamic_q_scale_check.py

Exits non-zero on failure (dynamic worse than static, or saturation
not detected on the static path).
"""

import sys
import numpy as np


# E4M3 mantissa+exponent grid. We approximate the cast via clip to
# [-448, 448] and round to a 7-bit log-grid; coarse, but enough to
# show the saturation cliff on the static path. The Rust kernel does
# real `__nv_fp8_e4m3` casts which are tighter.
E4M3_MAX = 448.0


def fake_e4m3_round(x: np.ndarray) -> np.ndarray:
    """Saturate-and-round-toward-zero on the e4m3 normal grid.

    Approximates `__nv_fp8_e4m3` by:
      1. Clip to [-E4M3_MAX, E4M3_MAX] (saturation).
      2. Round mantissa to 3 bits at the dominant exponent (8 levels
         per binade).
    Good enough to demonstrate the relative-error gap; the production
    cast is sharper but the qualitative result is identical.
    """
    sign = np.sign(x)
    mag = np.abs(x)
    saturated = np.clip(mag, 0.0, E4M3_MAX)
    # Mantissa rounding: 3-bit (8 steps) per power-of-two binade.
    safe = np.where(saturated > 0, saturated, 1.0)
    exp = np.floor(np.log2(safe))
    # 3-bit mantissa => 8 steps in [1, 2)
    step = np.power(2.0, exp) / 8.0
    rounded = np.round(saturated / step) * step
    rounded = np.where(saturated > 0, rounded, 0.0)
    return sign * rounded


def hadamard_matrix(n: int) -> np.ndarray:
    """Return the normalized (1/sqrt(n) Walsh-Hadamard) matrix for
    a power of two n. Produces orthonormal H so H * H^T = I."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be power of two"
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def quant_dequant_static(q: np.ndarray, scale: float) -> np.ndarray:
    inv = 1.0 / scale
    qi = fake_e4m3_round(q * inv)
    return qi * scale


def quant_dequant_dynamic(q: np.ndarray) -> np.ndarray:
    # Per-vector amax → fresh scale.
    amax = float(np.max(np.abs(q)))
    scale = max(amax / E4M3_MAX, 1e-12)
    inv = 1.0 / scale
    qi = fake_e4m3_round(q * inv)
    return qi * scale


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def run_case(label: str, q: np.ndarray, hadamard: bool, static_scale: float,
             D_override: np.ndarray = None):
    head_dim = q.shape[-1]
    if hadamard:
        H = hadamard_matrix(head_dim)
        if D_override is None:
            D = np.where(np.random.default_rng(7).random(head_dim) < 0.5, -1.0, 1.0)
        else:
            D = D_override
        # Apply rotation (D then H), as the kernel does.
        q_rot = (q * D) @ H.T
        q_active = q_rot
    else:
        q_active = q

    q_static = quant_dequant_static(q_active, static_scale)
    q_dynamic = quant_dequant_dynamic(q_active)

    rel_static = relative_error(q_active, q_static)
    rel_dynamic = relative_error(q_active, q_dynamic)
    amax = float(np.max(np.abs(q_active)))
    sat_count = int(np.sum(np.abs(q_active / static_scale) > E4M3_MAX))

    print(f"[{label}]")
    print(f"  amax(|Q|)            = {amax:.4f}")
    print(f"  static scale         = {static_scale:.4f}  "
          f"(=> max|q/scale| = {amax/static_scale:.2f}, "
          f"E4M3_MAX = {E4M3_MAX})")
    print(f"  saturated entries    = {sat_count} / {q_active.size}")
    print(f"  static  rel_err      = {rel_static:.4e}")
    print(f"  dynamic rel_err      = {rel_dynamic:.4e}")
    return rel_static, rel_dynamic, sat_count


def main():
    rng = np.random.default_rng(0)
    head_dim = 256
    # Production-like Q: small magnitudes (post-RMSNorm Q is
    # roughly unit-variance, magnitudes O(1)) — the static scale
    # of 0.1 was hand-tuned for the un-rotated distribution where
    # max(|Q|) ~ a few. Without rotation, q/scale stays well under
    # 448 (~30-50). With Hadamard, a structured Q can have one or
    # two heavy components that get *amplified* on specific output
    # lanes by sqrt(D)≈16. We simulate this by making Q low-rank-ish
    # so that the rotation projects mass onto a single output lane.
    # Worst-case constructive interference: post-Hadamard one lane
    # equals sum_j H[lane, j] * D[j] * Q[j] / 1. If we pick Q[j] =
    # sign(H[0, j] * D[j]) * mag, then output lane 0 = mag * sqrt(D).
    # With mag = 3 and D=256, lane 0 hits ~48 — ratio q/scale =
    # 480 > 448 (saturation cliff) for static_scale=0.1.
    H = hadamard_matrix(head_dim)
    D_sign = np.where(rng.random(head_dim) < 0.5, -1.0, 1.0).astype(np.float32)
    # Pick Q[j] aligned with H[0, j] * D[j] so the rotation packs all
    # mass on lane 0. Note: kernel applies q_rot = (q*D) @ H^T, so
    # lane k of q_rot = sum_j H[k, j] * D[j] * q[j].
    q_un = (3.0 * np.sign(H[0]) * D_sign).astype(np.float32)
    # Override the rng-generated D inside `run_case` — easiest way is
    # to monkey-patch (or just override the seed there). Instead we
    # do the rotation here directly so the saturation reproduces.
    static_scale = 0.1

    print("Per-(token, head) dynamic Q FP8 scale — math validation")
    print("=" * 60)

    rel_s_un, rel_d_un, sat_un = run_case(
        "un-rotated  / static=0.1", q_un, hadamard=False,
        static_scale=static_scale)
    print()
    rel_s_rot, rel_d_rot, sat_rot = run_case(
        "Hadamard    / static=0.1", q_un, hadamard=True,
        static_scale=static_scale, D_override=D_sign)
    print()
    # Workaround the operator already validated: bumping static to
    # 2.0 lets Hadamard work. Confirm that, but show that dynamic is
    # still at least as good without needing a manual tuning knob.
    rel_s_rot2, rel_d_rot2, sat_rot2 = run_case(
        "Hadamard    / static=2.0", q_un, hadamard=True,
        static_scale=2.0, D_override=D_sign)
    print()

    print("Verdict:")
    print("-" * 60)
    ok = True
    # 1. Static path must saturate when Hadamard runs and static=0.1.
    if sat_rot == 0:
        print("  [FAIL] expected static scale 0.1 to saturate post-Hadamard")
        ok = False
    else:
        print(f"  [OK]   static=0.1 + Hadamard saturates "
              f"{sat_rot}/{q_un.size} entries (the FP8 bug)")
    # 2. Dynamic must beat static-0.1 under Hadamard.
    if rel_d_rot >= rel_s_rot:
        print("  [FAIL] dynamic should beat static-0.1 under Hadamard")
        ok = False
    else:
        print(f"  [OK]   dynamic rel_err {rel_d_rot:.4e} "
              f"<< static-0.1 rel_err {rel_s_rot:.4e}  "
              f"(ratio {rel_s_rot/max(rel_d_rot,1e-12):.1f}x)")
    # 3. Dynamic should be on par with hand-tuned static=2.0 under
    #    Hadamard (within 2x, since dynamic uses the exact amax).
    if rel_d_rot > 2.0 * rel_s_rot2 + 1e-6:
        print(f"  [WARN] dynamic ({rel_d_rot:.4e}) noticeably worse than "
              f"hand-tuned static=2.0 ({rel_s_rot2:.4e})")
    else:
        print(f"  [OK]   dynamic on par with hand-tuned static=2.0 "
              f"({rel_d_rot:.4e} vs {rel_s_rot2:.4e})")
    # 4. Without Hadamard, dynamic should also be no worse.
    if rel_d_un > rel_s_un * 1.5 + 1e-6:
        print(f"  [WARN] dynamic ({rel_d_un:.4e}) noticeably worse than "
              f"static-0.1 ({rel_s_un:.4e}) on un-rotated Q")
    else:
        print(f"  [OK]   dynamic no worse than static-0.1 on un-rotated Q "
              f"({rel_d_un:.4e} vs {rel_s_un:.4e})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
# === END DYNAMIC NVFP4 Q SCALE ===
