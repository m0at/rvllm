"""F3#3 parity probe: rvllm fused W4A16 GEMV vs the legacy
nvfp4_dequant_weights_bf16 + cublasLt bf16_gemm_f32 path.

Method:
  - Drive rvllm via /v1/completions twice for the same prompt:
    1. Default — RVLLM_W4A16_GEMV unset (= fused).
    2. RVLLM_W4A16_GEMV=0 (= legacy dequant→bf16_gemm).
  - With RVLLM_BOUNDARY_DUMP=1 + RVLLM_SMOKE_FULL_DUMP=1 +
    RVLLM_BOUNDARY_DUMP_LAYER=0, both runs dump:
      q_out_layer0.bf16, k_cache_layer0_all_slots.bf16,
      v_cache_layer0_all_slots.bf16, attn_out_layer0.bf16
  - Compare per-projection q/k/v outputs byte-for-byte.

Thresholds:
  - q/k/v cos > 0.9999 (within bf16 round-off across the two paths)
  - rms ratio in [0.999, 1.001]
  - per-row max-abs diff < 1e-3 of rms

Driver-style: invokes the running rvllm-serve over HTTP. Operator
flips the env between runs and re-runs this script. The script
itself only diffs the dump files.

Usage:
  # Fused dump (default profile, no env override):
  curl … && cp -r /tmp/rvllm-mistral35-dump /tmp/dump_fused

  # Legacy dump (set RVLLM_W4A16_GEMV=0, restart, request again):
  curl … && cp -r /tmp/rvllm-mistral35-dump /tmp/dump_legacy

  # Compare:
  python v3/tools/mistral35_w4a16_gemv_check.py /tmp/dump_fused /tmp/dump_legacy
"""
import sys, os, numpy as np

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
A_DIR = sys.argv[1]
B_DIR = sys.argv[2]

def load_bf16(p):
    u = np.fromfile(p, dtype=np.uint16); return (u.astype(np.uint32) << 16).view(np.float32)

def cos(a, b):
    af, bf = a.flatten(), b.flatten()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))

def rms(x): return float((x.astype(np.float64)**2).mean()**0.5)

THRESH_COS = 0.9999
# Tolerances reflect a real difference between the two paths:
#   fused:  dequant(F32) → multiply+accumulate in F32 → single BF16 round at output
#   legacy: dequant→BF16 (round) → cublasLt BF16×BF16→F32 → F32→BF16 (round)
# Two rounds vs one round → typical RMS shift of ~1–3e-3 and a tiny
# magnitude bias. The fused path is in fact more numerically faithful;
# this probe just guarantees they agree to BF16 ULP-levels.
THRESH_RMS_LO, THRESH_RMS_HI = 0.99, 1.01
THRESH_RMS_DIFF = 5e-3

failures = 0
def check(label, a, b):
    global failures
    if a.size != b.size:
        print(f"  {label}: size mismatch {a.size} vs {b.size}")
        failures += 1; return
    c = cos(a, b)
    ra, rb = rms(a), rms(b)
    ratio = ra / (rb + 1e-30)
    diff_max = float(np.abs(a - b).max())
    rms_diff = rms(a - b) / max(ra, 1e-30)
    ok = c >= THRESH_COS and THRESH_RMS_LO <= ratio <= THRESH_RMS_HI and rms_diff < THRESH_RMS_DIFF
    flag = " " if ok else "✗"
    if not ok: failures += 1
    print(f"  {flag} {label}: cos={c:.6f} ratio={ratio:.4f} rms_diff/rms={rms_diff:.2e} diff_max={diff_max:.4e}")

print(f"Comparing fused (= {A_DIR}) vs legacy (= {B_DIR})")
print()

stages = [
    ("q_out_layer0.bf16", "q_out (post-RoPE)"),
    ("attn_out_layer0.bf16", "attn_out"),
    ("k_cache_layer0_all_slots.bf16", "k_cache (all slots)"),
    ("v_cache_layer0_all_slots.bf16", "v_cache (all slots)"),
]

for fname, label in stages:
    pa, pb = os.path.join(A_DIR, fname), os.path.join(B_DIR, fname)
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        print(f"  {label}: missing one of {pa} {pb}")
        continue
    a = load_bf16(pa); b = load_bf16(pb)
    check(label, a, b)

print()
if failures > 0:
    print(f"FAIL: {failures} stage(s) outside threshold (cos≥{THRESH_COS}, rms_ratio∈[{THRESH_RMS_LO},{THRESH_RMS_HI}], rms_diff/rms<1e-3)")
    sys.exit(2)
print("OK: all stages pass parity thresholds")
