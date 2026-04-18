#!/usr/bin/env python3
"""Generate manifest.json for v3 kernel loader from a directory of PTX files.

Each entry maps the logical v3 name -> (relative path, sha256 digest, bytes).
"""
import hashlib
import json
import sys
from pathlib import Path

NAME_TO_FILE = {
    "fused_rmsnorm_fp8_quant": "fused_rmsnorm_fp8_quant.ptx",
    "fused_rope_cache": "fused_rope_cache.ptx",
    "fused_rope_cache_f16tbl": "fused_rope_cache_f16tbl.ptx",
    "fused_rope_cache_fp8kv": "fused_rope_cache_fp8kv.ptx",
    "fused_rope_partial_fp8kv": "fused_rope_partial_fp8kv.ptx",
    "add_bias_f16": "add_bias_f16.ptx",
    "fused_silu_fp8_quant": "fused_silu_fp8_quant.ptx",
    "fused_gelu_mul_fp8_quant": "fused_gelu_mul_fp8_quant.ptx",
    "argmax": "argmax.ptx",
    "quantize_activation_fp8": "quantize_activation_fp8.ptx",
    "fused_lm_head_argmax": "fused_lm_head_argmax.ptx",
    "embedding_gather": "embedding_gather.ptx",
    "embedding_gather_f16": "embedding_gather_f16.ptx",
    "fp8_rescale": "fp8_rescale.ptx",
    "rmsnorm_inplace_f16": "rmsnorm_inplace_f16.ptx",
    "fused_qk_rmsnorm": "fused_qk_rmsnorm.ptx",
    "logit_softcap": "logit_softcap.ptx",
    "residual_scale_f16": "residual_scale_f16.ptx",
    "vnorm_f16": "vnorm_f16.ptx",
    "vector_add_f16": "vector_add_f16.ptx",
    "bf16_to_f16_sat": "bf16_to_f16_sat.ptx",
    "rmsnorm_inplace_bf16": "rmsnorm_inplace_bf16.ptx",
    "vector_add_bf16_to_f16": "vector_add_bf16_to_f16.ptx",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <kernels_dir> <arch> [revision]")
    kernels_dir = Path(sys.argv[1])
    arch = sys.argv[2]
    revision = sys.argv[3] if len(sys.argv) >= 4 else "unknown"

    entries = {}
    for name, fname in NAME_TO_FILE.items():
        p = kernels_dir / fname
        if not p.exists():
            print(f"skip {name}: {p} not found", file=sys.stderr)
            continue
        entries[name] = {
            "path": fname,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
    manifest = {
        "revision": revision,
        "arch": arch,
        "entries": entries,
    }
    out = kernels_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(entries)} entries")


if __name__ == "__main__":
    main()
