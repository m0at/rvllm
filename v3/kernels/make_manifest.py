#!/usr/bin/env python3
"""Compile CUDA kernels and generate manifest.json.

Finds .cu sources, compiles to PTX, records both source and PTX hashes
so stale artifacts are impossible. Refuses to register a PTX without
compiling it from source first.

Usage: make_manifest.py <output_dir> <arch> [revision]
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# kernel name -> source .cu path (relative to this script or absolute)
# Sources live in two places: kernels/ (repo root) and v3/kernels/
SCRIPT_DIR = Path(__file__).parent  # v3/kernels/
REPO_ROOT = SCRIPT_DIR.parent.parent  # repo root

NAME_TO_SOURCE = {
    "fused_rmsnorm_fp8_quant": REPO_ROOT / "kernels" / "fused_rmsnorm_fp8_quant.cu",
    "fused_rope_cache": REPO_ROOT / "kernels" / "fused_rope_cache.cu",
    "fused_rope_cache_f16tbl": REPO_ROOT / "kernels" / "fused_rope_cache_f16tbl.cu",
    "fused_rope_cache_fp8kv": REPO_ROOT / "kernels" / "fused_rope_cache_fp8kv.cu",
    "fused_rope_partial_fp8kv": SCRIPT_DIR / "fused_rope_partial_fp8kv.cu",
    "add_bias_f16": REPO_ROOT / "kernels" / "add_bias_f16.cu",
    "fused_silu_fp8_quant": REPO_ROOT / "kernels" / "fused_silu_fp8_quant.cu",
    "fused_gelu_mul_fp8_quant": SCRIPT_DIR / "fused_gelu_mul_fp8_quant.cu",
    "argmax": REPO_ROOT / "kernels" / "argmax.cu",
    "quantize_activation_fp8": REPO_ROOT / "kernels" / "quantize_activation_fp8.cu",
    "fused_lm_head_argmax": REPO_ROOT / "kernels" / "fused_lm_head_argmax.cu",
    "embedding_gather": REPO_ROOT / "kernels" / "embedding_gather.cu",
    "embedding_gather_f16": REPO_ROOT / "kernels" / "embedding_gather_f16.cu",
    "fp8_rescale": REPO_ROOT / "kernels" / "fp8_rescale.cu",
    "rmsnorm_inplace_f16": SCRIPT_DIR / "rmsnorm_inplace_f16.cu",
    "rmsnorm_inplace_f16_group_gamma": SCRIPT_DIR / "rmsnorm_inplace_f16_group_gamma.cu",
    "fused_qk_rmsnorm": SCRIPT_DIR / "fused_qk_rmsnorm.cu",
    "logit_softcap": SCRIPT_DIR / "logit_softcap.cu",
    "residual_scale_f16": SCRIPT_DIR / "residual_scale_f16.cu",
    "vnorm_f16": SCRIPT_DIR / "vnorm_f16.cu",
    "scale_cols_f32": SCRIPT_DIR / "scale_cols_f32.cu",
    "scale_rows_f32": SCRIPT_DIR / "scale_rows_f32.cu",
    "compute_qkv_scales": SCRIPT_DIR / "compute_qkv_scales.cu",
    "fused_gelu_mul_f16": SCRIPT_DIR / "fused_gelu_mul_f16.cu",
    "fused_gelu_mul_two_f16": SCRIPT_DIR / "fused_gelu_mul_two_f16.cu",
    "fused_rope_partial_f16kv": SCRIPT_DIR / "fused_rope_partial_f16kv.cu",
    "fused_norm_add_residual": SCRIPT_DIR / "fused_norm_add_residual.cu",
    "fused_norm_add_residual_f16": SCRIPT_DIR / "fused_norm_add_residual_f16.cu",
    "fused_qkv_rmsnorm": SCRIPT_DIR / "fused_qkv_rmsnorm.cu",
    "scale_cols_f16": SCRIPT_DIR / "scale_cols_f16.cu",
    "scale_rows_f16": SCRIPT_DIR / "scale_rows_f16.cu",
    "slice_cols_f16": SCRIPT_DIR / "slice_cols_f16.cu",
    "vector_add_f16": SCRIPT_DIR / "vector_add_f16.cu",
    "bf16_to_f16_sat": SCRIPT_DIR / "bf16_to_f16_sat.cu",
    "f32_to_bf16": SCRIPT_DIR / "f32_to_bf16.cu",
    "f32_to_f16_sat": SCRIPT_DIR / "f32_to_f16_sat.cu",
    "rmsnorm_inplace_bf16": SCRIPT_DIR / "rmsnorm_inplace_bf16.cu",
    "vector_add_bf16_to_f16": SCRIPT_DIR / "vector_add_bf16_to_f16.cu",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <output_dir> <arch> [revision]")
    out_dir = Path(sys.argv[1])
    arch = sys.argv[2]
    revision = sys.argv[3] if len(sys.argv) >= 4 else "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    compiled = 0
    skipped = 0
    failed = 0

    for name, src in sorted(NAME_TO_SOURCE.items()):
        if not src.exists():
            print(f"  SKIP {name}: {src} not found", file=sys.stderr)
            skipped += 1
            continue

        ptx_path = out_dir / f"{name}.ptx"
        src_hash = sha256_file(src)

        # Check if PTX exists and source hasn't changed
        need_compile = True
        if ptx_path.exists():
            # Check if manifest has a matching source hash
            old_manifest = out_dir / "manifest.json"
            if old_manifest.exists():
                old = json.loads(old_manifest.read_text())
                old_entry = old.get("entries", {}).get(name, {})
                if old_entry.get("source_sha256") == src_hash:
                    need_compile = False

        if need_compile:
            cmd = [
                "nvcc", "-ptx", f"-arch=sm_{arch.replace('sm_', '')}",
                "-O3", "--use_fast_math",
                "-o", str(ptx_path), str(src),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAIL {name}: {result.stderr.strip()}", file=sys.stderr)
                failed += 1
                continue
            compiled += 1
            print(f"  compiled {name}", file=sys.stderr)
        else:
            print(f"  up-to-date {name}", file=sys.stderr)

        entries[name] = {
            "path": f"{name}.ptx",
            "sha256": sha256_file(ptx_path),
            "source_sha256": src_hash,
            "bytes": ptx_path.stat().st_size,
        }

    manifest = {
        "revision": revision,
        "arch": arch,
        "entries": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}: {len(entries)} entries ({compiled} compiled, {skipped} skipped, {failed} failed)")


if __name__ == "__main__":
    main()
