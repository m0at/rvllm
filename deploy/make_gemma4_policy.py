#!/usr/bin/env python3
"""Emit policy.json covering all Gemma4 decode GEMM shapes for a sweep of batch buckets.

Uses the canonical variant catalog from v3/crates/rvllm-cutlass/src/variants.rs:
  VariantId(0)   = FP8_GEMM_COOP_128_128_128    (non-residual, Coop/Coop)
  VariantId(100) = FP8_GEMM_RESIDUAL_COOP       (residual-fused, Coop/Coop)

Reads <model_dir>/config.json and computes (qkv_rows, gate_up_rows, o_rows,
down_rows, lm_head_rows) for both sliding and global attention using the
correct per-path KV head counts:
  sliding: num_key_value_heads, head_dim
  global:  num_global_key_value_heads (fallback num_key_value_heads),
           global_head_dim (fallback head_dim)
"""
import json, sys
from pathlib import Path

def vd(vid, tile, mainloop, epilogue):
    return {
        "id": vid,
        "tile":    {"m": tile[0], "n": tile[1], "k": tile[2]},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": mainloop,
        "epilogue": epilogue,
    }

CANONICAL = [
    vd(0,   (128, 128, 128), "Coop", "Coop"),
    vd(1,   (128, 256, 128), "Coop", "Coop"),
    vd(2,   ( 64, 128, 128), "WS",   "WS"),
    vd(3,   (128, 128, 128), "Fp8Coop", "Fp8Coop"),
    vd(4,   ( 64, 128, 128), "Fp8WS",   "Fp8WS"),
    vd(100, (128, 128, 128), "Coop", "Coop"),
]

def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <model_dir> <out_policy_json>")
    model_dir = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    cfg = json.loads((model_dir / "config.json").read_text())
    tc  = cfg.get("text_config", cfg)

    hidden     = int(tc["hidden_size"])
    n_heads    = int(tc["num_attention_heads"])
    n_kv_s     = int(tc["num_key_value_heads"])
    n_kv_g     = int(tc.get("num_global_key_value_heads") or n_kv_s)
    head_dim_s = int(tc.get("head_dim", 256))
    head_dim_g = int(tc.get("global_head_dim", head_dim_s))
    inter      = int(tc["intermediate_size"])
    vocab      = int(tc["vocab_size"])

    q_dim_s    = n_heads * head_dim_s
    kv_dim_s   = n_kv_s  * head_dim_s
    qkv_rows_s = q_dim_s + 2 * kv_dim_s

    q_dim_g    = n_heads * head_dim_g
    kv_dim_g   = n_kv_g  * head_dim_g
    qkv_rows_g = q_dim_g + 2 * kv_dim_g

    print(f"[policy] hidden={hidden} heads={n_heads} n_kv_s={n_kv_s} n_kv_g={n_kv_g} "
          f"head_dim_s={head_dim_s} head_dim_g={head_dim_g} "
          f"inter={inter} vocab={vocab}", file=sys.stderr)
    print(f"[policy] qkv_rows_s={qkv_rows_s} qkv_rows_g={qkv_rows_g} "
          f"q_dim_s={q_dim_s} q_dim_g={q_dim_g}", file=sys.stderr)

    BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]

    ws = 16 * 1024 * 1024
    entries = {}

    def add_nonres(m, n, k):
        entries[f"{m}_{n}_{k}_Fp8E4M3"]     = {"variant": 0,   "workspace_bytes": ws}
    def add_res(m, n, k):
        entries[f"{m}_{n}_{k}_Fp8E4M3_res"] = {"variant": 100, "workspace_bytes": ws}

    for m in BUCKETS:
        add_nonres(m, qkv_rows_s, hidden)
        if (qkv_rows_g, hidden) != (qkv_rows_s, hidden):
            add_nonres(m, qkv_rows_g, hidden)
        add_nonres(m, 2 * inter, hidden)
        add_nonres(m, vocab,     hidden)
        add_res(m, hidden, q_dim_s)
        if q_dim_g != q_dim_s:
            add_res(m, hidden, q_dim_g)
        add_res(m, hidden, inter)

    policy = {
        "revision": "gemma4-bringup",
        "arch": "sm_90",
        "variants": CANONICAL,
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2))
    print(f"[policy] wrote {out_path}: {len(entries)} entries across {len(BUCKETS)} buckets",
          file=sys.stderr)

if __name__ == "__main__":
    main()
