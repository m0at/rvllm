#!/usr/bin/env python3
"""Dump HF Gemma4VisionEncoderLayer per-sub-step reference buffers for one
target block, matching the layout written by `forward_gemma_vision`'s
`dump_substep` helper (gate: `RVLLM_GEMMA4_VIT_SUBSTEP_BLK=<idx>`).

Use to diff bf16 wiring sub-step-by-sub-step against HF after the
existing `RVLLM_GEMMA4_VIT_DUMP_DIR` block-level audit confirms the
front + back of the tower stays byte-faithful.

Filenames written (all f16 little-endian, [N, D] row-major):
  g4v_blk{B}_input_ln.bin       [N, 1152]
  g4v_blk{B}_q_proj.bin         [N, 1152]
  g4v_blk{B}_k_proj.bin         [N, 1152]
  g4v_blk{B}_v_proj.bin         [N, 1152]
  g4v_blk{B}_q_norm.bin         [N, 1152]
  g4v_blk{B}_k_norm.bin         [N, 1152]
  g4v_blk{B}_v_norm.bin         [N, 1152]
  g4v_blk{B}_q_rot.bin          [N, 1152]
  g4v_blk{B}_k_rot.bin          [N, 1152]
  g4v_blk{B}_attn_out.bin       [N, 1152]
  g4v_blk{B}_o_proj.bin         [N, 1152]
  g4v_blk{B}_post_attn_ln.bin   [N, 1152]
  g4v_blk{B}_post_attn_resid.bin[N, 1152]
  g4v_blk{B}_pre_ff_ln.bin      [N, 1152]
  g4v_blk{B}_gate_proj.bin      [N, 4304]
  g4v_blk{B}_up_proj.bin        [N, 4304]
  g4v_blk{B}_gelu_mul.bin       [N, 4304]
  g4v_blk{B}_down_proj.bin      [N, 1152]
  g4v_blk{B}_post_ff_ln.bin     [N, 1152]

Diff against rvllm dumps with /tmp/cmp_g4v_substep.py (ad-hoc; adapt the
existing /tmp/cmp_g4v_stages.py — same f16 layout).

Usage:
  python3 v3/tools/gemma_vision_substep_hf_dump.py \\
    --image v3/crates/rvllm-runtime/tests/fixtures/test_224.png \\
    --block 0 \\
    --out  /tmp/hf_g4v_blk0
"""
import argparse, os, struct
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def write_f16(path: Path, t: torch.Tensor):
    arr = t.detach().to(torch.float16).cpu().contiguous().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  {path.name:<32} {tuple(arr.shape)}")


def patch_one_layer(layer, block_idx: int, out_dir: Path, ctx: dict):
    """Monkey-patch the target Gemma4VisionEncoderLayer.forward to emit
    every intermediate buffer matching dump_substep names."""
    orig = layer.forward

    def hooked(hidden_states, attention_mask=None, **kw):
        cfg = layer.config if hasattr(layer, "config") else None
        # 1) input_layernorm
        x = layer.input_layernorm(hidden_states)
        write_f16(out_dir / f"g4v_blk{block_idx}_input_ln.bin", x[0])
        # 2) Q/K/V proj
        attn = layer.self_attn
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        write_f16(out_dir / f"g4v_blk{block_idx}_q_proj.bin", q[0])
        write_f16(out_dir / f"g4v_blk{block_idx}_k_proj.bin", k[0])
        write_f16(out_dir / f"g4v_blk{block_idx}_v_proj.bin", v[0])
        # Reshape to [B, N, H, D]
        b, n, _ = q.shape
        h, d = attn.config.num_attention_heads, attn.head_dim
        q = q.view(b, n, h, d)
        k = k.view(b, n, h, d)
        v = v.view(b, n, h, d)
        # 3) q_norm / k_norm / v_norm (per-head)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        # v_norm: parameter-free RMS in rvllm; HF Gemma4 vision applies
        # an explicit rms-like normalisation only inside attn — record
        # both for parity. If HF lacks v_norm the rvllm bf16 wiring
        # would still want to compare against pre-rotary V.
        v_norm = v / (v.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).to(v.dtype)
        write_f16(out_dir / f"g4v_blk{block_idx}_q_norm.bin", q[0].reshape(n, h * d))
        write_f16(out_dir / f"g4v_blk{block_idx}_k_norm.bin", k[0].reshape(n, h * d))
        write_f16(out_dir / f"g4v_blk{block_idx}_v_norm.bin", v_norm[0].reshape(n, h * d))
        # 4) Rotary
        position_embeddings = ctx["pos_embeddings"]  # (cos, sin) precomputed
        cos, sin = position_embeddings
        from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
        q_rot, k_rot = apply_multidimensional_rope(q, k, cos, sin)
        write_f16(out_dir / f"g4v_blk{block_idx}_q_rot.bin", q_rot[0].reshape(n, h * d))
        write_f16(out_dir / f"g4v_blk{block_idx}_k_rot.bin", k_rot[0].reshape(n, h * d))
        # 5) Attention scaled-dot-product → output
        scale = 1.0  # Gemma vision: no 1/sqrt(d)
        scores = torch.einsum("bnhd,bmhd->bhnm", q_rot.float(), k_rot.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhnm,bmhd->bnhd", probs, v.float()).to(v.dtype)
        write_f16(out_dir / f"g4v_blk{block_idx}_attn_out.bin", attn_out[0].reshape(n, h * d))
        # 6) o_proj
        attn_flat = attn_out.reshape(b, n, h * d)
        proj = attn.o_proj(attn_flat)
        write_f16(out_dir / f"g4v_blk{block_idx}_o_proj.bin", proj[0])
        # 7) post_attn_ln + residual
        post_attn = layer.post_attention_layernorm(proj)
        write_f16(out_dir / f"g4v_blk{block_idx}_post_attn_ln.bin", post_attn[0])
        h_after_attn = hidden_states + post_attn
        write_f16(out_dir / f"g4v_blk{block_idx}_post_attn_resid.bin", h_after_attn[0])
        # 8) FFN
        x = layer.pre_feedforward_layernorm(h_after_attn)
        write_f16(out_dir / f"g4v_blk{block_idx}_pre_ff_ln.bin", x[0])
        mlp = layer.mlp
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        write_f16(out_dir / f"g4v_blk{block_idx}_gate_proj.bin", gate[0])
        write_f16(out_dir / f"g4v_blk{block_idx}_up_proj.bin", up[0])
        gelu = torch.nn.functional.gelu(gate, approximate="tanh") * up
        write_f16(out_dir / f"g4v_blk{block_idx}_gelu_mul.bin", gelu[0])
        down = mlp.down_proj(gelu)
        write_f16(out_dir / f"g4v_blk{block_idx}_down_proj.bin", down[0])
        post_ff = layer.post_feedforward_layernorm(down)
        write_f16(out_dir / f"g4v_blk{block_idx}_post_ff_ln.bin", post_ff[0])
        # Return original output so the rest of the model still runs.
        return orig(hidden_states, attention_mask, **kw)

    layer.forward = hooked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--image", required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {args.device}…")
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()

    vision = model.vision_tower
    encoder = vision.vision_model.encoder
    target = encoder.layers[args.block]

    # Pre-compute rotary embeddings the same way HF does inside
    # the encoder forward; stash on a context dict the hook reads.
    img = Image.open(args.image).convert("RGB")
    inputs = proc(images=img, text="<image>", return_tensors="pt").to(args.device)
    with torch.inference_mode():
        # Drive the vision tower so the hook fires.
        # We rely on HF computing position_embeddings inside the encoder
        # forward — we mirror the patcher's needs by capturing them via
        # an outer hook on the encoder.
        captured = {}
        def cap_pos(_module, _inp, _out):
            # _inp is (hidden_states, attention_mask, position_embeddings)
            try:
                captured["pos"] = _inp[2]
            except Exception:
                pass
        h_handle = encoder.register_forward_pre_hook(cap_pos)
        # Run a no-op forward up to the encoder to get pos embeddings.
        _ = vision(inputs["pixel_values"])
        h_handle.remove()
        ctx = {"pos_embeddings": captured.get("pos")}
        if ctx["pos_embeddings"] is None:
            print("WARN: could not capture rotary cos/sin from encoder hook; "
                  "q_rot/k_rot dumps will be skipped.")

    patch_one_layer(target, args.block, out_dir, ctx)

    print(f"dumping block {args.block} sub-steps → {out_dir}")
    with torch.inference_mode():
        _ = vision(inputs["pixel_values"])
    print("done.")


if __name__ == "__main__":
    main()
