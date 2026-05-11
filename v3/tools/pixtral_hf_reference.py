"""HF Pixtral vision-tower + Mistral3 projector reference dumps.

Phase 3-test of the Round-12 Pixtral vision integration.

Loads the vision tower + multimodal projector from
/home/r00t/mistral-3.5 directly into HuggingFace's reference
implementations (PixtralVisionModel + Mistral3MultiModalProjector
from transformers 5.8) on CPU, runs a single test image, and dumps
per-stage activations as little-endian BF16 bytes — matching the
on-disk format rvllm-serve writes under
`RVLLM_DEBUG_MISTRAL35=1 RVLLM_PIXTRAL_DUMP_DIR=...`.

Stages dumped (matching rvllm's keys):
  patches_bf16       — host-preprocessed [N, 3*14*14] BF16 patches
  post_patch_conv    — conv output [N, v_hidden=1664] BF16
  post_ln_pre        — post pre-transformer RMSNorm [N, 1664] BF16
  post_blocks        — vision tower output (after 48 blocks) [N, 1664] BF16
  post_proj_norm     — post projector RMSNorm (still pre-merge) [N, 1664]
  post_merge         — after spatial 2x2 concat + merging_layer [N/4, 1664]
  output             — final projector output [N/4, 12288] BF16

Usage:
  python v3/tools/pixtral_hf_reference.py \\
    --image /path/to/test.png \\
    --out-dir /tmp/pixtral_hf_ref/

Then rerun rvllm-serve with `RVLLM_PIXTRAL_DUMP_DIR=/tmp/rvllm_dump/`
and use `pixtral_hf_compare.py` (separate, lands later) to walk the
two dump dirs and emit cosines per stage.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3MultiModalProjector,
)


def f32_to_bf16_bytes(x: torch.Tensor) -> bytes:
    """f32 / f64 / bf16 / f16 tensor → little-endian BF16 bytes."""
    bf16 = x.detach().to(torch.bfloat16).contiguous().cpu()
    raw = bf16.view(torch.uint16).numpy().tobytes()
    return raw


def load_vision_weights(model_dir: Path, vision_tower: PixtralVisionModel):
    """Pull `vision_tower.*` keys from the sharded safetensors into the
    HF module's state_dict — keys map 1:1 after stripping the
    `model.vision_tower.` prefix."""
    target = vision_tower.state_dict()
    found = 0
    needed = set(target.keys())
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(str(model_dir / fname), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("model.vision_tower."):
                    continue
                inner = k[len("model.vision_tower."):]
                if inner in target:
                    target[inner].copy_(f.get_tensor(k).to(target[inner].dtype))
                    needed.discard(inner)
                    found += 1
    if needed:
        raise RuntimeError(f"Missing vision-tower weights: {sorted(needed)[:5]}…")
    print(f"  loaded {found} vision-tower tensors")


def load_projector_weights(model_dir: Path, projector: nn.Module):
    target = projector.state_dict()
    found = 0
    needed = set(target.keys())
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(str(model_dir / fname), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("model.multi_modal_projector."):
                    continue
                inner = k[len("model.multi_modal_projector."):]
                if inner in target:
                    target[inner].copy_(f.get_tensor(k).to(target[inner].dtype))
                    needed.discard(inner)
                    found += 1
    if needed:
        raise RuntimeError(f"Missing projector weights: {sorted(needed)}")
    print(f"  loaded {found} projector tensors")


def stage_hook_dumper(out_dir: Path, name: str):
    """torch.nn.Module forward hook that dumps the module's primary
    output tensor as BF16 bytes."""
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        if not isinstance(t, torch.Tensor):
            return
        path = out_dir / f"{name}.bin"
        path.write_bytes(f32_to_bf16_bytes(t))
        print(f"  dumped {name}: shape={tuple(t.shape)} → {path}")
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/r00t/mistral-3.5")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Load image + processor (CPU, BF16-faithful preprocessing).
    img = Image.open(args.image).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(args.model_dir, use_fast=False)
    pp = processor(images=img, return_tensors="pt")
    # `pixel_values` shape is [1, 3, H, W]. mistral_common's processor
    # returns the un-patchified tensor; PixtralVisionModel patchifies
    # internally during patch_conv.
    pixel_values = pp["pixel_values"].to(device)
    if isinstance(pixel_values, list):
        # Some HF processor variants return a list of tensors (one per
        # image), normalise to a 4-D tensor.
        pixel_values = pixel_values[0]
    if pixel_values.dim() == 5:
        pixel_values = pixel_values[0]
    image_sizes = torch.tensor([list(pixel_values.shape[-2:])], device=device)
    print(f"  pixel_values: shape={tuple(pixel_values.shape)}, image_sizes={image_sizes.tolist()}")

    # ── 2. Build HF vision tower + projector with the exact config.
    config = AutoConfig.from_pretrained(args.model_dir)
    vision_tower = PixtralVisionModel(config.vision_config).to(
        device=device, dtype=torch.bfloat16,
    )
    projector = Mistral3MultiModalProjector(config).to(
        device=device, dtype=torch.bfloat16,
    )
    vision_tower.eval(); projector.eval()

    # ── 3. Load weights from disk.
    print("Loading vision tower weights…")
    load_vision_weights(model_dir, vision_tower)
    print("Loading projector weights…")
    load_projector_weights(model_dir, projector)

    # ── 4. Wire forward hooks at the stages we care about.
    handles = []
    handles.append(vision_tower.patch_conv.register_forward_hook(
        stage_hook_dumper(out_dir, "post_patch_conv")))
    handles.append(vision_tower.ln_pre.register_forward_hook(
        stage_hook_dumper(out_dir, "post_ln_pre")))
    handles.append(projector.norm.register_forward_hook(
        stage_hook_dumper(out_dir, "post_proj_norm")))
    handles.append(projector.patch_merger.register_forward_hook(
        stage_hook_dumper(out_dir, "post_merge")))
    handles.append(projector.linear_1.register_forward_hook(
        stage_hook_dumper(out_dir, "post_linear_1")))

    # Round-12 phase 3-test (c): per-block dumps for the bisect.
    # Hooks the residual stream after each PixtralAttentionLayer so the
    # compare script can locate the first block where rvllm drifts
    # from HF.
    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    for layer_idx, block in enumerate(vision_tower.transformer.layers):
        name = f"blocks/block_{layer_idx:02}"
        handles.append(block.register_forward_hook(
            stage_hook_dumper(out_dir, name)))

    # ── 5. Run the forward.
    print("Running HF Pixtral vision forward…")
    with torch.no_grad():
        # PixtralVisionModel returns (last_hidden_state, ...) in
        # newer transformers; we pull last_hidden_state explicitly.
        vision_out = vision_tower(
            pixel_values.to(torch.bfloat16),
            image_sizes=image_sizes,
        )
        post_blocks = (
            vision_out.last_hidden_state if hasattr(vision_out, "last_hidden_state")
            else vision_out[0]
        )
        # Dump post_blocks explicitly (the encoder's output module
        # may not match the hook target).
        (out_dir / "post_blocks.bin").write_bytes(f32_to_bf16_bytes(post_blocks))
        print(f"  dumped post_blocks: shape={tuple(post_blocks.shape)}")
        # Projector expects 2D [N, hidden] (HF feeds it the squeezed
        # vision-tower output via `image_features.view(-1, hidden)`).
        if post_blocks.dim() == 3:
            post_blocks_2d = post_blocks.view(-1, post_blocks.shape[-1])
        else:
            post_blocks_2d = post_blocks
        proj_out = projector(post_blocks_2d, image_sizes=image_sizes)
        (out_dir / "output.bin").write_bytes(f32_to_bf16_bytes(proj_out))
        print(f"  dumped output: shape={tuple(proj_out.shape)}")

    for h in handles:
        h.remove()
    print("Done.")


if __name__ == "__main__":
    main()
