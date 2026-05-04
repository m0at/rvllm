"""rvllm-vision-sidecar: HTTP service that runs the vision tower + projector
of either Qwen 3.6 (Qwen3_5MoeForConditionalGeneration) or Gemma 4 31B
(Gemma4ForConditionalGeneration) and returns f16 token-level embeddings
ready to be spliced into rvllm-serve's post-embed hidden buffer.

Loaded model is selected at startup via env RVLLM_VISION_MODEL ∈
{qwen3_vl, gemma4_mm}. Loading uses meta-init for the full model then
materializes ONLY the vision tower + projector / embedder, so the
language-model weights never hit GPU memory.

API:
  GET  /health                  → {model, cuda, hidden_dim}
  POST /embed                   → forward one image
       body  {image_b64, mime}  (mime is informational; PIL detects)
       resp  {num_tokens, hidden_dim, dtype, embeddings_b64,
              grid_thw?: [t,h,w]}

Embeddings are f16, layout [num_tokens, hidden_dim], serialized as raw
little-endian bytes then base64. Caller (Rust rvllm-serve) decodes with
half::f16 and HtoD-blits into the right post-embed slot.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from typing import Any

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ----- Config from env -----------------------------------------------------

VISION_MODEL = os.environ.get("RVLLM_VISION_MODEL", "qwen3_vl")
QWEN_PATH = os.environ.get(
    "QWEN36_PATH", "/home/r00t/.vllm/models/qwen3-6-35b-a3b-fp8"
)
GEMMA_PATH = os.environ.get(
    "GEMMA4_PATH", "/home/r00t/.vllm/models/gemma-4-31b-it-fp8-block"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


def _materialize_submodules(model: torch.nn.Module, prefixes: list[str]) -> None:
    """Move the named submodule(s) off the meta device onto DEVICE in fp16."""
    for prefix in prefixes:
        sub: torch.nn.Module | None = model
        for part in prefix.split("."):
            if not part:
                continue
            sub = getattr(sub, part)
        if sub is None:
            raise RuntimeError(f"submodule {prefix!r} not found")
        sub.to_empty(device=DEVICE)


def _load_state_dict_filtered(
    model_dir: str, key_filter: callable
) -> dict[str, torch.Tensor]:
    """Walk *.safetensors in model_dir and return all tensors whose key passes
    `key_filter(name)`."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fname)
        with safe_open(path, framework="pt") as sf:
            for key in sf.keys():
                if key_filter(key):
                    out[key] = sf.get_tensor(key).to(DTYPE)
    return out


# ----- Qwen 3.6 vision loader ---------------------------------------------


class QwenVisionStack:
    """Holds materialized model.visual + processor for Qwen3-VL."""

    hidden_dim = 2048  # out_hidden_size

    def __init__(self, model_dir: str):
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen3_5MoeForConditionalGeneration,
        )

        t0 = time.time()
        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        with torch.device("meta"):
            full = Qwen3_5MoeForConditionalGeneration(self.config)
        # Materialize ONLY the visual tower
        _materialize_submodules(full, ["model.visual"])
        # Load weights for visual.*
        sd = _load_state_dict_filtered(
            model_dir, lambda k: k.startswith("model.visual.")
        )
        # The submodule's own state_dict expects keys without the
        # 'model.visual.' prefix
        sd = {k[len("model.visual.") :]: v for k, v in sd.items()}
        missing, unexpected = full.model.visual.load_state_dict(
            sd, strict=False
        )
        if unexpected:
            print(
                f"[qwen-vision] WARN unexpected keys: {unexpected[:5]} (total {len(unexpected)})",
                file=sys.stderr,
            )
        if missing:
            # Some buffers may be re-init at first forward; warn but proceed.
            print(
                f"[qwen-vision] missing keys: {missing[:5]} (total {len(missing)})",
                file=sys.stderr,
            )
        full.model.visual.to(DTYPE).eval()
        # Drop language model to free meta-allocations
        self._full = full  # keep for get_image_features delegation
        print(
            f"[qwen-vision] loaded in {time.time()-t0:.1f}s on {DEVICE}",
            file=sys.stderr,
        )

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> tuple[torch.Tensor, list[int]]:
        # Use HF processor to get pixel_values + image_grid_thw
        proc_out = self.processor(
            text=["<|image_pad|>"],
            images=[image],
            return_tensors="pt",
        )
        pixel_values = proc_out["pixel_values"].to(DEVICE, dtype=DTYPE)
        grid_thw = proc_out["image_grid_thw"].to(DEVICE)
        # Forward through the visual tower. Qwen3VLVisionModel forward
        # signature: forward(pixel_values, grid_thw) → (num_tokens, hidden)
        # but the high-level model's get_image_features accepts the same
        # named args and returns a list of per-image tensors.
        # We invoke the visual tower directly to avoid touching the
        # un-materialized language model.
        out = self._full.model.visual(pixel_values, grid_thw=grid_thw)
        # Qwen3VLVisionModel returns BaseModelOutputWithDeepstackFeatures:
        #   last_hidden_state: pre-merge per-patch hidden  [seq, hidden=1152]
        #   pooler_output:     post-PatchMerger embeddings [num_tokens, 2048]
        #   deepstack_features: list of auxiliary merger outputs (skipped here)
        # The post-merge tensor is what vLLM splices into the text stream.
        feats = out.pooler_output
        thw = grid_thw[0].tolist()
        return feats.to(DTYPE).contiguous(), thw


# ----- Gemma 4 vision loader ----------------------------------------------


class GemmaVisionStack:
    """Holds materialized model.vision_tower + model.embed_vision + processor."""

    hidden_dim: int = 0  # filled at init

    def __init__(self, model_dir: str):
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Gemma4ForConditionalGeneration,
        )

        t0 = time.time()
        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        text_hidden = self.config.text_config.hidden_size
        self.hidden_dim = text_hidden
        with torch.device("meta"):
            full = Gemma4ForConditionalGeneration(self.config)
        _materialize_submodules(
            full, ["model.vision_tower", "model.embed_vision"]
        )
        # Load only vision_tower.* and embed_vision.* state dict entries
        sd_vt = _load_state_dict_filtered(
            model_dir, lambda k: k.startswith("model.vision_tower.")
        )
        sd_vt = {k[len("model.vision_tower.") :]: v for k, v in sd_vt.items()}
        full.model.vision_tower.load_state_dict(sd_vt, strict=False)
        sd_ev = _load_state_dict_filtered(
            model_dir, lambda k: k.startswith("model.embed_vision.")
        )
        sd_ev = {k[len("model.embed_vision.") :]: v for k, v in sd_ev.items()}
        full.model.embed_vision.load_state_dict(sd_ev, strict=False)
        full.model.vision_tower.to(DTYPE).eval()
        full.model.embed_vision.to(DTYPE).eval()
        self._full = full
        print(
            f"[gemma-vision] loaded in {time.time()-t0:.1f}s, hidden_dim={text_hidden}",
            file=sys.stderr,
        )

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> tuple[torch.Tensor, list[int] | None]:
        proc_out = self.processor(
            text=["<image>"],
            images=[image],
            return_tensors="pt",
            add_special_tokens=False,
        )
        pixel_values = proc_out["pixel_values"].to(DEVICE, dtype=DTYPE)
        # Gemma4 processor names it `image_position_ids`; the model arg
        # is `pixel_position_ids`.
        pixel_position_ids = proc_out["image_position_ids"].to(DEVICE)
        # Forward: vision_tower returns BaseModelOutputWithPast whose
        # last_hidden_state is [num_valid_soft_tokens, vision_hidden=768]
        # (vLLM strips padding internally per gemma4_mm.py:230-258).
        vt_out = self._full.model.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )
        vision_hidden = vt_out.last_hidden_state
        # embed_vision: linear (768→text_hidden) + RMSNorm post-projection.
        embeds = self._full.model.embed_vision(vision_hidden)
        if embeds.dim() == 3 and embeds.shape[0] == 1:
            embeds = embeds.squeeze(0)
        return embeds.to(DTYPE).contiguous(), None


# ----- App lifecycle -------------------------------------------------------


app = FastAPI(title="rvllm-vision-sidecar")
STACK: QwenVisionStack | GemmaVisionStack | None = None


@app.on_event("startup")
def _load_model() -> None:
    global STACK
    if VISION_MODEL == "qwen3_vl":
        STACK = QwenVisionStack(QWEN_PATH)
    elif VISION_MODEL == "gemma4_mm":
        STACK = GemmaVisionStack(GEMMA_PATH)
    else:
        raise RuntimeError(
            f"unsupported RVLLM_VISION_MODEL={VISION_MODEL!r}"
        )


# ----- Schemas -------------------------------------------------------------


class EmbedRequest(BaseModel):
    image_b64: str
    mime: str | None = None


class EmbedResponse(BaseModel):
    num_tokens: int
    hidden_dim: int
    dtype: str
    embeddings_b64: str
    grid_thw: list[int] | None = None


class HealthResponse(BaseModel):
    model: str
    cuda: bool
    hidden_dim: int


# ----- Endpoints -----------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if STACK is None:
        raise HTTPException(503, "model not loaded")
    return HealthResponse(
        model=VISION_MODEL,
        cuda=torch.cuda.is_available(),
        hidden_dim=STACK.hidden_dim,
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if STACK is None:
        raise HTTPException(503, "model not loaded")
    try:
        raw = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"bad image: {e}") from e
    try:
        embeds, grid_thw = STACK.embed(img)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"vision forward failed: {e}") from e
    if embeds.dim() != 2:
        raise HTTPException(
            500, f"unexpected embed shape {tuple(embeds.shape)}"
        )
    num_tokens, hidden_dim = embeds.shape
    if hidden_dim != STACK.hidden_dim:
        raise HTTPException(
            500,
            f"hidden_dim mismatch: got {hidden_dim} expected {STACK.hidden_dim}",
        )
    raw = embeds.cpu().numpy().tobytes()
    return EmbedResponse(
        num_tokens=int(num_tokens),
        hidden_dim=int(hidden_dim),
        dtype="float16",
        embeddings_b64=base64.b64encode(raw).decode("ascii"),
        grid_thw=grid_thw,
    )
