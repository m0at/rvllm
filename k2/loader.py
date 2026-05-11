"""Kimi K2.6 weight loader -- safetensors -> GPU/CPU split.

GPU gets: attention, shared experts, router, embedding, lm_head, norms.
CPU experts are loaded lazily from mmap'd safetensors during inference.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def dequant_int4_group32(packed: torch.Tensor, scale: torch.Tensor,
                          shape: torch.Tensor) -> torch.Tensor:
    """Dequantize compressed-tensors pack-quantized INT4 to BF16.
    Fully vectorized -- no Python loops.
    """
    out_f = int(shape[0].item())
    in_f = int(shape[1].item())
    n_groups = in_f // 32

    packed_bytes = packed.contiguous().view(torch.uint8)
    low = (packed_bytes & 0x0F).to(torch.int8) - 8
    high = ((packed_bytes >> 4) & 0x0F).to(torch.int8) - 8
    unpacked = torch.stack((low, high), dim=-1).reshape(-1)[:out_f * in_f]
    unpacked = unpacked.view(out_f, n_groups, 32).to(torch.bfloat16)

    s = scale.to(torch.bfloat16).view(out_f, n_groups, 1)
    return (unpacked * s).reshape(out_f, in_f)


class ShardCache:
    """Keeps safetensors file handles open for lazy reads."""

    def __init__(self, model_dir: Path, index: dict):
        self.model_dir = model_dir
        self.weight_map = index["weight_map"]
        self._handles: Dict[str, object] = {}

    def _get_handle(self, shard_file: str):
        if shard_file not in self._handles:
            path = str(self.model_dir / shard_file)
            self._handles[shard_file] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_file]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        h = self._get_handle(shard)
        return h.get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def close(self):
        self._handles.clear()


class K2Weights:
    """Kimi K2.6 weights with GPU/CPU split."""

    def __init__(self, model_dir: Path, device: str = "cuda:0",
                 expert_cache_size: int = 4096):
        self.model_dir = model_dir
        self.cfg = load_config(model_dir)
        self.device = torch.device(device)

        self.n_layers = self.cfg["num_hidden_layers"]
        self.hidden_size = self.cfg["hidden_size"]
        self.n_heads = self.cfg["num_attention_heads"]
        self.q_lora_rank = self.cfg["q_lora_rank"]
        self.kv_lora_rank = self.cfg["kv_lora_rank"]
        self.qk_nope_head_dim = self.cfg["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.cfg["qk_rope_head_dim"]
        self.v_head_dim = self.cfg["v_head_dim"]
        self.n_routed_experts = self.cfg["n_routed_experts"]
        self.n_experts_per_tok = self.cfg["num_experts_per_tok"]
        self.moe_intermediate_size = self.cfg["moe_intermediate_size"]
        self.intermediate_size = self.cfg["intermediate_size"]
        self.routed_scaling_factor = self.cfg["routed_scaling_factor"]
        self.first_k_dense = self.cfg.get("first_k_dense_replace", 1)
        self.vocab_size = self.cfg["vocab_size"]
        self.rope_theta = self.cfg["rope_theta"]
        self.rms_norm_eps = self.cfg["rms_norm_eps"]
        self.rope_scaling = self.cfg.get("rope_scaling", {})
        # Read the model's declared positional ceiling from config so
        # `model.py` can size RoPE tables to match instead of the
        # previous hard-coded `max_seq=8192`. Falls back to 8192
        # only when the field is genuinely absent (older configs).
        # Above this position, `apply_rope` would IndexError.
        self.max_position_embeddings = int(
            self.cfg.get("max_position_embeddings", 8192)
        )
        self.expert_cache_size = expert_cache_size

        index_path = model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        self.shards = ShardCache(model_dir, self.index)

        self.P = "language_model.model."
        self.LP = "language_model."

        self.embed = None
        self.lm_head = None
        self.final_norm = None
        self.layers_attn = []
        self.layers_norm = []
        self.layers_mlp_dense = []
        self.layers_mlp_shared = []
        self.layers_router = []
        self.get_expert_weight_dequant = lru_cache(maxsize=expert_cache_size)(
            self._get_expert_weight_dequant_uncached
        )

        self._load_gpu_weights()

    def _to_gpu(self, name: str) -> torch.Tensor:
        return self.shards.get(name).to(torch.bfloat16).to(self.device)

    def _load_gpu_weights(self):
        print("[loader] Loading GPU-resident weights...", file=sys.stderr)

        self.embed = self._to_gpu(f"{self.P}embed_tokens.weight")
        print(f"  embed: {self.embed.shape}", file=sys.stderr)

        lm_name = f"{self.LP}lm_head.weight"
        self.lm_head = self._to_gpu(lm_name)
        print(f"  lm_head: {self.lm_head.shape}", file=sys.stderr)

        self.final_norm = self._to_gpu(f"{self.P}norm.weight")

        for li in range(self.n_layers):
            lp = f"{self.P}layers.{li}"
            if li % 10 == 0:
                print(f"  layer {li}/{self.n_layers}...", file=sys.stderr)

            attn = {
                "input_layernorm": self._to_gpu(f"{lp}.input_layernorm.weight"),
                "q_a_proj": self._to_gpu(f"{lp}.self_attn.q_a_proj.weight"),
                "q_a_layernorm": self._to_gpu(f"{lp}.self_attn.q_a_layernorm.weight"),
                "q_b_proj": self._to_gpu(f"{lp}.self_attn.q_b_proj.weight"),
                "kv_a_proj": self._to_gpu(f"{lp}.self_attn.kv_a_proj_with_mqa.weight"),
                "kv_a_layernorm": self._to_gpu(f"{lp}.self_attn.kv_a_layernorm.weight"),
                "kv_b_proj": self._to_gpu(f"{lp}.self_attn.kv_b_proj.weight"),
                "o_proj": self._to_gpu(f"{lp}.self_attn.o_proj.weight"),
            }
            self.layers_attn.append(attn)

            norms = {
                "post_attn": self._to_gpu(f"{lp}.post_attention_layernorm.weight"),
            }
            self.layers_norm.append(norms)

            is_moe = li >= self.first_k_dense

            if is_moe:
                se = {
                    "gate_proj": self._to_gpu(f"{lp}.mlp.shared_experts.gate_proj.weight"),
                    "up_proj": self._to_gpu(f"{lp}.mlp.shared_experts.up_proj.weight"),
                    "down_proj": self._to_gpu(f"{lp}.mlp.shared_experts.down_proj.weight"),
                }
                self.layers_mlp_shared.append(se)

                router = {
                    "weight": self._to_gpu(f"{lp}.mlp.gate.weight"),
                    "e_score_correction_bias": self._to_gpu(f"{lp}.mlp.gate.e_score_correction_bias"),
                }
                self.layers_router.append(router)
                self.layers_mlp_dense.append(None)
            else:
                dense = {
                    "gate_proj": self._to_gpu(f"{lp}.mlp.gate_proj.weight"),
                    "up_proj": self._to_gpu(f"{lp}.mlp.up_proj.weight"),
                    "down_proj": self._to_gpu(f"{lp}.mlp.down_proj.weight"),
                }
                self.layers_mlp_dense.append(dense)
                self.layers_mlp_shared.append(None)
                self.layers_router.append(None)

        gpu_gb = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"[loader] GPU weights loaded: {gpu_gb:.1f} GB", file=sys.stderr)

    def _get_expert_weight_dequant_uncached(self, layer_idx: int, expert_idx: int,
                                            proj: str) -> torch.Tensor:
        """Lazy-load and dequantize one expert projection, cached via LRU."""
        lp = f"{self.P}layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
        packed = self.shards.get(f"{lp}.weight_packed")
        scale = self.shards.get(f"{lp}.weight_scale")
        shape = self.shards.get(f"{lp}.weight_shape")
        return dequant_int4_group32(packed, scale, shape)
