"""Kimi K2.6 forward pass -- MLA attention on GPU, MoE experts on CPU."""

import math
import time
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from typing import List


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight.float()).to(dtype)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_rope(dim: int, max_seq: int, theta: float, scaling: dict,
                     device: torch.device) -> torch.Tensor:
    factor = scaling.get("factor", 1.0)
    beta_fast = scaling.get("beta_fast", 32.0)
    beta_slow = scaling.get("beta_slow", 1.0)
    orig_max = scaling.get("original_max_position_embeddings", 4096)
    mscale_all = scaling.get("mscale_all_dim", 1.0)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    if factor > 1.0:
        low = math.floor(dim * math.log(orig_max / (beta_fast * 2 * math.pi)) / (2 * math.log(theta)))
        high = math.ceil(dim * math.log(orig_max / (beta_slow * 2 * math.pi)) / (2 * math.log(theta)))
        low = max(low, 0)
        high = min(high, dim // 2 - 1)
        smooth = torch.zeros(dim // 2, device=device)
        for i in range(dim // 2):
            if i < low:
                smooth[i] = 0.0
            elif i > high:
                smooth[i] = 1.0
            else:
                smooth[i] = (i - low) / (high - low) if high > low else 1.0
        freqs = (1 - smooth) * (freqs / factor) + smooth * freqs

    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    attn_mscale = yarn_get_mscale(factor, mscale_all)
    cos = torch.cos(angles) * attn_mscale
    sin = torch.sin(angles) * attn_mscale
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: int) -> torch.Tensor:
    """x: [B, n_heads, rope_dim]

    Indexing `cos[pos]` / `sin[pos]` directly used to IndexError silently
    once `pos >= cos.shape[0]` (the precomputed RoPE table length, set
    from `max_position_embeddings`). Raise a clear error pointing at the
    table-size bound instead of letting a generic IndexError surface
    deep inside the attention layer.
    """
    if pos >= cos.shape[0]:
        raise ValueError(
            f"K2 RoPE position {pos} exceeds precomputed table size "
            f"{cos.shape[0]}. Increase `max_position_embeddings` in the "
            f"model config (K2Model picks it up automatically) or cap "
            f"the request prompt+generation length below the table size."
        )
    B, nh, d = x.shape
    x = x.view(B, nh, d // 2, 2)
    c = cos[pos].unsqueeze(0).unsqueeze(0)
    s = sin[pos].unsqueeze(0).unsqueeze(0)
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
    return out.view(B, nh, d)


class K2Model:
    def __init__(self, weights, device: str = "cuda:0",
                 expert_workers: int | None = None):
        self.w = weights
        self.device = torch.device(device)
        self.n_layers = weights.n_layers
        self.n_heads = weights.n_heads
        self.q_lora_rank = weights.q_lora_rank
        self.kv_lora_rank = weights.kv_lora_rank
        self.qk_nope_head_dim = weights.qk_nope_head_dim
        self.qk_rope_head_dim = weights.qk_rope_head_dim
        self.v_head_dim = weights.v_head_dim
        self.hidden_size = weights.hidden_size
        self.n_routed_experts = weights.n_routed_experts
        self.n_experts_per_tok = weights.n_experts_per_tok
        self.routed_scaling_factor = weights.routed_scaling_factor
        self.first_k_dense = weights.first_k_dense
        self.rms_norm_eps = weights.rms_norm_eps

        # RoPE table size matches the model's declared
        # max_position_embeddings (read from config in `K2Weights`).
        # The previous `max_seq=8192` hard-coding silently truncated
        # any model whose config declared a larger ceiling and made
        # `apply_rope` IndexError above pos 8191 on long contexts.
        self.max_position_embeddings = int(
            getattr(weights, "max_position_embeddings", 8192)
        )
        self.rope_cos, self.rope_sin = precompute_rope(
            self.qk_rope_head_dim,
            max_seq=self.max_position_embeddings,
            theta=weights.rope_theta,
            scaling=weights.rope_scaling,
            device=self.device,
        )

        self.kv_cache = [{"ckv": [], "k_rope": []} for _ in range(self.n_layers)]
        max_workers = expert_workers if expert_workers is not None else self.n_experts_per_tok
        max_workers = max(1, min(max_workers, self.n_experts_per_tok))
        self.expert_workers = max_workers
        self.cpu_pool = ThreadPoolExecutor(max_workers=max_workers)

    def clear_cache(self):
        for c in self.kv_cache:
            c["ckv"].clear()
            c["k_rope"].clear()

    def mla_attention(self, x: torch.Tensor, layer_idx: int, pos: int) -> torch.Tensor:
        attn = self.w.layers_attn[layer_idx]
        B = x.shape[0]

        c_q = F.linear(x, attn["q_a_proj"])
        c_q = rms_norm(c_q, attn["q_a_layernorm"], self.rms_norm_eps)
        q_full = F.linear(c_q, attn["q_b_proj"])
        q_full = q_full.view(B, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope = q_full[:, :, :self.qk_nope_head_dim]
        q_rope = q_full[:, :, self.qk_nope_head_dim:]

        kv_a = F.linear(x, attn["kv_a_proj"])
        c_kv = kv_a[:, :self.kv_lora_rank]
        k_rope_raw = kv_a[:, self.kv_lora_rank:]
        c_kv = rms_norm(c_kv, attn["kv_a_layernorm"], self.rms_norm_eps)

        k_rope_raw = k_rope_raw.unsqueeze(1).expand(B, self.n_heads, self.qk_rope_head_dim)
        q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin, pos)
        k_rope_cur = apply_rope(k_rope_raw, self.rope_cos, self.rope_sin, pos)

        cache = self.kv_cache[layer_idx]
        cache["ckv"].append(c_kv)
        cache["k_rope"].append(k_rope_cur[:, 0:1, :])

        all_ckv = torch.cat(cache["ckv"], dim=0)
        all_k_rope = torch.cat(cache["k_rope"], dim=0)
        seq_len = all_ckv.shape[0]

        kv_full = F.linear(all_ckv, attn["kv_b_proj"])
        kv_full = kv_full.view(seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_full[:, :, :self.qk_nope_head_dim]
        v = kv_full[:, :, self.qk_nope_head_dim:]

        k_rope_all = all_k_rope.expand(seq_len, self.n_heads, self.qk_rope_head_dim)

        scores_nope = torch.einsum("bhd,shd->bhs", q_nope, k_nope)
        scores_rope = torch.einsum("bhd,shd->bhs", q_rope, k_rope_all)
        scale = math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        scores = (scores_nope + scores_rope) / scale

        if seq_len > 1:
            mask = torch.full((1, 1, seq_len), float("-inf"), device=self.device, dtype=scores.dtype)
            mask[0, 0, :pos + 1] = 0.0
            scores = scores + mask

        attn_w = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.einsum("bhs,shd->bhd", attn_w, v)
        out = out.reshape(B, self.n_heads * self.v_head_dim)
        return F.linear(out, attn["o_proj"])

    def dense_mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        d = self.w.layers_mlp_dense[layer_idx]
        gate = F.linear(x, d["gate_proj"])
        up = F.linear(x, d["up_proj"])
        return F.linear(F.silu(gate) * up, d["down_proj"])

    def shared_expert_mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        se = self.w.layers_mlp_shared[layer_idx]
        gate = F.linear(x, se["gate_proj"])
        up = F.linear(x, se["up_proj"])
        return F.linear(F.silu(gate) * up, se["down_proj"])

    def _run_expert_cpu(self, x_cpu: torch.Tensor, layer_idx: int, expert_idx: int) -> torch.Tensor:
        x_vec = x_cpu.squeeze(0)
        gate_w = self.w.get_expert_weight_dequant(layer_idx, expert_idx, "gate_proj")
        up_w = self.w.get_expert_weight_dequant(layer_idx, expert_idx, "up_proj")
        down_w = self.w.get_expert_weight_dequant(layer_idx, expert_idx, "down_proj")
        gate = torch.mv(gate_w, x_vec)
        up = torch.mv(up_w, x_vec)
        down = torch.mv(down_w, F.silu(gate) * up)
        return down.unsqueeze(0)

    def moe_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        router = self.w.layers_router[layer_idx]

        logits = F.linear(x, router["weight"])
        scores = torch.sigmoid(logits)
        scores = scores + router["e_score_correction_bias"]

        topk_scores, topk_idx = torch.topk(scores, self.n_experts_per_tok, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores * self.routed_scaling_factor

        x_cpu = x.to(dtype=torch.bfloat16, device="cpu")
        expert_ids = topk_idx[0].tolist()

        shared_out = self.shared_expert_mlp(x, layer_idx)
        combined = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        w_cpu = topk_scores[0].float().cpu()

        if self.expert_workers == 1:
            for i, eid in enumerate(expert_ids):
                combined += w_cpu[i] * self._run_expert_cpu(x_cpu, layer_idx, eid).float()
        else:
            futures = [
                self.cpu_pool.submit(self._run_expert_cpu, x_cpu, layer_idx, eid)
                for eid in expert_ids
            ]
            for i, future in enumerate(futures):
                combined += w_cpu[i] * future.result().float()

        routed_out = combined.to(x.dtype).to(self.device)
        return shared_out + routed_out

    def forward_step(self, token_id: int, pos: int) -> torch.Tensor:
        x = self.w.embed[token_id].unsqueeze(0)

        for li in range(self.n_layers):
            attn_w = self.w.layers_attn[li]
            norm_w = self.w.layers_norm[li]

            h = rms_norm(x, attn_w["input_layernorm"], self.rms_norm_eps)
            h = self.mla_attention(h, li, pos)
            x = x + h

            h = rms_norm(x, norm_w["post_attn"], self.rms_norm_eps)

            if li < self.first_k_dense:
                h = self.dense_mlp(h, li)
            else:
                h = self.moe_layer(h, li)

            x = x + h

        x = rms_norm(x, self.w.final_norm, self.rms_norm_eps)
        return F.linear(x, self.w.lm_head)

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_tokens: int = 100,
                 temperature: float = 0.6, top_p: float = 0.95) -> List[int]:
        self.clear_cache()
        generated = []
        eos = 163586

        print(f"[gen] Prefill {len(prompt_ids)} tokens...", end="", flush=True)
        t0 = time.time()
        for i, tid in enumerate(prompt_ids):
            logits = self.forward_step(tid, i)
            elapsed = time.time() - t0
            print(
                f"\r[gen] Prefill {i + 1}/{len(prompt_ids)} "
                f"({(i + 1) / elapsed:.2f} tok/s)",
                end="",
                flush=True,
            )
        prefill_time = time.time() - t0
        print(f" {prefill_time:.1f}s ({len(prompt_ids)/prefill_time:.1f} tok/s)")

        print(f"[gen] Generating...", flush=True)
        t0 = time.time()
        for step in range(max_tokens):
            pos = len(prompt_ids) + step

            if temperature <= 0:
                next_token = logits[0].argmax().item()
            else:
                probs = F.softmax(logits[0].float() / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

            generated.append(next_token)
            if next_token == eos:
                break

            elapsed = time.time() - t0
            tps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"\r[gen] step {step+1}/{max_tokens} ({tps:.1f} tok/s)", end="", flush=True)

            logits = self.forward_step(next_token, pos)

        elapsed = time.time() - t0
        print(f"\n[gen] {len(generated)} tokens in {elapsed:.1f}s ({len(generated)/elapsed:.1f} tok/s)")
        return generated
