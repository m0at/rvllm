#!/usr/bin/env python3
"""HF spot-check for Gemma4 DecomposeK on sliding-layer o_proj only.

This is a safety harness, not a runtime integration. It patches only the
HuggingFace model's sliding-layer attention o_proj modules, then compares:

- prompt loss / spot-check perplexity
- final-token logit drift on a fixed prompt
- greedy decode token parity over a short continuation

Use this to decide whether the DecomposeK o_sliding policy is numerically
safe enough to promote, without touching the rvLLM 31B runtime path.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(torch.nn, "Buffer"):
    torch.nn.Buffer = torch.Tensor


def patch_compressed_tensors_buffer_compat() -> None:
    try:
        import compressed_tensors.compressors.base as ct_base
        import compressed_tensors.utils.module as ct_module
    except Exception:
        return

    def get_direct_state_dict(module: torch.nn.Module):
        return {
            name: (
                tensor.data
                if isinstance(tensor, (torch.nn.Parameter, torch.Tensor))
                else tensor
            )
            for name, tensor in chain(module._parameters.items(), module._buffers.items())
        }

    ct_module.get_direct_state_dict = get_direct_state_dict
    ct_base.get_direct_state_dict = get_direct_state_dict


DEFAULT_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a quiet village, the baker lit the ovens before dawn."
)

DEFAULT_EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a quiet village, the baker lit the ovens before dawn. "
    "A cold wind moved through the empty square while the first carts rolled in. "
    "By sunrise, the streets were full of people carrying milk, bread, and letters. "
    "Nothing seemed unusual, but every small sound was clear in the winter air."
)


@dataclass
class LossSummary:
    tokens: int
    loss: float
    perplexity: float


@dataclass
class LogitSummary:
    max_abs_diff: float
    max_rel_diff: float
    l2_rel_diff: float
    topk_match: bool
    baseline_topk_ids: list[int]
    patched_topk_ids: list[int]


@dataclass
class DecodeSummary:
    prompt_tokens: int
    max_new_tokens: int
    baseline_new_ids: list[int]
    patched_new_ids: list[int]
    exact_match: bool
    baseline_text: str
    patched_text: str
    baseline_ms: float
    patched_ms: float


@dataclass
class ModuleParitySummary:
    layer: int
    max_abs_diff: float
    l2_rel_diff: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/workspace/models/gemma4-31b-fp8",
        help="Local model dir on the H100 box.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="HF attention backend to request when loading the model.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument(
        "--variant",
        default="hf_quant_fp32",
        choices=["native", "sum_fp32", "hf_quant_bf16", "hf_quant_fp32"],
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for logit and decode parity checks.",
    )
    parser.add_argument(
        "--eval-text",
        default=DEFAULT_EVAL_TEXT,
        help="Text used for the spot-check perplexity measurement.",
    )
    parser.add_argument(
        "--eval-text-file",
        default=None,
        help="Optional file to override --eval-text.",
    )
    parser.add_argument("--max-eval-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-sliding-layers", type=int, default=0)
    parser.add_argument("--module-parity-layers", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_text_model(model):
    text_model = getattr(model, "model", model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    return text_model


def load_eval_text(args: argparse.Namespace) -> str:
    if args.eval_text_file is None:
        return args.eval_text
    return Path(args.eval_text_file).read_text()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_logits(
    baseline_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    topk: int,
) -> LogitSummary:
    baseline = baseline_logits.float().cpu()
    patched = patched_logits.float().cpu()
    diff = (patched - baseline).abs()
    denom = baseline.abs().clamp_min(1e-2)
    baseline_topk_ids = torch.topk(baseline, k=topk).indices.tolist()
    patched_topk_ids = torch.topk(patched, k=topk).indices.tolist()
    return LogitSummary(
        max_abs_diff=float(diff.max().item()),
        max_rel_diff=float((diff / denom).max().item()),
        l2_rel_diff=float(diff.norm().item() / baseline.norm().clamp_min(1e-6).item()),
        topk_match=baseline_topk_ids == patched_topk_ids,
        baseline_topk_ids=baseline_topk_ids,
        patched_topk_ids=patched_topk_ids,
    )


def compute_prompt_loss(
    model,
    tokenizer,
    text: str,
    max_eval_tokens: int,
) -> LossSummary:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][:, :max_eval_tokens].to(next(model.parameters()).device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = float(outputs.loss.float().item())
    return LossSummary(
        tokens=int(input_ids.shape[1]),
        loss=loss,
        perplexity=float(math.exp(loss)),
    )


def compute_last_token_logits(model, tokenizer, prompt: str) -> tuple[int, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
    return int(input_ids.shape[1]), outputs.logits[0, -1].detach()


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> DecodeSummary:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    with torch.inference_mode():
        t0 = perf_counter()
        output_ids = model.generate(**generation_kwargs)
        synchronize()
        elapsed_ms = (perf_counter() - t0) * 1000.0
    new_ids = output_ids[0, input_ids.shape[1]:].tolist()
    return DecodeSummary(
        prompt_tokens=int(input_ids.shape[1]),
        max_new_tokens=max_new_tokens,
        baseline_new_ids=new_ids,
        patched_new_ids=[],
        exact_match=False,
        baseline_text=tokenizer.decode(new_ids, skip_special_tokens=True),
        patched_text="",
        baseline_ms=elapsed_ms,
        patched_ms=0.0,
    )


def decomposek_linear(
    hidden_states: torch.Tensor,
    module: torch.nn.Linear,
    splits: int,
    variant: str,
) -> torch.Tensor:
    if variant.startswith("hf_quant_"):
        from compressed_tensors.quantization.lifecycle.forward import forward_quantize

        scheme = getattr(module, "quantization_scheme", None)
        status = getattr(module, "quantization_status", None)
        enabled = (
            getattr(module, "quantization_enabled", True)
            and scheme is not None
            and status is not None
            and scheme.input_activations is not None
        )
        if enabled:
            hidden_states = forward_quantize(
                module, hidden_states, "input", scheme.input_activations
            )

    orig_dtype = hidden_states.dtype
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    weight_out_k = module.weight
    if variant in ("hf_quant_fp32",):
        weight_out_k = weight_out_k.float()
        flat = flat.float()
    k = flat.shape[-1]
    if k % splits != 0:
        raise ValueError(f"k={k} is not divisible by splits={splits}")
    k_per = k // splits
    x_bmm = flat.unflatten(1, (splits, k_per)).transpose(0, 1).contiguous()
    weight_k_n = weight_out_k.transpose(0, 1).contiguous()
    w_bmm = weight_k_n.unflatten(0, (splits, k_per)).contiguous()
    partials = torch.bmm(x_bmm, w_bmm)
    if variant == "sum_fp32":
        reduced = partials.float().sum(dim=0)
    elif variant == "hf_quant_fp32":
        reduced = partials.sum(dim=0)
    elif variant in ("native", "hf_quant_bf16"):
        reduced = partials.sum(dim=0)
    else:
        raise ValueError(f"unknown variant: {variant}")
    if module.bias is not None:
        reduced = reduced + module.bias.float()
    return reduced.reshape(*hidden_states.shape[:-1], weight_out_k.shape[0]).to(orig_dtype)


def summarize_module_parity(
    layer: int,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> ModuleParitySummary:
    diff = (actual - expected).abs().float()
    return ModuleParitySummary(
        layer=layer,
        max_abs_diff=float(diff.max().item()),
        l2_rel_diff=float(diff.norm().item() / expected.float().norm().clamp_min(1e-6).item()),
    )


def collect_o_proj_parity(
    model,
    tokenizer,
    prompt: str,
    splits: int,
    variant: str,
    max_layers: int,
) -> list[ModuleParitySummary]:
    if max_layers <= 0:
        return []
    text_model = resolve_text_model(model)
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    seen = 0
    for idx, layer in enumerate(text_model.layers):
        o_proj = layer.self_attn.o_proj
        if int(o_proj.weight.shape[1]) != 8192:
            continue
        if seen >= max_layers:
            break
        seen += 1

        def hook(module, inputs, output, _idx=idx):
            captures[_idx] = (inputs[0].detach(), output.detach())

        handles.append(o_proj.register_forward_hook(hook))
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()

    summaries: list[ModuleParitySummary] = []
    with torch.inference_mode():
        for idx in sorted(captures):
            o_proj = text_model.layers[idx].self_attn.o_proj
            hidden_states, expected = captures[idx]
            actual = decomposek_linear(
                hidden_states=hidden_states,
                module=o_proj,
                splits=splits,
                variant=variant,
            )
            summaries.append(summarize_module_parity(idx, expected, actual))
    return summaries


def patch_sliding_o_proj(
    model,
    splits: int,
    variant: str,
    max_sliding_layers: int,
) -> list[int]:
    text_model = resolve_text_model(model)
    layers = text_model.layers
    patched_indices: list[int] = []
    limit = max_sliding_layers if max_sliding_layers > 0 else len(layers)
    for idx, layer in enumerate(layers):
        o_proj = layer.self_attn.o_proj
        in_features = int(o_proj.weight.shape[1])
        if in_features != 8192:
            continue
        if len(patched_indices) >= limit:
            break
        original_forward = o_proj.forward

        def forward(self, hidden_states, _original_forward=original_forward):
            return decomposek_linear(
                hidden_states=hidden_states,
                module=self,
                splits=splits,
                variant=variant,
            )

        o_proj.forward = types.MethodType(forward, o_proj)
        patched_indices.append(idx)
    return patched_indices


def run_decode_pair(model, tokenizer, prompt: str, max_new_tokens: int) -> DecodeSummary:
    baseline = greedy_decode(model, tokenizer, prompt, max_new_tokens)
    patched = greedy_decode(model, tokenizer, prompt, max_new_tokens)
    baseline_ids = baseline.baseline_new_ids
    patched_ids = patched.baseline_new_ids
    return DecodeSummary(
        prompt_tokens=baseline.prompt_tokens,
        max_new_tokens=max_new_tokens,
        baseline_new_ids=baseline_ids,
        patched_new_ids=patched_ids,
        exact_match=baseline_ids == patched_ids,
        baseline_text=baseline.baseline_text,
        patched_text=patched.baseline_text,
        baseline_ms=baseline.baseline_ms,
        patched_ms=patched.baseline_ms,
    )


def main() -> None:
    args = parse_args()
    eval_text = load_eval_text(args)
    dtype = resolve_dtype(args.torch_dtype)
    patch_compressed_tensors_buffer_compat()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    module_parity = collect_o_proj_parity(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        splits=args.splits,
        variant=args.variant,
        max_layers=args.module_parity_layers,
    )
    prompt_tokens, baseline_logits = compute_last_token_logits(model, tokenizer, args.prompt)
    baseline_loss = compute_prompt_loss(model, tokenizer, eval_text, args.max_eval_tokens)
    baseline_decode = greedy_decode(model, tokenizer, args.prompt, args.max_new_tokens)

    patched_layers = patch_sliding_o_proj(
        model=model,
        splits=args.splits,
        variant=args.variant,
        max_sliding_layers=args.max_sliding_layers,
    )
    if not patched_layers:
        raise RuntimeError("did not find any sliding-layer o_proj modules to patch")

    _, patched_logits = compute_last_token_logits(model, tokenizer, args.prompt)
    patched_loss = compute_prompt_loss(model, tokenizer, eval_text, args.max_eval_tokens)
    patched_decode = greedy_decode(model, tokenizer, args.prompt, args.max_new_tokens)

    logits = summarize_logits(
        baseline_logits=baseline_logits,
        patched_logits=patched_logits,
        topk=args.topk,
    )
    decode = DecodeSummary(
        prompt_tokens=baseline_decode.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        baseline_new_ids=baseline_decode.baseline_new_ids,
        patched_new_ids=patched_decode.baseline_new_ids,
        exact_match=baseline_decode.baseline_new_ids == patched_decode.baseline_new_ids,
        baseline_text=baseline_decode.baseline_text,
        patched_text=patched_decode.baseline_text,
        baseline_ms=baseline_decode.baseline_ms,
        patched_ms=patched_decode.baseline_ms,
    )

    result = {
        "model_path": args.model_path,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "variant": args.variant,
        "splits": args.splits,
        "patched_layer_count": len(patched_layers),
        "patched_layers": patched_layers,
        "module_parity": [asdict(item) for item in module_parity],
        "prompt_tokens": prompt_tokens,
        "baseline_loss": asdict(baseline_loss),
        "patched_loss": asdict(patched_loss),
        "ppl_ratio": patched_loss.perplexity / baseline_loss.perplexity,
        "logits": asdict(logits),
        "decode": asdict(decode),
    }

    print(json.dumps(result, indent=2))
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
