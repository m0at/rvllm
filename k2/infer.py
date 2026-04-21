#!/usr/bin/env python3
"""Kimi K2.6 inference -- MLA on GPU, MoE experts on CPU.

Usage:
    python3 k2/infer.py --model-dir /path/to/Kimi-K2.6 --prompt "Hello"
    python3 k2/infer.py --model-dir /path/to/Kimi-K2.6 --max-tokens 200
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Kimi K2.6 inference (CPU-offload MoE)")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello! Can you tell me about yourself?")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch-cpu-threads", type=int, default=16)
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument("--expert-cache-size", type=int, default=4096)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    os.environ["OMP_NUM_THREADS"] = str(args.torch_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_cpu_threads)
    torch.set_num_threads(args.torch_cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(f"[init] Loading tokenizer...", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"[init] Loading model weights...", file=sys.stderr)
    from loader import K2Weights
    from model import K2Model

    t0 = time.time()
    weights = K2Weights(
        model_dir,
        device=args.device,
        expert_cache_size=args.expert_cache_size,
    )
    load_time = time.time() - t0
    print(f"[init] Weights loaded in {load_time:.1f}s", file=sys.stderr)

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[init] GPU memory: {gpu_mem:.1f} GB", file=sys.stderr)

    model = K2Model(
        weights,
        device=args.device,
        expert_workers=args.expert_workers,
    )

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[gen] Prompt: {args.prompt}", file=sys.stderr)
    print(f"[gen] Prompt tokens: {len(prompt_ids)}", file=sys.stderr)

    t0 = time.time()
    output_ids = model.generate(
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_time = time.time() - t0

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    tok_per_sec = len(output_ids) / gen_time if gen_time > 0 else 0

    print(f"\n--- Output ({len(output_ids)} tokens, {gen_time:.1f}s, {tok_per_sec:.1f} tok/s) ---")
    print(output_text)
    print(f"--- End ---")


if __name__ == "__main__":
    main()
