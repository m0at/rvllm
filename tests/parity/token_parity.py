#!/usr/bin/env python3
"""Token-level parity test: rvLLM vs Python vLLM.

Sends identical prompts to both servers with temperature=0 (greedy decoding)
and compares output token-by-token. Any divergence indicates a numerical
difference in the forward pass.

Usage:
    # Both servers must be running
    python3 tests/parity/token_parity.py \
        --rust-url http://localhost:8000 \
        --python-url http://localhost:8001 \
        --model Qwen/Qwen2.5-1.5B
"""
import argparse, json, sys, requests

PROMPTS = [
    # Factual
    "The capital of France is",
    "The speed of light in a vacuum is approximately",
    "Water freezes at",
    "The largest planet in our solar system is",
    # Code
    "def fibonacci(n):",
    "fn main() {\n    println!(",
    "SELECT * FROM users WHERE",
    "import numpy as np\ndef matrix_multiply(a, b):",
    # Math
    "The derivative of x^3 is",
    "Solve for x: 2x + 5 = 13. x =",
    "The integral of sin(x) dx is",
    # Creative / open-ended
    "Once upon a time in a land far away,",
    "The most important invention of the 20th century was",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean:",
    # Multilingual
    "Translate to French: Hello, how are you?",
    "La capital de Espana es",
    "Der schnelle braune Fuchs",
    # Long context / structured
    "List the first 10 prime numbers: 2, 3, 5, 7,",
    "JSON example: {\"name\": \"Alice\", \"age\":",
    # Edge cases
    "1 + 1 = 2\n2 + 2 = 4\n3 + 3 =",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. Now reversed: Z Y X W V U T S R Q P O N",
]

MAX_TOKENS = 50


def query_completions(url, model, prompt, logprobs=None):
    """Send a completion request and return parsed response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def extract_tokens(response):
    """Extract token list from response logprobs if available, else split text."""
    choice = response["choices"][0]
    if choice.get("logprobs") and choice["logprobs"].get("tokens"):
        return choice["logprobs"]["tokens"]
    return list(choice["text"])


def extract_text(response):
    return response["choices"][0]["text"]


def extract_finish_reason(response):
    return response["choices"][0]["finish_reason"]


def extract_top_logprobs(response):
    """Extract top logprobs list-of-dicts if available."""
    choice = response["choices"][0]
    if choice.get("logprobs") and choice["logprobs"].get("top_logprobs"):
        return choice["logprobs"]["top_logprobs"]
    return None


def compare_tokens(rust_tokens, python_tokens):
    """Find first divergence point. Returns (match_count, total, diverge_idx or None)."""
    total = max(len(rust_tokens), len(python_tokens))
    for i in range(min(len(rust_tokens), len(python_tokens))):
        if rust_tokens[i] != python_tokens[i]:
            return i, total, i
    if len(rust_tokens) != len(python_tokens):
        return min(len(rust_tokens), len(python_tokens)), total, min(len(rust_tokens), len(python_tokens))
    return total, total, None


def format_logprob_comparison(rust_top, python_top, idx):
    """Format logprob comparison at a specific token index."""
    lines = []
    if rust_top and idx < len(rust_top) and rust_top[idx]:
        sorted_r = sorted(rust_top[idx].items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"    Rust top-5:   {sorted_r}")
    if python_top and idx < len(python_top) and python_top[idx]:
        sorted_p = sorted(python_top[idx].items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"    Python top-5: {sorted_p}")
    return "\n".join(lines)


def truncate(s, n=60):
    return s[:n] + "..." if len(s) > n else s


def run_parity(rust_url, python_url, model, use_logprobs):
    logprobs_param = 5 if use_logprobs else None
    exact = 0
    diverged = 0
    divergence_points = []
    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        label = truncate(prompt.replace("\n", "\\n"), 50)
        print(f"\nPrompt {i}: \"{label}\"")

        try:
            rust_resp = query_completions(rust_url, model, prompt, logprobs_param)
            python_resp = query_completions(python_url, model, prompt, logprobs_param)
        except requests.RequestException as e:
            print(f"  ERROR: {e}")
            results.append({"prompt": prompt, "status": "error", "error": str(e)})
            continue

        rust_text = extract_text(rust_resp)
        python_text = extract_text(python_resp)
        rust_finish = extract_finish_reason(rust_resp)
        python_finish = extract_finish_reason(python_resp)

        # Token-level comparison
        rust_tokens = extract_tokens(rust_resp)
        python_tokens = extract_tokens(python_resp)
        match_count, total, diverge_idx = compare_tokens(rust_tokens, python_tokens)

        print(f"  Rust:   \"{truncate(rust_text)}\"")
        print(f"  Python: \"{truncate(python_text)}\"")

        if diverge_idx is None and rust_finish == python_finish:
            print(f"  Match: EXACT ({match_count}/{total} tokens)")
            exact += 1
            results.append({"prompt": prompt, "status": "exact", "tokens": total})
        else:
            diverged += 1
            divergence_points.append(diverge_idx if diverge_idx is not None else total)
            reason_parts = []
            if diverge_idx is not None:
                r_tok = rust_tokens[diverge_idx] if diverge_idx < len(rust_tokens) else "<EOF>"
                p_tok = python_tokens[diverge_idx] if diverge_idx < len(python_tokens) else "<EOF>"
                reason_parts.append(f"token {diverge_idx} (rust={repr(r_tok)} python={repr(p_tok)})")
            if rust_finish != python_finish:
                reason_parts.append(f"finish_reason (rust={rust_finish} python={python_finish})")
            print(f"  Match: DIVERGED at {', '.join(reason_parts)}")

            # Logprob comparison at divergence point
            if use_logprobs and diverge_idx is not None:
                rust_top = extract_top_logprobs(rust_resp)
                python_top = extract_top_logprobs(python_resp)
                lp_str = format_logprob_comparison(rust_top, python_top, diverge_idx)
                if lp_str:
                    print(f"  Logprob comparison at divergence:")
                    print(lp_str)

            results.append({
                "prompt": prompt, "status": "diverged",
                "diverge_idx": diverge_idx, "total": total,
            })

    # Summary
    total_prompts = len(PROMPTS)
    errors = sum(1 for r in results if r["status"] == "error")
    tested = total_prompts - errors
    print(f"\n{'='*60}")
    print(f"SUMMARY: {exact}/{tested} exact match, {diverged}/{tested} diverged", end="")
    if errors:
        print(f", {errors} errors", end="")
    print()

    if divergence_points:
        avg_div = sum(divergence_points) / len(divergence_points)
        print(f"  Average divergence point: token {avg_div:.1f} (out of {MAX_TOKENS})")
        print(f"  Root cause: likely f32 vs bf16 accumulation or kv-cache precision")

    if exact == tested:
        print("RESULT: PASS -- all outputs identical")
        return 0
    elif exact >= tested * 0.8:
        print(f"RESULT: WARN -- {exact}/{tested} match (>=80% threshold)")
        return 0
    else:
        print(f"RESULT: FAIL -- {exact}/{tested} match (<80% threshold)")
        return 1


def main():
    p = argparse.ArgumentParser(description="Token-level parity: rvLLM vs Python vLLM")
    p.add_argument("--rust-url", default="http://localhost:8000")
    p.add_argument("--python-url", default="http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--logprobs", action="store_true", help="Compare top-5 logprobs at divergence")
    args = p.parse_args()

    print(f"Token Parity Test: rvLLM vs Python vLLM")
    print(f"  Rust:   {args.rust_url}")
    print(f"  Python: {args.python_url}")
    print(f"  Model:  {args.model}")
    print(f"  Logprobs: {args.logprobs}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Temperature: 0 (greedy)")

    sys.exit(run_parity(args.rust_url, args.python_url, args.model, args.logprobs))


if __name__ == "__main__":
    main()
