#!/usr/bin/env python3
"""OpenAI-compatible API server for rvLLM Gemma 4 TPU inference.

Single-worker, single-request-at-a-time server. Loads model once at startup,
serves /v1/chat/completions with streaming and non-streaming support.

Usage:
    python3 api_server.py --model-dir ~/models/gemma-4-31B-it --port 8080
"""
import argparse, json, os, sys, time, uuid, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_tpu_infer import (
    load_model, load_tokenizer, make_mesh, precompute_rope,
    forward, forward_step,
    H, S_HD, NL, MAX_KV, MAX_KVH, SOFTCAP_VAL, B,
)

# ── globals set at startup ──
MODEL_NAME = "gemma-4-31B-it"
MESH = None
EMBED = None
FINAL_NORM = None
WEIGHTS = None
COS_S = COS_G = SIN_S = SIN_G = None
TOKENIZER = None
MAX_CTX = 2048
GEN_LOCK = threading.Lock()

BOS_TOKEN = 2
EOS_TOKENS = {1, 2, 107}
TURN_TOKENS = None  # populated after tokenizer loads


def format_chat_prompt(messages):
    """Apply Gemma 4 chat template to a list of messages."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            # Gemma 4 has no system role; prepend as user context
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
    # Open model turn for generation
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def tokenize_prompt(text):
    """Tokenize with BOS prefix."""
    ids = TOKENIZER.encode(text).ids
    return [BOS_TOKEN] + ids


def make_fresh_caches():
    """Allocate zeroed KV caches on device."""
    kv_sh = NamedSharding(MESH, P(None, None, 'tp'))
    kvs_sh = NamedSharding(MESH, P(None, None, None))
    return {
        'kc': jax.device_put(jnp.zeros((NL, MAX_CTX, MAX_KV), dtype=jnp.int8), kv_sh),
        'vc': jax.device_put(jnp.zeros((NL, MAX_CTX, MAX_KV), dtype=jnp.int8), kv_sh),
        'kc_s': jax.device_put(jnp.zeros((NL, MAX_CTX, MAX_KVH), dtype=jnp.bfloat16), kvs_sh),
        'vc_s': jax.device_put(jnp.zeros((NL, MAX_CTX, MAX_KVH), dtype=jnp.bfloat16), kvs_sh),
    }


def sample_token(logits, temperature):
    """Greedy (temperature=0) or multinomial sampling."""
    if temperature <= 0:
        return int(jnp.argmax(logits, axis=-1)[0])
    scaled = logits[0] / temperature
    probs = jax.nn.softmax(scaled, axis=-1)
    return int(jax.random.categorical(
        jax.random.PRNGKey(int(time.time() * 1e6) & 0xFFFFFFFF), jnp.log(probs)))


def generate(prompt_ids, max_tokens, temperature, stop_sequences):
    """Token-by-token generation. Yields (token_id, token_text) pairs."""
    fwd_jit = jax.jit(forward, donate_argnums=(6,))
    caches = make_fresh_caches()
    num_prompt = len(prompt_ids)

    # Prefill: feed prompt tokens one at a time
    for step in range(num_prompt):
        tok_arr = jnp.array([prompt_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        if step == num_prompt - 1:
            next_tok, log_probs, caches = fwd_jit(
                tok_arr, pos, ctx, EMBED, FINAL_NORM, WEIGHTS, caches,
                COS_S, SIN_S, COS_G, SIN_G)
        else:
            _, _, caches = fwd_jit(
                tok_arr, pos, ctx, EMBED, FINAL_NORM, WEIGHTS, caches,
                COS_S, SIN_S, COS_G, SIN_G)

    # First decode token from last prefill logits
    if temperature <= 0:
        sampled = int(jnp.argmax(log_probs, axis=-1)[0])  # greedy from full logits dim
    else:
        sampled = sample_token(log_probs, temperature)

    # Actually use the argmax already computed by forward()
    sampled = int(next_tok[0])

    accumulated = ""
    generated_count = 0

    for decode_step in range(max_tokens):
        token_text = TOKENIZER.decode([sampled])
        accumulated += token_text
        generated_count += 1
        yield sampled, token_text

        if sampled in EOS_TOKENS:
            return

        # Check stop sequences
        if stop_sequences:
            for ss in stop_sequences:
                if ss in accumulated:
                    return

        if num_prompt + decode_step + 1 >= MAX_CTX - 1:
            return

        # Next forward step
        step = num_prompt + decode_step
        tok_arr = jnp.array([sampled], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        next_tok, log_probs, caches = fwd_jit(
            tok_arr, pos, ctx, EMBED, FINAL_NORM, WEIGHTS, caches,
            COS_S, SIN_S, COS_G, SIN_G)

        if temperature <= 0:
            sampled = int(next_tok[0])
        else:
            sampled = sample_token(log_probs, temperature)


def make_chunk(request_id, model, delta_content=None, finish_reason=None):
    """Build one SSE chunk in OpenAI streaming format."""
    choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if delta_content is not None:
        choice["delta"]["content"] = delta_content
    if finish_reason is None and delta_content is None:
        choice["delta"]["role"] = "assistant"
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }


def make_response(request_id, model, content, finish_reason, prompt_tokens, completion_tokens):
    """Build a non-streaming OpenAI response."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class APIHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests."""

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", file=sys.stderr)

    def _cors_headers(self):
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        })

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rvllm",
                }],
            })
        else:
            self._send_error(404, "not found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_error(404, "not found")
            return

        body = self._read_body()
        if body is None:
            self._send_error(400, "empty request body")
            return

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_error(400, "'messages' is required and must be a list")
            return

        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.0)
        stream = body.get("stream", False)
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        model = body.get("model", MODEL_NAME)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Build prompt
        prompt_text = format_chat_prompt(messages)
        prompt_ids = tokenize_prompt(prompt_text)
        num_prompt = len(prompt_ids)

        if num_prompt >= MAX_CTX - 1:
            self._send_error(400, f"prompt too long: {num_prompt} tokens >= max_ctx {MAX_CTX}")
            return

        # Clamp max_tokens to remaining context
        max_tokens = min(max_tokens, MAX_CTX - num_prompt - 1)

        acquired = GEN_LOCK.acquire(timeout=0.1)
        if not acquired:
            self._send_error(503, "server busy, try again")
            return

        try:
            if stream:
                self._handle_stream(request_id, model, prompt_ids, num_prompt,
                                    max_tokens, temperature, stop)
            else:
                self._handle_sync(request_id, model, prompt_ids, num_prompt,
                                  max_tokens, temperature, stop)
        except Exception as e:
            self._send_error(500, str(e))
        finally:
            GEN_LOCK.release()

    def _handle_stream(self, request_id, model, prompt_ids, num_prompt,
                       max_tokens, temperature, stop):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

        # Initial chunk with role
        chunk = make_chunk(request_id, model)
        chunk["choices"][0]["delta"]["role"] = "assistant"
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

        finish_reason = "length"
        completion_tokens = 0
        for token_id, token_text in generate(prompt_ids, max_tokens, temperature, stop):
            if token_id in EOS_TOKENS:
                finish_reason = "stop"
                break
            completion_tokens += 1
            chunk = make_chunk(request_id, model, delta_content=token_text)
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Final chunk
        chunk = make_chunk(request_id, model, finish_reason=finish_reason)
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_sync(self, request_id, model, prompt_ids, num_prompt,
                     max_tokens, temperature, stop):
        content_parts = []
        finish_reason = "length"
        completion_tokens = 0

        for token_id, token_text in generate(prompt_ids, max_tokens, temperature, stop):
            if token_id in EOS_TOKENS:
                finish_reason = "stop"
                break
            content_parts.append(token_text)
            completion_tokens += 1

        content = "".join(content_parts)
        resp = make_response(request_id, model, content, finish_reason,
                             num_prompt, completion_tokens)
        self._send_json(200, resp)


def main():
    global MESH, EMBED, FINAL_NORM, WEIGHTS, COS_S, SIN_S, COS_G, SIN_G
    global TOKENIZER, MAX_CTX, MODEL_NAME

    parser = argparse.ArgumentParser(description="rvLLM OpenAI-compatible API server")
    parser.add_argument("--model-dir", required=True, help="Path to Gemma 4 model directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-ctx", type=int, default=2048)
    parser.add_argument("--model-name", default=None, help="Model name for API responses")
    args = parser.parse_args()

    MAX_CTX = args.max_ctx
    if args.model_name:
        MODEL_NAME = args.model_name
    else:
        MODEL_NAME = os.path.basename(args.model_dir.rstrip("/"))

    # Load tokenizer
    print(f"loading tokenizer from {args.model_dir}", file=sys.stderr)
    TOKENIZER = load_tokenizer(args.model_dir)
    if TOKENIZER is None:
        print("FATAL: no tokenizer.json found", file=sys.stderr)
        sys.exit(1)

    # Load model
    MESH = make_mesh()
    print(f"mesh: {MESH}", file=sys.stderr)
    EMBED, FINAL_NORM, WEIGHTS, _ = load_model(args.model_dir, MESH, MAX_CTX)

    # Precompute RoPE
    cos_s, sin_s = precompute_rope(10000.0, S_HD, MAX_CTX)
    cos_g, sin_g = precompute_rope(1000000.0, 128, MAX_CTX)
    COS_S = jax.device_put(jnp.array(cos_s), NamedSharding(MESH, P(None, None)))
    SIN_S = jax.device_put(jnp.array(sin_s), NamedSharding(MESH, P(None, None)))
    COS_G = jax.device_put(jnp.array(cos_g), NamedSharding(MESH, P(None, None)))
    SIN_G = jax.device_put(jnp.array(sin_g), NamedSharding(MESH, P(None, None)))

    # Warm up JIT with a dummy forward pass
    print("warming up JIT...", file=sys.stderr, flush=True)
    t0 = time.time()
    dummy_caches = make_fresh_caches()
    fwd_jit = jax.jit(forward, donate_argnums=(6,))
    tok = jnp.array([BOS_TOKEN], dtype=jnp.int32)
    _, _, _ = fwd_jit(tok, jnp.int32(0), jnp.int32(1),
                      EMBED, FINAL_NORM, WEIGHTS, dummy_caches,
                      COS_S, SIN_S, COS_G, SIN_G)
    jax.effects_barrier()
    print(f"JIT warm-up: {time.time() - t0:.1f}s", file=sys.stderr)
    del dummy_caches

    # Start server
    server = HTTPServer((args.host, args.port), APIHandler)
    print(f"serving on {args.host}:{args.port}", file=sys.stderr)
    print(f"  model:   {MODEL_NAME}", file=sys.stderr)
    print(f"  max_ctx: {MAX_CTX}", file=sys.stderr)
    print(f"  POST /v1/chat/completions", file=sys.stderr)
    print(f"  GET  /v1/models", file=sys.stderr)
    print(f"  GET  /health", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
