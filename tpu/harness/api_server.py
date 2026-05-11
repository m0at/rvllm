#!/usr/bin/env python3
"""OpenAI-compatible API server for rvLLM Gemma 4 TPU inference.

Single-worker, single-request-at-a-time server. Loads model once at startup,
serves /v1/chat/completions with streaming and non-streaming support.

Usage:
    python3 api_server.py --model-dir ~/models/gemma-4-31B-it --port 8080
"""
import argparse, json, os, sys, time, uuid, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Sentinel returned from `_read_body` when a parse/limit error has
# already been written to the wire. Distinct from `None` (which means
# "empty body, that's allowed for a GET-shaped POST").
_BODY_REJECTED = object()

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_tpu_infer import (
    load_model, load_tokenizer, make_mesh, precompute_rope,
    forward_step,
    H, S_HD, NL, MAX_KV, MAX_KVH, SOFTCAP_VAL, B,
    _sharded_zeros, N_SLIDING, N_GLOBAL, WINDOW, S_KV, G_KV, S_KVH, G_KVH,
)

# ── globals set at startup ──
MODEL_NAME = "gemma-4-31B-it"
MESH = None
EMBED = None
FINAL_NORM = None
SL_WEIGHTS = None
GL_WEIGHTS = None
COS_S = COS_G = SIN_S = SIN_G = None
TOKENIZER = None
MAX_CTX = 2048
GEN_LOCK = threading.Lock()

BOS_TOKEN = 2
EOS_TOKENS = {1, 2}
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
    """Allocate split KV caches: sliding (WINDOW) + global (MAX_CTX)."""
    kv_sh = NamedSharding(MESH, P(None, None, 'tp'))
    kvs_sh = NamedSharding(MESH, P(None, None, None))
    sl = {
        'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
    }
    gl = {
        'kc': _sharded_zeros((N_GLOBAL, MAX_CTX, G_KV), np.int8, kv_sh),
        'vc': _sharded_zeros((N_GLOBAL, MAX_CTX, G_KV), np.int8, kv_sh),
        'kc_s': _sharded_zeros((N_GLOBAL, MAX_CTX, G_KVH), ml_dtypes.bfloat16, kvs_sh),
        'vc_s': _sharded_zeros((N_GLOBAL, MAX_CTX, G_KVH), ml_dtypes.bfloat16, kvs_sh),
    }
    return sl, gl


def sample_token(logits, temperature):
    """Greedy (temperature=0) or multinomial sampling."""
    if temperature <= 0:
        return int(jnp.argmax(logits, axis=-1)[0])
    scaled = logits[0] / temperature
    probs = jax.nn.softmax(scaled, axis=-1)
    return int(jax.random.categorical(
        jax.random.PRNGKey(int(time.time() * 1e6) & 0xFFFFFFFF), jnp.log(probs)))


class GenStatus:
    """Mutable container the caller passes into `generate()` so it
    can read back WHY generation ended after the iterator is
    exhausted. Without this, a caller has no way to distinguish
    "stop sequence hit" (OpenAI `finish_reason="stop"`) from
    "ran out of `max_tokens`" (`finish_reason="length"`) — the
    earlier code defaulted to "length" in both cases."""
    __slots__ = ("finish_reason",)
    def __init__(self):
        # Default; overwritten on EOS / stop-sequence paths.
        self.finish_reason = "length"


def generate(prompt_ids, max_tokens, temperature, stop_sequences, status=None):
    """Token-by-token generation. Yields (token_id, token_text) pairs.

    NOTE — current limitations of this experimental TPU API harness:
    * `temperature > 0` is rejected at the HTTP layer because
      `forward_step` argmaxes inside the JIT and discards logits.
      A future logits-out variant + host-side `sample_token` would
      lift this; the dead `sample_token` helper above is kept as a
      reference implementation.
    * Prompt prefill is token-at-a-time (one JIT call per prompt
      token). This is a known TTFT bottleneck for long prompts;
      a batched/scan'd prefill is the right fix and is out of
      scope for this round.
    """
    fwd_jit = jax.jit(forward_step)
    sl_caches, gl_caches = make_fresh_caches()
    num_prompt = len(prompt_ids)

    # Prefill: feed prompt tokens one at a time. See docstring.
    for step in range(num_prompt):
        tok_arr = jnp.array([prompt_ids[step]], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        next_tok, sl_caches, gl_caches = fwd_jit(
            tok_arr, pos, ctx, EMBED, FINAL_NORM,
            SL_WEIGHTS, GL_WEIGHTS, sl_caches, gl_caches,
            COS_S, SIN_S, COS_G, SIN_G)

    sampled = int(next_tok[0])
    accumulated = ""
    generated_count = 0

    for decode_step in range(max_tokens):
        token_text = TOKENIZER.decode([sampled])
        # Stop-check BEFORE yielding so the stop sequence does not
        # leak into the client's visible output. EOS is special-cased
        # to also stop, but EOS is allowed in output so we still emit
        # it before returning.
        if stop_sequences:
            tentative = accumulated + token_text
            if any(ss in tentative for ss in stop_sequences):
                if status is not None:
                    status.finish_reason = "stop"
                return
        accumulated += token_text
        generated_count += 1
        yield sampled, token_text

        if sampled in EOS_TOKENS:
            if status is not None:
                status.finish_reason = "stop"
            return

        if num_prompt + decode_step + 1 >= MAX_CTX - 1:
            # Ran out of context window before max_tokens — count as
            # "length" per OpenAI semantics (truncated by capacity).
            return

        step = num_prompt + decode_step
        tok_arr = jnp.array([sampled], dtype=jnp.int32)
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        next_tok, sl_caches, gl_caches = fwd_jit(
            tok_arr, pos, ctx, EMBED, FINAL_NORM,
            SL_WEIGHTS, GL_WEIGHTS, sl_caches, gl_caches,
            COS_S, SIN_S, COS_G, SIN_G)

        sampled = int(next_tok[0])


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

    # Hard cap on request body size before we buffer it. Without this
    # a bogus or hostile `Content-Length: 1_000_000_000` would have
    # the single-process server read a gigabyte into RAM. 16 MiB is
    # generous for a tokens+strings JSON; override via env if you
    # actually need vision / very long pasted prompts on the TPU
    # harness (not the production rvllm-serve path).
    _MAX_BODY_BYTES = int(os.environ.get("RVLLM_TPU_MAX_BODY_BYTES",
                                          str(16 * 1024 * 1024)))

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_error(400, "invalid Content-Length header")
            return _BODY_REJECTED
        if length < 0:
            self._send_error(400, "negative Content-Length")
            return _BODY_REJECTED
        if length == 0:
            return None
        if length > self._MAX_BODY_BYTES:
            self._send_error(413, f"request body exceeds {self._MAX_BODY_BYTES} bytes")
            return _BODY_REJECTED
        try:
            raw = self.rfile.read(length)
        except Exception as e:
            self._send_error(400, f"body read failed: {e}")
            return _BODY_REJECTED
        if len(raw) != length:
            self._send_error(400, "body shorter than Content-Length")
            return _BODY_REJECTED
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._send_error(400, f"invalid JSON: {e}")
            return _BODY_REJECTED

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
        if body is _BODY_REJECTED:
            # `_read_body` already wrote the error response.
            return
        if body is None:
            self._send_error(400, "empty request body")
            return

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_error(400, "'messages' is required and must be a list")
            return

        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.0)
        # `forward_step` argmaxes inside the JIT and discards logits,
        # so non-greedy sampling cannot be honoured here. Reject
        # explicitly so callers cannot believe their `temperature`
        # was applied. Lift when a logits-out forward_step variant
        # lands and `sample_token` is wired into the generate loop.
        if temperature is not None and temperature > 0:
            self._send_error(
                400,
                "temperature > 0 not supported on the TPU API harness "
                "(forward_step is greedy). Set temperature=0 or omit it.",
            )
            return
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

        # GenStatus catches stop-sequence completion (which can't be
        # observed from the yielded token stream alone — generate()
        # exits via `return` before yielding the stop tokens).
        status = GenStatus()
        completion_tokens = 0
        for token_id, token_text in generate(prompt_ids, max_tokens, temperature, stop, status):
            if token_id in EOS_TOKENS:
                # generate() also sets status.finish_reason="stop"
                # for EOS, so we don't override here.
                break
            completion_tokens += 1
            chunk = make_chunk(request_id, model, delta_content=token_text)
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Final chunk
        chunk = make_chunk(request_id, model, finish_reason=status.finish_reason)
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_sync(self, request_id, model, prompt_ids, num_prompt,
                     max_tokens, temperature, stop):
        content_parts = []
        status = GenStatus()
        completion_tokens = 0

        for token_id, token_text in generate(prompt_ids, max_tokens, temperature, stop, status):
            if token_id in EOS_TOKENS:
                break
            content_parts.append(token_text)
            completion_tokens += 1

        content = "".join(content_parts)
        resp = make_response(request_id, model, content, status.finish_reason,
                             num_prompt, completion_tokens)
        self._send_json(200, resp)


def main():
    global MESH, EMBED, FINAL_NORM, SL_WEIGHTS, GL_WEIGHTS, COS_S, SIN_S, COS_G, SIN_G
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
    EMBED, FINAL_NORM, SL_WEIGHTS, GL_WEIGHTS, _, _ = load_model(args.model_dir, MESH, MAX_CTX)

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
    sl_c, gl_c = make_fresh_caches()
    fwd_jit = jax.jit(forward_step)
    tok = jnp.array([BOS_TOKEN], dtype=jnp.int32)
    _, sl_c, gl_c = fwd_jit(tok, jnp.int32(0), jnp.int32(1),
                             EMBED, FINAL_NORM, SL_WEIGHTS, GL_WEIGHTS, sl_c, gl_c,
                             COS_S, SIN_S, COS_G, SIN_G)
    jax.effects_barrier()
    print(f"JIT warm-up: {time.time() - t0:.1f}s", file=sys.stderr)
    del sl_c, gl_c

    # Start server
    # ThreadingHTTPServer instead of HTTPServer: a single in-flight
    # generate() must NOT serialise concurrent /health probes or
    # follow-up requests. GEN_LOCK still guarantees only one
    # forward-pass at a time and returns 503 to extras; with the
    # plain HTTPServer that 503 path was unreachable because the
    # second request never even got dispatched.
    server = ThreadingHTTPServer((args.host, args.port), APIHandler)
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
