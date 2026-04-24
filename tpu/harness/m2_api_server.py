#!/usr/bin/env python3
"""OpenAI-compatible MiniMax-M2.7 TPU server with safe decode microbatching."""

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_tpu_infer as m2
from m2_attention import precompute_rope_m2
from m2_chat import apply_chat_template, load_tokenizer_m2
from m2_kv_cache import make_kv_caches


MODEL_NAME = "MiniMax-M2.7-NVFP4"
MODEL_DIR = None
TOKENIZER = None
MAX_CTX = 8192
EOS_TOKENS = {1, 2}
SCHEDULER = None


@dataclass
class CompletionJob:
    request_id: str
    model: str
    prompt_ids: list[int]
    max_tokens: int
    stop: list[str] | None
    created: int = field(default_factory=lambda: int(time.time()))
    event: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None
    error: str | None = None

    @property
    def key(self):
        return (len(self.prompt_ids), self.max_tokens, tuple(self.stop or ()))


def _json_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return None
    return json.loads(handler.rfile.read(length))


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


def _make_response(job, content, finish_reason, completion_tokens, batch_size):
    return {
        "id": job.request_id,
        "object": "chat.completion",
        "created": job.created,
        "model": job.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(job.prompt_ids),
            "completion_tokens": completion_tokens,
            "total_tokens": len(job.prompt_ids) + completion_tokens,
        },
        "rvllm": {"microbatch_size": batch_size},
    }


class M2BatchScheduler:
    def __init__(self, mesh, model, cos, sin, batch_size, max_ctx, wait_ms):
        self.mesh = mesh
        self.model = model
        self.cos = cos
        self.sin = sin
        self.batch_size = batch_size
        self.max_ctx = max_ctx
        self.wait_s = wait_ms / 1000.0
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.fwd = jax.jit(
            m2._decode_one_step,
            static_argnames=("mesh",),
            donate_argnums=(4,),
        )
        self.thread = threading.Thread(target=self._run, name="m2-batcher", daemon=True)
        self.thread.start()

    def submit(self, job):
        self.jobs.put(job)

    def _run(self):
        while not self.stop.is_set():
            try:
                first = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.wait_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = self.jobs.get(timeout=remaining)
                except queue.Empty:
                    break
                if candidate.key == first.key:
                    batch.append(candidate)
                else:
                    self.jobs.put(candidate)
                    break
            self._complete_batch(batch)

    def _complete_batch(self, jobs):
        try:
            responses = self._generate_batch(jobs)
            for job, resp in zip(jobs, responses):
                job.response = resp
                job.event.set()
        except Exception as exc:
            for job in jobs:
                job.error = str(exc)
                job.event.set()

    def _generate_batch(self, jobs):
        real_b = len(jobs)
        padded = list(jobs)
        while len(padded) < self.batch_size:
            padded.append(jobs[-1])

        caches = make_kv_caches(self.batch_size, self.max_ctx, self.mesh)
        prompt_len = len(jobs[0].prompt_ids)
        completions = [[] for _ in jobs]
        text_parts = [[] for _ in jobs]
        finished = [False] * real_b
        finish_reasons = ["length"] * real_b

        next_tok = None
        for step in range(prompt_len):
            toks = [job.prompt_ids[step] for job in padded]
            tok_arr = jnp.asarray(toks, dtype=jnp.int32)
            pos = jnp.int32(step)
            ctx = jnp.full((self.batch_size,), step + 1, dtype=jnp.int32)
            next_tok, _lp, caches = self.fwd(
                tok_arr, pos, ctx, self.model, caches, self.cos, self.sin, self.mesh)
            next_tok.block_until_ready()

        last = [int(x) for x in list(next_tok)]
        for decode_step in range(jobs[0].max_tokens):
            for i, tok in enumerate(last[:real_b]):
                if finished[i]:
                    continue
                if tok in EOS_TOKENS:
                    finished[i] = True
                    finish_reasons[i] = "stop"
                    continue
                piece = TOKENIZER.decode([tok])
                text_parts[i].append(piece)
                completions[i].append(tok)
                if jobs[i].stop and any(ss in "".join(text_parts[i]) for ss in jobs[i].stop):
                    finished[i] = True
                    finish_reasons[i] = "stop"

            if all(finished) or prompt_len + decode_step + 1 >= self.max_ctx - 1:
                break

            feed = [last[i] if i < real_b and not finished[i] else TOKENIZER.eos_token_id
                    for i in range(self.batch_size)]
            pos_i = prompt_len + decode_step
            tok_arr = jnp.asarray(feed, dtype=jnp.int32)
            pos = jnp.int32(pos_i)
            ctx = jnp.full((self.batch_size,), pos_i + 1, dtype=jnp.int32)
            next_tok, _lp, caches = self.fwd(
                tok_arr, pos, ctx, self.model, caches, self.cos, self.sin, self.mesh)
            next_tok.block_until_ready()
            last = [int(x) for x in list(next_tok)]

        return [
            _make_response(
                job,
                "".join(text_parts[i]),
                finish_reasons[i],
                len(completions[i]),
                real_b,
            )
            for i, job in enumerate(jobs)
        ]


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", file=sys.stderr)

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        })

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rvllm",
            }]})
        else:
            self._send_error(404, "not found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_error(404, "not found")
            return
        body = _json_body(self)
        if not body:
            self._send_error(400, "empty request body")
            return
        if body.get("stream", False):
            self._send_error(400, "streaming is not supported by m2_api_server.py")
            return
        if float(body.get("temperature", 0.0)) != 0.0:
            self._send_error(400, "MiniMax M2 TPU server currently supports temperature=0 only")
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error(400, "'messages' is required and must be a non-empty list")
            return

        prompt = apply_chat_template(messages, model_dir=MODEL_DIR)
        prompt_ids = [TOKENIZER.bos_token_id] + list(TOKENIZER.encode(prompt))
        if len(prompt_ids) >= MAX_CTX - 1:
            self._send_error(400, f"prompt too long: {len(prompt_ids)} tokens >= max_ctx {MAX_CTX}")
            return

        max_tokens = int(body.get("max_tokens", 256))
        max_tokens = max(0, min(max_tokens, MAX_CTX - len(prompt_ids) - 1))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif stop is not None and not isinstance(stop, list):
            self._send_error(400, "'stop' must be a string or list of strings")
            return

        job = CompletionJob(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=body.get("model", MODEL_NAME),
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            stop=stop,
        )
        SCHEDULER.submit(job)
        if not job.event.wait(timeout=float(os.environ.get("RVLLM_M2_HTTP_TIMEOUT", "600"))):
            self._send_error(503, "request timed out waiting for generation")
            return
        if job.error:
            self._send_error(500, job.error)
            return
        self._send_json(200, job.response)


def main():
    global MODEL_DIR, MODEL_NAME, TOKENIZER, MAX_CTX, SCHEDULER

    ap = argparse.ArgumentParser(description="MiniMax-M2.7 OpenAI-compatible TPU server")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batch-wait-ms", type=float, default=3.0)
    ap.add_argument("--path", choices=["A", "B"], default="A")
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    MAX_CTX = args.max_ctx
    MODEL_NAME = args.model_name or os.path.basename(args.model_dir.rstrip("/"))

    print(f"loading tokenizer from {args.model_dir}", file=sys.stderr)
    TOKENIZER = load_tokenizer_m2(args.model_dir)

    m2.PATH = args.path
    m2.B = args.batch_size
    m2.load_config_m2(args.model_dir)
    mesh, _axes = m2.make_mesh_v6e8()
    print(f"mesh: {mesh}", file=sys.stderr)
    model = m2.load_model_m2(args.model_dir, mesh, args.max_ctx, path=args.path)

    cos_np, sin_np = precompute_rope_m2(m2.ROPE_THETA, m2.ROTARY_DIM, args.max_ctx)
    cos = jax.device_put(jnp.asarray(cos_np), NamedSharding(mesh, P(None, None)))
    sin = jax.device_put(jnp.asarray(sin_np), NamedSharding(mesh, P(None, None)))

    SCHEDULER = M2BatchScheduler(
        mesh, model, cos, sin, args.batch_size, args.max_ctx, args.batch_wait_ms)

    print("warming up fixed microbatch...", file=sys.stderr, flush=True)
    warm = CompletionJob("warmup", MODEL_NAME, [TOKENIZER.bos_token_id], 1, None)
    SCHEDULER._complete_batch([warm])
    if warm.error:
        raise RuntimeError(warm.error)

    server = ThreadingHTTPServer((args.host, args.port), APIHandler)
    print(f"serving on {args.host}:{args.port}", file=sys.stderr)
    print(f"  model:       {MODEL_NAME}", file=sys.stderr)
    print(f"  max_ctx:     {MAX_CTX}", file=sys.stderr)
    print(f"  microbatch:  {args.batch_size} wait={args.batch_wait_ms}ms", file=sys.stderr)
    print("  POST /v1/chat/completions", file=sys.stderr)
    print("  GET  /v1/models", file=sys.stderr)
    print("  GET  /health", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
