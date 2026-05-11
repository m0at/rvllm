#!/usr/bin/env python3
"""NVFP4 + prefill server-level stability harness (sm_121).

Spawns `rvllm-server` with varying `RVLLM_NVFP4_KV` / `RVLLM_BATCH_PREFILL`
combinations, sends prompts of increasing length to /v1/chat/completions,
and records TTFT / completion tokens / verdict. Each case has a hard
per-request timeout (default 120 s). The RAM gate requires MemAvailable
>= 80 GB unless `--force-ram` is passed — Gemma 4 31B fp8-block + arena
needs ~75 GB resident, and loading on top of a near-full system pages
catastrophically on GB10 unified memory.

Usage:
  ~/.venv/bin/python3 v3/tools/nvfp4_server_stability_check.py [options]
    --force-ram         skip the MemAvailable>=80 GB precheck
    --arena-gb N        RVLLM_ARENA_GB (default 60)
    --timeout-secs N    per-request hard timeout (default 120)
    --only NAME         run only the case with this name
    --binary PATH       rvllm-server binary (default auto)
    --model-dir PATH    RVLLM_MODEL_DIR (default Gemma 4 31B fp8-block)
    --log-dir PATH      server+harness logs (default $PWD/nvfp4_stability_logs)
    --quiet             drop progress prints, report summary only
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_BINARY = REPO_ROOT / "v3" / "target" / "release" / "rvllm-server"
DEFAULT_MODEL = pathlib.Path.home() / ".vllm" / "models" / "gemma-4-31b-it-fp8-block"
DEFAULT_KERNELS = REPO_ROOT / "kernels"
DEFAULT_CUTLASS_SO = DEFAULT_KERNELS / "sm_121" / "libcutlass_sm120.so"
DEFAULT_LOG_DIR = pathlib.Path.cwd() / "nvfp4_stability_logs"

PORT = 8010  # match the service default — harness asserts it's free on start

# --- prompts --------------------------------------------------------------

# One roughly-30-token German sentence, repeated to hit target lengths.
PARAGRAPH = (
    "Der Satz des Pythagoras besagt, dass in einem rechtwinkligen Dreieck "
    "das Quadrat der Hypotenuse gleich der Summe der Quadrate der beiden "
    "Katheten ist. "
)
QUESTION = "Antworte in einem einzigen Satz: wie heißt die Hauptstadt von Frankreich?"


def make_prompt(target_tokens: int) -> str:
    """Pad `PARAGRAPH` until ~`target_tokens`, append the question."""
    # Approx 4.5 chars per token for mixed German prose w/ SentencePiece.
    target_chars = max(int(target_tokens * 4.5) - len(QUESTION), 0)
    padding = ""
    while len(padding) < target_chars:
        padding += PARAGRAPH
    return padding + QUESTION


# --- cases ----------------------------------------------------------------
# (name, server_env, prompt_target_tokens, required_substr_in_reply or None)
# `required_substr_in_reply` is a loose sanity check: case fails if the
# reply text doesn't contain this substring (case-insensitive). None skips
# the content check and only asserts "reply is non-empty and on time".

CASES = [
    ("nvfp4-pertoken-short",   {"RVLLM_NVFP4_KV": "1"},                                      20,   "paris"),
    ("nvfp4-pertoken-medium",  {"RVLLM_NVFP4_KV": "1"},                                     256,   "paris"),
    ("nvfp4-batch-short",      {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},          20,   "paris"),
    ("nvfp4-batch-medium",     {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},         256,   "paris"),
    ("nvfp4-batch-large",      {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},        1024,   "paris"),
    ("nvfp4-batch-xlarge",     {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},        2048,   "paris"),
    # Long-context cases — post-Phase-2b (commit 8e8f517) the NVFP4
    # unified kernel makes these tractable, but the n² tile walk
    # still bites at 15k+. These cases are the regression net for
    # BLOCK_M / tile-size perf work + chunked prefill follow-ups.
    ("nvfp4-batch-5k",         {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},        5120,   "paris"),
    ("nvfp4-batch-10k",        {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},       10240,   "paris"),
    ("nvfp4-batch-15k",        {"RVLLM_NVFP4_KV": "1", "RVLLM_BATCH_PREFILL": "1"},       15360,   "paris"),
    # Reference: same sizes on FP8-batch (known-good after 04fec77).
    ("fp8-batch-medium",       {"RVLLM_BATCH_PREFILL": "1"},                                256,   "paris"),
    ("fp8-batch-large",        {"RVLLM_BATCH_PREFILL": "1"},                               1024,   "paris"),
    ("fp8-batch-15k",          {"RVLLM_BATCH_PREFILL": "1"},                              15360,   "paris"),
]

# --- helpers --------------------------------------------------------------


def check_ram(force: bool, quiet: bool) -> None:
    # MemAvailable is the kernel's post-reclaim estimate — what we actually
    # get before swapping. Plain MemFree under-reports because most of our
    # ~44 GB page cache is reclaimable.
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                break
        else:
            raise RuntimeError("MemAvailable missing from /proc/meminfo")
    available_gb = kb / (1024 * 1024)
    if not quiet:
        print(f"  MemAvailable = {available_gb:.1f} GiB (gate: >= 80 GiB)")
    if available_gb < 80.0 and not force:
        raise SystemExit(
            f"\nRAM gate failed: only {available_gb:.1f} GiB available, "
            f"need >= 80 GiB.\n"
            f"  - stop rvllm-serve / vllm-embedding / zeroclaw first, OR\n"
            f"  - re-run with --force-ram to skip (risks OOM during model "
            f"load on GB10 unified memory)."
        )


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def kill_any_server() -> None:
    subprocess.run(["pkill", "-f", "rvllm-server"], check=False)
    # wait for port free
    deadline = time.time() + 15
    while port_open(PORT):
        if time.time() > deadline:
            raise RuntimeError(f"port {PORT} stuck busy after pkill")
        time.sleep(0.5)


def launch_server(binary: pathlib.Path, env_overrides: dict[str, str],
                  model_dir: pathlib.Path, arena_gb: int,
                  log_path: pathlib.Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "RVLLM_BIND": f"127.0.0.1:{PORT}",
        "RVLLM_MODEL_ID": "gemma-4-31b-it",
        "RVLLM_MODEL_DIR": str(model_dir),
        "RVLLM_KERNELS_DIR": str(DEFAULT_KERNELS),
        "RVLLM_CUTLASS_SM120_SO": str(DEFAULT_CUTLASS_SO),
        "RVLLM_FA3_SO": str(DEFAULT_CUTLASS_SO),  # placeholder, unused on sm_121
        "RVLLM_FP8_GEMM_CUTLASS_SM120": "1",
        "RVLLM_ARENA_GB": str(arena_gb),
        "RVLLM_F16_KV": "0",
        "RVLLM_MAX_TOKENS_CAP": "4096",
        "RUST_LOG": "rvllm_serve=info,rvllm_runtime=info",
    })
    # Scrub any BATCH_PREFILL / NVFP4 leaking in from the outer shell so
    # env_overrides is authoritative.
    env.pop("RVLLM_BATCH_PREFILL", None)
    env.pop("RVLLM_NVFP4_KV", None)
    env.update(env_overrides)

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(binary)],
        stdout=log_f, stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return proc


def wait_ready(proc: subprocess.Popen, boot_timeout: int = 120) -> None:
    deadline = time.time() + boot_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code={proc.returncode}); see log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",
                                        timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server did not open /v1/models within {boot_timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)


def send_prompt(prompt: str, max_tokens: int, timeout_s: int) -> dict:
    payload = json.dumps({
        "model": "gemma-4-31b-it",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read().decode()
    elapsed = time.perf_counter() - t0
    data = json.loads(body)
    reply = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reply": reply,
    }


# --- run loop -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-ram", action="store_true")
    ap.add_argument("--arena-gb", type=int, default=60)
    ap.add_argument("--timeout-secs", type=int, default=120)
    ap.add_argument("--only", default=None, help="run only the case with this name")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary)
    if not binary.exists():
        sys.exit(f"binary not found: {binary}\n  build: "
                 "cargo build --release -p rvllm-serve --features gb10")
    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"model dir not found: {model_dir}")

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"== NVFP4 server stability harness ==")
        print(f"  binary    {binary}")
        print(f"  model     {model_dir}")
        print(f"  arena_gb  {args.arena_gb}")
        print(f"  timeout   {args.timeout_secs}s per request")
        print(f"  logs      {log_dir}")
        print()

    check_ram(args.force_ram, args.quiet)
    if port_open(PORT):
        sys.exit(f"port {PORT} busy — stop rvllm-serve / any other LLM "
                 f"service before running.")

    cases = CASES
    if args.only:
        cases = [c for c in cases if c[0] == args.only]
        if not cases:
            sys.exit(f"no case named {args.only!r}")

    # Group consecutive cases that share env to reuse the server.
    results = []
    i = 0
    while i < len(cases):
        env = cases[i][1]
        group_end = i
        while group_end + 1 < len(cases) and cases[group_end + 1][1] == env:
            group_end += 1
        group = cases[i:group_end + 1]

        env_tag = "_".join(f"{k}={v}" for k, v in sorted(env.items())) or "default"
        server_log = log_dir / f"server_{env_tag}.log"
        if not args.quiet:
            print(f"-- server up [{env_tag}] log={server_log.name}")

        try:
            proc = launch_server(binary, env, model_dir, args.arena_gb, server_log)
            try:
                wait_ready(proc)
            except Exception as e:
                stop_server(proc)
                for name, _env, _ptoks, _need in group:
                    results.append({
                        "case": name, "verdict": "BOOT_FAIL",
                        "detail": str(e), "log": str(server_log),
                    })
                i = group_end + 1
                continue

            for name, _env, ptoks, need in group:
                prompt = make_prompt(ptoks)
                if not args.quiet:
                    print(f"  [{name}] target={ptoks} tok, "
                          f"prompt_chars={len(prompt)}")
                try:
                    r = send_prompt(prompt, max_tokens=48,
                                    timeout_s=args.timeout_secs)
                    verdict = "PASS"
                    detail = ""
                    if r["completion_tokens"] == 0:
                        verdict = "FAIL"
                        detail = "empty reply"
                    elif need and need.lower() not in r["reply"].lower():
                        verdict = "SOFT_FAIL"
                        detail = f"expected substring {need!r} missing"
                    results.append({
                        "case": name, "verdict": verdict, "detail": detail,
                        "elapsed_s": r["elapsed_s"],
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "reply_head": r["reply"][:80].replace("\n", " "),
                        "log": str(server_log),
                    })
                    if not args.quiet:
                        print(f"    -> {verdict} "
                              f"prompt={r['prompt_tokens']} "
                              f"gen={r['completion_tokens']} "
                              f"time={r['elapsed_s']:.1f}s "
                              f"reply={r['reply'][:60]!r}")
                except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                    # Hit the harness-side timeout: server may still be
                    # alive but stuck; kill it so the next case gets a
                    # fresh state.
                    results.append({
                        "case": name, "verdict": "TIMEOUT",
                        "detail": f"> {args.timeout_secs}s ({e.__class__.__name__})",
                        "log": str(server_log),
                    })
                    if not args.quiet:
                        print(f"    -> TIMEOUT (> {args.timeout_secs}s)")
                    # Abort the rest of this group — the server is suspect.
                    break
                except Exception as e:
                    results.append({
                        "case": name, "verdict": "ERROR",
                        "detail": f"{e.__class__.__name__}: {e}",
                        "log": str(server_log),
                    })
                    if not args.quiet:
                        print(f"    -> ERROR {e.__class__.__name__}: {e}")
                    break
        finally:
            stop_server(proc)
            # Let GB10 unified memory drain before next server.
            time.sleep(5)
            kill_any_server()

        i = group_end + 1

    # --- summary ---------------------------------------------------------
    print("\n== summary ==")
    header = f"{'case':26} {'verdict':10} {'prompt':>7} {'gen':>5} {'time':>7}  reply / detail"
    print(header)
    print("-" * len(header))
    any_failed = False
    for r in results:
        v = r["verdict"]
        if v != "PASS":
            any_failed = True
        tail = r.get("reply_head", "") or r.get("detail", "")
        print(f"{r['case']:26} {v:10} "
              f"{r.get('prompt_tokens', 0):>7} "
              f"{r.get('completion_tokens', 0):>5} "
              f"{r.get('elapsed_s', 0):>7.1f}  {tail}")
    print(f"\nlogs under: {log_dir}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
