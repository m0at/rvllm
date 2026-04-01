#!/usr/bin/env python3
"""Apples-to-apples 10k-token lifecycle benchmark: launch to last token."""

import asyncio
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

MODEL = "/root/models/Qwen2.5-7B"
RESULTS_DIR = Path("/root/bench_results")
BENCH_CLIENT = Path("/root/rvllm/deploy/benchmark_client.py")

RVLLM_CMD = [
    "/root/rvllm/target/release/rvllm", "serve",
    "--model", MODEL,
    "--dtype", "half",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.90",
    "--gpu-memory-reserve-gb", "0.0",
    "--port", "8000",
]

VLLM_CMD = [
    "/root/venv/bin/vllm", "serve", MODEL,
    "--dtype", "half",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.90",
    "--host", "0.0.0.0",
    "--port", "8000",
]

HEALTH_TIMEOUT = 300
CONCURRENCY = 32
TARGET_COMPLETION_TOKENS = 10_000
MAX_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 1.0
# 128 * 128 = 16,384 max completion tokens. This gives plenty of headroom while
# keeping the race small and stable.
NUM_REQUESTS = max(CONCURRENCY * 2, math.ceil(TARGET_COMPLETION_TOKENS / MAX_TOKENS) + CONCURRENCY)
MAX_REASONABLE_THROUGHPUT = 50_000
PROBE_PROMPT = "Count to one."

def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def port_closed(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.5)
    return False


async def wait_first_completion(port: int, timeout: float = HEALTH_TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROBE_PROMPT,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        body = await r.json()
                        usage = body.get("usage", {})
                        if usage.get("completion_tokens", 0) > 0:
                            return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


def read_last_lines(path: Path, count: int = 200) -> str:
    if not path.exists():
        return ""
    with open(path) as f:
        return "".join(f.readlines()[-count:])


def fail_with_logs(message: str, log_path: Path):
    print(message)
    tail = read_last_lines(log_path)
    if tail:
        print("\n--- server log tail ---")
        print(tail)


def run_benchmark_via_client(port: int, output_path: Path):
    cmd = [
        sys.executable,
        str(BENCH_CLIENT),
        "--url", f"http://127.0.0.1:{port}",
        "--model", MODEL,
        "--concurrent", str(CONCURRENCY),
        "--max-tokens", str(MAX_TOKENS),
        "--temperature", str(TEMPERATURE),
        "--top-p", str(TOP_P),
        "--target-completion-tokens", str(TARGET_COMPLETION_TOKENS),
        "--output", str(output_path),
    ]

    driver_start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    driver_wall = time.perf_counter() - driver_start

    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"benchmark client failed with exit code {proc.returncode}")

    if not output_path.exists():
        raise RuntimeError(f"benchmark client did not produce {output_path}")

    with open(output_path) as f:
        result = json.load(f)

    result["benchmark_driver_wall_sec"] = round(driver_wall, 2)
    return result


def validate_benchmark(name: str, bench: dict):
    required = [
        "total_time_sec",
        "total_completion_tokens",
        "throughput_tok_per_sec",
        "avg_latency_ms",
        "successful_requests",
        "num_errors",
    ]
    missing = [key for key in required if key not in bench]
    if missing:
        raise RuntimeError(f"{name}: benchmark result missing fields: {', '.join(missing)}")

    if bench["successful_requests"] <= 0:
        raise RuntimeError(f"{name}: no successful requests")

    if bench["num_errors"] != 0:
        raise RuntimeError(f"{name}: benchmark had {bench['num_errors']} request errors")

    if bench["total_completion_tokens"] < TARGET_COMPLETION_TOKENS:
        raise RuntimeError(
            f"{name}: only {bench['total_completion_tokens']} completion tokens, need >= {TARGET_COMPLETION_TOKENS}"
        )

    if bench["throughput_tok_per_sec"] <= 0 or bench["avg_latency_ms"] <= 0:
        raise RuntimeError(f"{name}: non-positive throughput or latency")

    if bench["throughput_tok_per_sec"] > MAX_REASONABLE_THROUGHPUT:
        raise RuntimeError(
            f"{name}: suspicious throughput {bench['throughput_tok_per_sec']:.2f} tok/s exceeds sanity ceiling"
        )


def run_lifecycle(name: str, cmd: list, log_path: Path, result_path: Path):
    print(f"\n{'='*60}")
    print(f"  {name} lifecycle benchmark")
    print(f"{'='*60}")

    log_f = open(log_path, "w")

    print(f"[{name}] Starting server...")
    launch_wall_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    pid = proc.pid
    print(f"[{name}] PID={pid}")

    first_completion = asyncio.run(wait_first_completion(8000, HEALTH_TIMEOUT))
    first_completion_sec = round(time.perf_counter() - launch_wall_start, 2)

    if not first_completion:
        print(f"[{name}] FAILED to serve first completion in {HEALTH_TIMEOUT}s")
        log_f.close()
        with open(log_path) as f:
            lines = f.readlines()
        print("".join(lines[-200:]))
        proc.kill()
        proc.wait()
        return None

    print(f"[{name}] First completion served after {first_completion_sec}s")

    # Benchmark
    bench_output_path = RESULTS_DIR / f"{name.lower().replace(' ', '_')}_bench.json"
    print(f"[{name}] Running benchmark (concurrency={CONCURRENCY}, target_tokens={TARGET_COMPLETION_TOKENS}, max_tokens={MAX_TOKENS})...")
    try:
        bench = run_benchmark_via_client(8000, bench_output_path)
        validate_benchmark(name, bench)
    except Exception as exc:
        fail_with_logs(f"[{name}] BENCHMARK FAILURE: {exc}", log_path)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.kill(pid, signal.SIGKILL)
            proc.wait()
        log_f.close()
        return None

    bench_wall = round(float(bench["total_time_sec"]), 2)
    launch_to_finished_tokens_sec = round(time.perf_counter() - launch_wall_start, 2)

    print(
        f"[{name}] Benchmark done: {bench['total_completion_tokens']} tokens, "
        f"{bench['throughput_tok_per_sec']:.2f} tok/s, {bench['avg_latency_ms']:.2f} ms avg"
    )

    # Shutdown
    print(f"[{name}] Sending SIGTERM to PID {pid}...")
    t_shut = time.time()
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        proc.wait(timeout=60)
        shutdown_ok = True
    except subprocess.TimeoutExpired:
        print(f"[{name}] SIGTERM timeout, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
        proc.wait()
        shutdown_ok = False

    shutdown_sec = round(time.time() - t_shut, 2)
    log_f.close()

    # Verify port closed
    closed = port_closed(8000, timeout=15)
    print(f"[{name}] Port closed: {closed}, shutdown_sec={shutdown_sec}")

    result = {
        "engine": name,
        "startup_sec": first_completion_sec,
        "launch_to_first_completion_sec": first_completion_sec,
        "benchmark_wall_sec": bench_wall,
        "shutdown_sec": shutdown_sec,
        "launch_to_finished_tokens_sec": launch_to_finished_tokens_sec,
        "end_to_end_sec": launch_to_finished_tokens_sec,
        "total_completion_tokens": bench["total_completion_tokens"],
        "throughput_tok_per_sec": bench["throughput_tok_per_sec"],
        "avg_latency_ms": bench["avg_latency_ms"],
        "errors": bench["num_errors"],
        "successful_requests": bench["successful_requests"],
        "requests_per_sec": bench["requests_per_sec"],
        "p50_latency_ms": bench["p50_latency_ms"],
        "p95_latency_ms": bench["p95_latency_ms"],
        "p99_latency_ms": bench["p99_latency_ms"],
        "benchmark_output": str(bench_output_path),
        "shutdown_ok": shutdown_ok,
        "port_closed": closed,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{name}] Result saved to {result_path}")
    return result


def main():
    engine = os.environ.get("BENCH_ENGINE", "both").strip().lower()
    ensure_results_dir()
    print("=== rvLLM vs vLLM Lifecycle Benchmark ===")
    print(f"Model: {MODEL}")
    print(
        f"Target completion tokens: {TARGET_COMPLETION_TOKENS}, "
        f"Concurrency: {CONCURRENCY}, Requests: {NUM_REQUESTS}, MaxTokens: {MAX_TOKENS}"
    )

    rv_result = None
    vl_result = None
    if engine in {"both", "rvllm"}:
        rv_result = run_lifecycle(
            "rvLLM",
            RVLLM_CMD,
            RESULTS_DIR / "rvllm.log",
            RESULTS_DIR / "rvllm_result.json",
        )
    if engine == "both":
        time.sleep(5)
    if engine in {"both", "vllm"}:
        vl_result = run_lifecycle(
            "vLLM",
            VLLM_CMD,
            RESULTS_DIR / "vllm.log",
            RESULTS_DIR / "vllm_result.json",
        )

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    headers = [
        "engine", "launch_to_first_completion_sec", "benchmark_wall_sec", "shutdown_sec",
        "launch_to_finished_tokens_sec", "total_completion_tokens", "throughput_tok_per_sec",
        "avg_latency_ms", "shutdown_ok",
    ]

    # Header row
    print(f"{'Metric':<28} {'rvLLM':>16} {'vLLM':>16}")
    print("-" * 62)

    for h in headers[1:]:
        rv_val = rv_result.get(h, "N/A") if rv_result else "FAILED"
        vl_val = vl_result.get(h, "N/A") if vl_result else "FAILED"
        print(f"  {h:<26} {str(rv_val):>16} {str(vl_val):>16}")

    print("=" * 80)

    # Save combined results
    combined = {"rvllm": rv_result, "vllm": vl_result}
    with open(RESULTS_DIR / "combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {RESULTS_DIR / 'combined_results.json'}")


if __name__ == "__main__":
    main()
