#!/usr/bin/env python3
"""Response format parity test: rvLLM vs Python vLLM.

Verifies that the JSON response structure, field names, types, nesting,
streaming SSE format, and error responses match between servers.

Usage:
    python3 tests/parity/response_format_parity.py \
        --rust-url http://localhost:8000 \
        --python-url http://localhost:8001 \
        --model Qwen/Qwen2.5-1.5B
"""
import argparse, json, sys, requests

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def check_type(name, obj, key, expected_type):
    if key not in obj:
        check(f"{name}.{key} exists", False, f"missing key '{key}'")
        return False
    actual = type(obj[key]).__name__
    ok = isinstance(obj[key], expected_type)
    check(f"{name}.{key} is {expected_type.__name__}", ok,
          f"got {actual}" if not ok else "")
    return ok


def check_field(name, obj, key):
    ok = key in obj
    check(f"{name} has '{key}'", ok)
    return ok


def compare_field_sets(label_r, rust_obj, label_p, python_obj):
    r_keys = set(rust_obj.keys())
    p_keys = set(python_obj.keys())
    missing_in_rust = p_keys - r_keys
    extra_in_rust = r_keys - p_keys
    if missing_in_rust:
        check(f"Rust response missing fields vs Python", False,
              f"missing: {missing_in_rust}")
    if extra_in_rust:
        # Extra fields are a warning, not failure
        print(f"  [INFO] Rust has extra fields vs Python: {extra_in_rust}")
    if not missing_in_rust:
        check(f"Rust has all Python response fields", True)


# -- Completion format tests --------------------------------------------------

def test_completion_format(rust_url, python_url, model):
    print("\n--- Completion Response Format ---")
    payload = {"model": model, "prompt": "Hello", "max_tokens": 5, "temperature": 0}

    try:
        rust_r = requests.post(f"{rust_url}/v1/completions", json=payload, timeout=30)
        python_r = requests.post(f"{python_url}/v1/completions", json=payload, timeout=30)
    except requests.RequestException as e:
        check("Server reachable", False, str(e))
        return

    check("Rust status 200", rust_r.status_code == 200, f"got {rust_r.status_code}")
    check("Python status 200", python_r.status_code == 200, f"got {python_r.status_code}")
    if rust_r.status_code != 200 or python_r.status_code != 200:
        return

    rd = rust_r.json()
    pd = python_r.json()

    compare_field_sets("rust", rd, "python", pd)

    # Required top-level fields
    for field, typ in [("id", str), ("object", str), ("created", int), ("model", str)]:
        check_type("rust", rd, field, typ)
        check_type("python", pd, field, typ)

    check("rust.object == 'text_completion'", rd.get("object") == "text_completion",
          f"got {rd.get('object')}")
    check("python.object == 'text_completion'", pd.get("object") == "text_completion",
          f"got {pd.get('object')}")

    # choices array
    check_type("rust", rd, "choices", list)
    check_type("python", pd, "choices", list)
    if rd.get("choices") and pd.get("choices"):
        rc = rd["choices"][0]
        pc = pd["choices"][0]
        compare_field_sets("rust.choices[0]", rc, "python.choices[0]", pc)
        for field in ["text", "index", "finish_reason"]:
            check_field("rust.choices[0]", rc, field)
        check_type("rust.choices[0]", rc, "index", int)
        check_type("rust.choices[0]", rc, "text", str)

    # usage object
    check_field("rust", rd, "usage")
    check_field("python", pd, "usage")
    if rd.get("usage") and pd.get("usage"):
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            check_field("rust.usage", rd["usage"], field)
            check_field("python.usage", pd["usage"], field)
            check_type("rust.usage", rd["usage"], field, int)


# -- Chat completion format tests ---------------------------------------------

def test_chat_format(rust_url, python_url, model):
    print("\n--- Chat Completion Response Format ---")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5, "temperature": 0,
    }

    try:
        rust_r = requests.post(f"{rust_url}/v1/chat/completions", json=payload, timeout=30)
        python_r = requests.post(f"{python_url}/v1/chat/completions", json=payload, timeout=30)
    except requests.RequestException as e:
        check("Server reachable", False, str(e))
        return

    check("Rust status 200", rust_r.status_code == 200, f"got {rust_r.status_code}")
    check("Python status 200", python_r.status_code == 200, f"got {python_r.status_code}")
    if rust_r.status_code != 200 or python_r.status_code != 200:
        return

    rd = rust_r.json()
    pd = python_r.json()

    compare_field_sets("rust", rd, "python", pd)

    check("rust.object == 'chat.completion'", rd.get("object") == "chat.completion",
          f"got {rd.get('object')}")

    if rd.get("choices") and pd.get("choices"):
        rc = rd["choices"][0]
        pc = pd["choices"][0]
        compare_field_sets("rust.choices[0]", rc, "python.choices[0]", pc)
        check_field("rust.choices[0]", rc, "message")
        if rc.get("message"):
            check_field("rust.choices[0].message", rc["message"], "role")
            check_field("rust.choices[0].message", rc["message"], "content")
            check("rust role == 'assistant'", rc["message"].get("role") == "assistant",
                  f"got {rc['message'].get('role')}")


# -- Streaming format tests ---------------------------------------------------

def parse_sse_stream(response):
    """Parse SSE stream into chunks and metadata."""
    chunks = []
    has_done = False
    raw_lines = []
    for line in response.iter_lines():
        decoded = line.decode() if isinstance(line, bytes) else line
        raw_lines.append(decoded)
        if decoded == "data: [DONE]":
            has_done = True
        elif decoded.startswith("data: "):
            try:
                chunks.append(json.loads(decoded[6:]))
            except json.JSONDecodeError:
                pass
    return chunks, has_done, raw_lines


def test_streaming_format(rust_url, python_url, model):
    print("\n--- Streaming SSE Format ---")

    # Completion streaming
    payload = {"model": model, "prompt": "Hello", "max_tokens": 10,
               "temperature": 0, "stream": True}
    try:
        rust_r = requests.post(f"{rust_url}/v1/completions", json=payload,
                               stream=True, timeout=30)
        python_r = requests.post(f"{python_url}/v1/completions", json=payload,
                                 stream=True, timeout=30)
    except requests.RequestException as e:
        check("Streaming server reachable", False, str(e))
        return

    r_chunks, r_done, _ = parse_sse_stream(rust_r)
    p_chunks, p_done, _ = parse_sse_stream(python_r)

    check("Rust stream has chunks", len(r_chunks) > 0, f"got {len(r_chunks)}")
    check("Python stream has chunks", len(p_chunks) > 0, f"got {len(p_chunks)}")
    check("Rust stream has [DONE] marker", r_done)
    check("Python stream has [DONE] marker", p_done)

    if r_chunks:
        rc = r_chunks[0]
        check("Rust chunk has 'id'", "id" in rc)
        check("Rust chunk has 'object'", "object" in rc)
        check("Rust chunk has 'choices'", "choices" in rc)
        if rc.get("choices"):
            check("Rust chunk.choices[0] has 'text'", "text" in rc["choices"][0])

    # Chat streaming
    payload_chat = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10, "temperature": 0, "stream": True,
    }
    try:
        rust_r2 = requests.post(f"{rust_url}/v1/chat/completions", json=payload_chat,
                                stream=True, timeout=30)
        python_r2 = requests.post(f"{python_url}/v1/chat/completions", json=payload_chat,
                                  stream=True, timeout=30)
    except requests.RequestException as e:
        check("Chat streaming server reachable", False, str(e))
        return

    r_chunks2, r_done2, _ = parse_sse_stream(rust_r2)
    p_chunks2, p_done2, _ = parse_sse_stream(python_r2)

    check("Rust chat stream has chunks", len(r_chunks2) > 0)
    check("Rust chat stream has [DONE]", r_done2)
    if r_chunks2:
        rc2 = r_chunks2[0]
        check("Rust chat chunk.object == 'chat.completion.chunk'",
              rc2.get("object") == "chat.completion.chunk",
              f"got {rc2.get('object')}")
        if rc2.get("choices"):
            delta = rc2["choices"][0]
            check("Rust chat chunk has 'delta'", "delta" in delta,
                  f"keys: {list(delta.keys())}")


# -- Error format tests -------------------------------------------------------

def test_error_format(rust_url, python_url, model):
    print("\n--- Error Response Format ---")

    # Missing prompt
    payload = {"model": model, "max_tokens": 5}
    try:
        rust_r = requests.post(f"{rust_url}/v1/completions", json=payload, timeout=10)
        python_r = requests.post(f"{python_url}/v1/completions", json=payload, timeout=10)
    except requests.RequestException as e:
        check("Error test server reachable", False, str(e))
        return

    check("Rust error status 4xx", 400 <= rust_r.status_code < 500,
          f"got {rust_r.status_code}")
    check("Python error status 4xx", 400 <= python_r.status_code < 500,
          f"got {python_r.status_code}")

    # Check error body structure
    try:
        rd = rust_r.json()
        check("Rust error is JSON", True)
        # vLLM typically returns {"object": "error", "message": "...", "type": "...", "code": ...}
        # or {"error": {"message": ..., "type": ..., "code": ...}}
        has_error_obj = "error" in rd
        has_message = "message" in rd or (has_error_obj and "message" in rd.get("error", {}))
        check("Rust error has message", has_message, f"body: {json.dumps(rd)[:200]}")
    except (json.JSONDecodeError, ValueError):
        check("Rust error is JSON", False)

    try:
        pd = python_r.json()
        check("Python error is JSON", True)
    except (json.JSONDecodeError, ValueError):
        check("Python error is JSON", False)

    # Invalid model
    payload2 = {"model": "nonexistent-model-xyz", "prompt": "Hi", "max_tokens": 5}
    try:
        rust_r2 = requests.post(f"{rust_url}/v1/completions", json=payload2, timeout=10)
        python_r2 = requests.post(f"{python_url}/v1/completions", json=payload2, timeout=10)
        check("Rust invalid model -> error", rust_r2.status_code >= 400,
              f"got {rust_r2.status_code}")
        check("Python invalid model -> error", python_r2.status_code >= 400,
              f"got {python_r2.status_code}")
    except requests.RequestException:
        pass

    # Negative max_tokens
    payload3 = {"model": model, "prompt": "Hi", "max_tokens": -1}
    try:
        rust_r3 = requests.post(f"{rust_url}/v1/completions", json=payload3, timeout=10)
        python_r3 = requests.post(f"{python_url}/v1/completions", json=payload3, timeout=10)
        # Both should either error or handle gracefully
        check("Rust/Python agree on negative max_tokens handling",
              (rust_r3.status_code >= 400) == (python_r3.status_code >= 400),
              f"rust={rust_r3.status_code} python={python_r3.status_code}")
    except requests.RequestException:
        pass


# -- Models endpoint format ---------------------------------------------------

def test_models_format(rust_url, python_url, model):
    print("\n--- Models Endpoint Format ---")
    try:
        rust_r = requests.get(f"{rust_url}/v1/models", timeout=10)
        python_r = requests.get(f"{python_url}/v1/models", timeout=10)
    except requests.RequestException as e:
        check("Models endpoint reachable", False, str(e))
        return

    check("Rust models 200", rust_r.status_code == 200)
    check("Python models 200", python_r.status_code == 200)

    if rust_r.status_code == 200:
        rd = rust_r.json()
        check_field("rust /v1/models", rd, "object")
        check_field("rust /v1/models", rd, "data")
        check("rust models.object == 'list'", rd.get("object") == "list",
              f"got {rd.get('object')}")
        if rd.get("data") and len(rd["data"]) > 0:
            m = rd["data"][0]
            check_field("rust model entry", m, "id")
            check_field("rust model entry", m, "object")


def main():
    p = argparse.ArgumentParser(description="Response format parity: rvLLM vs Python vLLM")
    p.add_argument("--rust-url", default="http://localhost:8000")
    p.add_argument("--python-url", default="http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = p.parse_args()

    print(f"Response Format Parity Test: rvLLM vs Python vLLM")
    print(f"  Rust:   {args.rust_url}")
    print(f"  Python: {args.python_url}")
    print(f"  Model:  {args.model}")

    test_completion_format(args.rust_url, args.python_url, args.model)
    test_chat_format(args.rust_url, args.python_url, args.model)
    test_streaming_format(args.rust_url, args.python_url, args.model)
    test_error_format(args.rust_url, args.python_url, args.model)
    test_models_format(args.rust_url, args.python_url, args.model)

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"SUMMARY: {PASS}/{total} checks passed, {FAIL}/{total} failed")
    if FAIL == 0:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
