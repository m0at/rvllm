# MiniMax-M2.7-NVFP4 on TPU v6e-8 — End-to-End Setup Guide

For a new engineer who's never touched this repo. Walk this top-to-bottom and
you'll have the exact same bench we ran, plus a chat interface and an
OpenAI-compatible API server.

---

## 0. What you're building

- **Target model**: `lukealonso/MiniMax-M2.7-NVFP4` (230B params total, 10B active,
  MoE 256 experts top-8, NVFP4 quantized, 62 layers, 3072 hidden, 196K max ctx).
- **Hardware**: 1× Google Cloud TPU v6e-8 slice (8 chips × 32 GB HBM, single host).
- **Software**: pure JAX + XLA on TPU. No custom CUDA / Rust for the inference path.
- **Current throughput** (as of this commit):
  - Load: ~80s (parallel ThreadPool over 61 safetensors shards)
  - **B=1**: 726 ms/step = **1.4 tok/s**
  - **B=8**: 871 ms/step = **9.2 tok/s** (6.7× amortization)
  - B≥16 currently OOMs on KV cache (see "Known limits" below)

---

## 1. Local prereqs

Install on your workstation (macOS/Linux):

```bash
# 1. gcloud CLI (https://cloud.google.com/sdk/docs/install)
#    After install, authenticate:
gcloud auth login
gcloud config set project <YOUR_GCP_PROJECT>

# 2. HuggingFace CLI + token (1.5+ renamed `huggingface-cli` to `hf`)
python3 -m pip install --upgrade huggingface_hub
hf auth login
# Paste your HF token. Verifies model access for lukealonso/MiniMax-M2.7-NVFP4.

# 3. Clone the repo
git clone <rvllm-repo-url>
cd rvllm
```

Verify local tools:

```bash
gcloud --version
hf auth whoami
git rev-parse HEAD
```

---

## 2. GCP TPU quota (one-time)

You need v6e-8 quota in the target zone. Only `europe-west4-a` is confirmed
to ship v6e-8 for most projects as of 2026-04.

1. Open https://console.cloud.google.com/iam-admin/quotas?project=<YOUR_GCP_PROJECT>
2. Filter: service **Cloud TPU API**, metric **TPU v6e cores per project per zone**
3. Request **16 cores** in `europe-west4-a` (one v6e-8 = 8 cores; the extra 8
   lets you run concurrent slices or delete/recreate safely).
4. Wait for approval (usually 1-24 hours).

Confirm your quota:

```bash
gcloud compute tpus accelerator-types list --zone=europe-west4-a | grep v6e-8
```

Should print `v6e-8` among the available types.

---

## 3. Deploy to TPU v6e-8

The repo ships a one-shot deploy script: `tpu/harness/deploy_m2_tpu.sh`. It:

1. Creates a v6e-8 VM (spot by default — save ~50%)
2. Uploads the repo as a SHA-pinned tarball
3. Installs `jax[tpu]`, safetensors, tokenizers, ml_dtypes
4. (Skips Zig build — Path A uses pure JAX)
5. Downloads the 130 GB model to `/dev/shm` (tmpfs, ~2 min with HF auth)
6. Runs a smoke inference (`python3 m2_tpu_infer.py --max-tokens 16`)

### Quick deploy

```bash
# From the repo root. Requires a clean git tree (or ALLOW_DIRTY=1).
SPOT=1 ZONE=europe-west4-a bash tpu/harness/deploy_m2_tpu.sh
```

Environment flags:

| Var | Default | Notes |
|---|---|---|
| `SPOT` | `0` | `1` = preemptible (cheaper, may be reclaimed) |
| `ZONE` | `us-east5-b` | Override to `europe-west4-a` for v6e-8 |
| `TPU_NAME` | `rvllm-m2` | Instance name |
| `ACCELERATOR` | `v6e-8` | Slice size |
| `HF_TOKEN` | `(unset)` | Forwarded to VM for auth'd HF downloads — **highly recommended** (20-100× faster) |
| `ALLOW_DIRTY` | `0` | Allow deploy with uncommitted changes (warn only) |

### Teardown

```bash
gcloud compute tpus tpu-vm delete rvllm-m2 --zone=europe-west4-a
```

---

## 4. Run the bench (same numbers we measured)

SSH to the VM:

```bash
gcloud compute tpus tpu-vm ssh rvllm-m2 --zone=europe-west4-a
```

On the VM, the full-bench harness is `/tmp/m2_full_bench.py` (after deploy).
It sweeps batch sizes, runs perplexity on wikitext-2, and generates 2048 tokens
from a prompt.

```bash
# All-in-one run matching our measured numbers.
# (Expect ~25 min total on first invocation — JIT compile per batch size.)
export PATH="$HOME/.local/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export M2_MOE=shardmap

cd /tmp
python3 -u m2_full_bench.py \
  --model-dir /dev/shm/m2-nvfp4 \
  --batches 1,8 \
  --ctx 2048 \
  --iters 10 \
  --warmup 3 \
  --workers 32 \
  --prompt "Explain angular momentum." \
  --gen-tokens 2048 \
  --ppl-text /tmp/wiki.txt \
  --out /tmp/m2_final.json
```

Flag reference:

| Flag | Effect |
|---|---|
| `--batches 1,8` | Batch sizes to sweep (comma-separated) |
| `--ctx 2048` | Max sequence length |
| `--iters 10` | Timing iterations per batch after warmup |
| `--warmup 3` | Warmup iterations (JIT compile happens here) |
| `--workers 32` | Parallel loader thread count |
| `--prompt ...` | Prompt for the 2048-token coherent-generation test |
| `--gen-tokens 2048` | How many tokens to generate |
| `--ppl-text <path>` | Text file for perplexity (wikitext-2 by default) |
| `--out <path>` | Output JSON |
| `--skip-sweep` | Skip the batch sweep |
| `--skip-ppl` | Skip perplexity |
| `--skip-gen` | Skip generation |
| `--single-batch N` | Run one batch size only (for subprocess-per-batch memory hygiene) |

### MoE dispatch variants (env var `M2_MOE`)

Set before running:

| Value | Impl | B=1 perf | B=8+ perf |
|---|---|---|---|
| `shardmap` (default) | sort + all_to_all dispatch | **1.4 tok/s** | **9.2 tok/s** |
| `dense` | compute all 32 local experts per shard | compile hangs | n/a |
| `gather` | dynamic-gather top-K + vmap | 0.02 tok/s (vmap doesn't fuse) | n/a |

Use `shardmap` unless actively experimenting.

### Cache the compile artifacts to HF

After a successful run, the JAX compile cache + python env can be pushed to a
private HF dataset so the **next** VM skips both pip install and JIT compile:

```bash
# On the VM, after a successful bench:
bash $HOME/runs/$SHA/tpu/harness/push_cache_to_hf.sh
# Requires HF_TOKEN environment variable. Pushes to `and-y/rvllm-m2-build`.
```

On a fresh VM, the deploy script will auto-pull via `pull_cache_from_hf.sh`.

---

## 5. Known limits and common failures

| Symptom | Cause | Fix |
|---|---|---|
| `RESOURCE_EXHAUSTED ... 35G of 31G hbm` at compile | KV cache + weights don't fit at high B | Set `--batches 1,8` only |
| `There is no more capacity in the zone` | GCP v6e-8 supply exhausted | Retry 5-15 min; or use `SPOT=1` (different pool) |
| `/dev/shm/m2-nvfp4/` disappears between SSH sessions | Suspected TPU VM per-session tmpfs cleanup | Re-run deploy; downloads in ~2 min with auth |
| VM preempted (spot) | Google reclaimed | Delete + recreate; model re-downloads |
| `huggingface-cli: command not found` | HF 1.5+ renamed to `hf` | Deploy script accepts both; manual commands use `hf download ...` |
| `zig build` fails | Zig 0.13/0.15 API divergence in this repo | Not needed for Path A; `BUILD_ZIG=1` deploy flag if you want it |
| Gen takes forever (14s TTFT for 20-token prompt) | Prefill is serial at B=1 | See Optimization Plan (prefill batching is the fix) |

---

## 6. Option A — Use the OpenAI-compatible API server

The repo ships `tpu/harness/api_server.py` but it's hardcoded to Gemma 4. To
serve MiniMax-M2.7, create a thin wrapper that reuses the m2 load + forward:

```python
# /tmp/m2_api_server.py  — minimal single-user OpenAI-compatible server
import argparse, json, os, sys, time, uuid, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_full_bench import load_tokenizer, encode_text, decode_tokens
from m2_real_bench import load_model_stacked
from m2_synth_bench import make_mesh_v6e8, forward_step

GEN_LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n))
        messages = req["messages"]
        max_tokens = req.get("max_tokens", 512)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "
        with GEN_LOCK:
            ids = encode_text(TK, prompt)
            out_ids = self.server._generate(ids, max_tokens)
        text = decode_tokens(TK, out_ids[len(ids):])
        resp = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "minimax-m2.7-nvfp4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "length"}],
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/dev/shm/m2-nvfp4")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    mesh = make_mesh_v6e8()
    STATE = load_model_stacked(args.model_dir, mesh, max_ctx=2048, B=1, n_workers=32)
    embed, final_norm, lm_head, stacked, k_cache, v_cache, cos, sin = STATE
    TK = load_tokenizer(args.model_dir)

    @jax.jit
    def _fwd(x, stacked, k, v, pos, cos, sin, fn, lh):
        return forward_step(x, stacked, k, v, pos, cos, sin, fn, lh, mesh)

    def _generate(ids, max_new):
        nonlocal_k, nonlocal_v = k_cache, v_cache
        out = list(map(int, ids))
        for i, tid in enumerate(ids):
            x = embed[jnp.array([int(tid)], dtype=jnp.int32)]
            tok, nonlocal_k, nonlocal_v = _fwd(
                x, stacked, nonlocal_k, nonlocal_v,
                jnp.int32(i), cos, sin, final_norm, lm_head)
        out.append(int(tok[0]))
        for i in range(max_new - 1):
            x = embed[tok]
            tok, nonlocal_k, nonlocal_v = _fwd(
                x, stacked, nonlocal_k, nonlocal_v,
                jnp.int32(len(ids) + i), cos, sin, final_norm, lm_head)
            out.append(int(tok[0]))
        return out

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server._generate = _generate
    print(f"listening on :{args.port}")
    server.serve_forever()
```

Run it on the TPU VM:

```bash
python3 -u /tmp/m2_api_server.py --model-dir /dev/shm/m2-nvfp4 --port 8080 &
```

Forward port from local:

```bash
gcloud compute tpus tpu-vm ssh rvllm-m2 --zone=europe-west4-a \
  -- -L 8080:localhost:8080
```

Hit it from your laptop:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain angular momentum."}],"max_tokens":256}'
```

Single-user, single-request-at-a-time. For real multi-tenant serving you'd
need vLLM or SGLang, but those don't target TPU v6e + NVFP4 today (Apr 2026).

---

## 7. Option B — Browser chat client

A minimal single-file HTML that hits the API above. Save as `chat.html` and
open locally — it posts to `http://localhost:8080/v1/chat/completions` via the
port forward.

```html
<!doctype html>
<html>
<head>
<title>MiniMax M2.7 chat</title>
<style>
body { font: 14px/1.4 -apple-system, sans-serif; margin: 2em auto; max-width: 48em; }
#log { border: 1px solid #ccc; padding: 1em; min-height: 30em; white-space: pre-wrap; overflow-y: auto; }
.u { color: #0366d6; font-weight: bold; }
.a { color: #111; }
#in { width: 100%; box-sizing: border-box; padding: .5em; font-size: 14px; }
</style>
</head>
<body>
<h2>MiniMax M2.7-NVFP4 — TPU v6e-8</h2>
<div id="log"></div>
<p><input id="in" placeholder="type and press enter..." /></p>
<script>
const log = document.getElementById('log');
const input = document.getElementById('in');
const history = [];

input.addEventListener('keydown', async (e) => {
  if (e.key !== 'Enter') return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  history.push({role: 'user', content: text});
  log.innerHTML += `<span class="u">user:</span> ${text}\n`;
  log.scrollTop = log.scrollHeight;

  const r = await fetch('http://localhost:8080/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages: history, max_tokens: 512}),
  });
  const j = await r.json();
  const reply = j.choices[0].message.content;
  history.push({role: 'assistant', content: reply});
  log.innerHTML += `<span class="a">assistant:</span> ${reply}\n\n`;
  log.scrollTop = log.scrollHeight;
});
</script>
</body>
</html>
```

1. Run the API server on the TPU VM (Section 6).
2. Keep the gcloud SSH port-forward open.
3. Open `chat.html` in any browser. Type, press enter.

---

## 8. Option C — egui native client

The repo already has `v3/crates/rvllm-swarm-egui/` — that's the swarm
controller for GPU workloads. For a chat GUI against the TPU API, the
cleanest path is to reuse that crate's HTTP client and point it at the
m2-api-server. Minimal chat-only egui:

```bash
# From repo root
cd chat-client
cargo build --release
RVLLM_API=http://localhost:8080 cargo run --release
```

`chat-client/src/api.rs` already implements an OpenAI-style POST loop.
Set `RVLLM_API` to the forwarded port from Section 6.

---

## 9. What next (performance)

Current bottleneck: MoE dispatch via `shard_map` + `all_to_all` at small B is
dominated by kernel launch overhead per expert. Known paths that would move
the needle:

1. **Pallas NVFP4 matmul kernel** (`tpu/harness/nvfp4_matmul_pallas.py` sketch
   exists). Fuses dequant + MXU into a single TPU kernel — est 2-5× at B=1.
2. **int8 KV cache** — unblocks B ≥ 16 (currently OOMs on replicated KV).
3. **Prefill batching** — 20-token prompt currently costs 14s TTFT because
   prefill runs serial B=1. Packing the prompt into one (1, 20) forward pass
   gives ~16× TTFT speedup.
4. **Async collective flags** (`LIBTPU_INIT_ARGS`) — Gemma4 uses these for
   ~5% gain at larger batch sizes. Free to add.
5. **EAGLE-3 speculative decode** — existing `tpu/harness/eagle3_*.py` scaffold,
   requires training the draft head (~$2-5K on 8× H100).

See `tpu/harness/M2_PERF_ADVISOR_SPEC.md` for the 16-agent perf analysis.

---

## 10. Files you'll actually touch

```
tpu/harness/
├── M2_SETUP_GUIDE.md          # this file
├── deploy_m2_tpu.sh           # one-shot TPU deploy
├── pull_cache_from_hf.sh      # fast re-boot from cached env
├── push_cache_to_hf.sh        # save compile cache to HF
├── m2_full_bench.py           # main bench harness (sweep + PPL + gen)
├── m2_synth_bench.py          # forward-step impl (shared)
├── m2_real_bench.py           # real-model loader (parallel ThreadPool)
├── m2_moe.py                  # shard_map MoE (production path)
├── m2_moe_dense.py            # dense-B1 variant (compile hangs)
├── m2_moe_gather.py           # gather-top-K variant (regresses)
├── nvfp4_jax_ops.py           # pure-JAX NVFP4 matmul
├── nvfp4_matmul_pallas.py     # Pallas kernel sketch (untested)
├── nvfp4_loader.py            # safetensors NVFP4 reader
├── m2_checkpoint_schema/      # ground-truth tensor names + real_schema.md
│   ├── REAL_SCHEMA.md
│   ├── model.safetensors.index.json
│   └── tensor_names_canonical.txt
└── m2_tpu_infer.py            # single-prompt inference (CLI match Gemma4)
```

One-file entry points:
- Bench: `m2_full_bench.py`
- Single-prompt: `m2_tpu_infer.py` (same CLI as `gemma4_tpu_infer.py`)
- API: `m2_api_server.py` (copy-paste from Section 6)
