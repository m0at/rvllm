# MiniMax-M2.7-NVFP4 on TPU v6e-8 — End-to-End Setup Guide

For a new engineer who's never touched this repo. Walk this top-to-bottom and
you'll have the exact same bench we ran, plus a chat interface and an
OpenAI-compatible API server.

---

## 0. What you're building

- **Target model**: `lukealonso/MiniMax-M2.7-NVFP4` (230B params total, 10B active,
  MoE 256 experts top-8, NVFP4 quantized, 62 layers, 3072 hidden, 196K max ctx).
- **Hardware**: 1× Google Cloud TPU v6e-8 slice (8 chips × 32 GB HBM, single host).
- **Software**: Rust PJRT/XLA runtime work under `v3/crates/`. The old
  Python/JAX harnesses are legacy reproduction/reference only.
- **Current throughput** (as of this commit):
  - Load: ~80s (parallel ThreadPool over 61 safetensors shards)
  - **B=1**: 726 ms/step = **1.4 tok/s**
  - **B=8**: 145 ms/step = **55.1 tok/s** (`RVLLM_M2_KV=int8`)
  - **B=16**: 155 ms/step = **103.5 tok/s**
  - **B=32**: 187 ms/step = **171.3 tok/s**
  - **B=64**: OOM at ctx 2048
  - Full correctness gate: **PPL 6.73** on 2047 tokens; 256-token generation
    starts coherent but repeats math text after the opening.

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
4. Builds/runs the Rust M2 planner path for the current runtime surface
5. Downloads the 130 GB model to `/dev/shm` (tmpfs, ~2 min with HF auth)
6. Leaves legacy Python/JAX smoke available only for reproducing historical
   numbers. See `PYTHON_LEGACY.md`.

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

## 4. Run the Rust M2 path

SSH to the VM:

```bash
gcloud compute tpus tpu-vm ssh rvllm-m2 --zone=europe-west4-a
```

On the VM, new M2 runtime work goes through Rust. The prefill/decode runner
plans the real M2 checkpoint, emits the Rust decode MLIR, owns the token loop,
and can execute through PJRT when built with `--features tpu` and the decode
custom calls are available.

```bash
cd $HOME/runs/$SHA/v3
cargo run --release -p rvllm-xla --bin m2_rust_prefill_decode -- \
  --model-dir /dev/shm/m2-nvfp4 \
  --batch 8 \
  --prompt-len 20 \
  --decode-steps 256 \
  --ctx 2048 \
  --kv-dtype int8 \
  --emit-decode-mlir /tmp/rvllm_m2_decode_graph.mlir
```

PJRT execution is guarded until the Mosaic NVFP4 custom calls are executable:

```bash
cargo run --release -p rvllm-xla --features tpu --bin m2_rust_prefill_decode -- \
  --model-dir /dev/shm/m2-nvfp4 \
  --batch 8 \
  --prompt-len 20 \
  --decode-steps 256 \
  --ctx 2048 \
  --kv-dtype int8 \
  --execute-prefill \
  --execute-decode \
  --max-weight-arena-bytes 160000000000 \
  --out-token-ids /tmp/m2_rust_generated_ids.txt
```

The OpenAI-compatible server is also Rust-owned:

```bash
cd $HOME/runs/$SHA/v3
cargo run --release -p rvllm-serve --bin rvllm-server -- \
  --model-dir /dev/shm/m2-nvfp4 \
  --host 0.0.0.0 \
  --port 8080 \
  --batch-size 8 \
  --prompt-len 20 \
  --decode-steps 256
```

### Legacy JAX reproduction of measured numbers

The measured B=8/B=16/B=32 numbers below came from the legacy Python/JAX harness
before the Rust graph was wired. Keep this for reproduction only; do not add new
runtime behavior here.

```bash
# All-in-one run matching our measured numbers.
# (Expect ~25 min total on first invocation — JIT compile per batch size.)
export PATH="$HOME/.local/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export M2_MOE=shardmap
# default: RVLLM_M2_MOE_IMPL=auto, which uses replicate-token MoE for B=8/16/32
export RVLLM_M2_KV=int8

cd /tmp
python3 -u m2_full_bench.py \
  --model-dir /dev/shm/m2-nvfp4 \
  --batches 1,8,16,32 \
  --ctx 2048 \
  --iters 10 \
  --warmup 3 \
  --workers 32 \
  --prompt "Explain angular momentum." \
  --gen-tokens 2048 \
  --ppl-text /tmp/wiki.txt \
  --out /tmp/m2_final.json
```

For throughput experiments, prefer the subprocess sweep wrapper. It runs each
batch in a fresh process, then runs a correctness gate that compares baseline
PPL/generation against the optimized env before writing `final.json`:

```bash
export M2_MOE=shardmap
export OPT_LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true"

MODEL_DIR=/dev/shm/m2-nvfp4 \
BATCHES="1 8 16 32" \
OUT_DIR=/tmp/m2_sweep \
LOG_DIR=/tmp/m2_sweep_logs \
bash ~/runs/$SHA/tpu/harness/run_sweep_subproc.sh
```

Gate outputs:

| File | Contents |
|---|---|
| `/tmp/m2_sweep/baseline_ppl_gen.json` | Baseline `M2_MOE=shardmap`, empty `LIBTPU_INIT_ARGS` |
| `/tmp/m2_sweep/optimized_ppl_gen.json` | Current optimized env |
| `/tmp/m2_sweep/correctness_gate.json` | PPL delta, generation-prefix match, pass/fail |
| `/tmp/m2_sweep/final.json` | Batch sweep plus optimized PPL/gen and gate result |

Default gate thresholds are conservative: PPL may not regress by more than
`max(0.10 absolute, 3% relative)`, the generated text must share the first 80
characters with baseline, and control-character floods fail the run. Override
with `PPL_ABS_TOL`, `PPL_REL_TOL`, `MIN_PREFIX_CHARS`, or set
`RUN_CORRECTNESS_GATE=0` only for exploratory timing.

Flag reference:

| Flag | Effect |
|---|---|
| `--batches 1,8,16,32` | Batch sizes to sweep (comma-separated) |
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

### MoE dispatch variants

`M2_MOE=shardmap` selects the main NVFP4 MoE implementation. Inside that path,
`RVLLM_M2_MOE_IMPL=auto` is the default:

- B=8/B=16/B=32 use exact replicate-token MoE: tokens/routing stay replicated, each
  chip skips inactive local experts with `lax.cond`, and outputs combine with `psum`.
- Other batch sizes use the original padded all-to-all path unless overridden.

Set `RVLLM_M2_MOE_IMPL=all_to_all` to force the old path, or
`RVLLM_M2_MOE_IMPL=replicate_tokens` to force the new path.

Measured B=8:

| Impl | B=8 perf | Notes |
|---|---:|---|
| `auto` / `replicate_tokens` | **55.1 tok/s** with int8 KV | exact token match vs all-to-all on B=8 probe |
| `all_to_all` | 10.0 tok/s | previous baseline |
| `RVLLM_NVFP4_BACKEND=pallas` | 7.2 tok/s | exact but slower two-stage Pallas matmul |

B=16 with `auto` + int8 KV measured **103.5 tok/s**. B=32 with replicate-token
MoE + int8 KV measured **171.3 tok/s** and is now included in `auto`. B=64
OOMs at ctx 2048 with a 992 MB allocation and 850 MB free.

### KV cache variants

Set before running:

| Value | Cache | Notes |
|---|---|---|
| `RVLLM_M2_KV=bf16` | bf16 KV | Default, simplest path |
| `RVLLM_M2_KV=int8` | int8 KV + bf16 per-vector scales | Full PPL gate 6.73; B=8 55.1 tok/s, B=32 171.3 tok/s |

### Legacy `M2_MOE` variants

Set before running:

| Value | Impl | B=1 perf | B=8+ perf |
|---|---|---|---|
| `shardmap` (default) | expert-sharded MoE (`RVLLM_M2_MOE_IMPL=auto`) | **1.4 tok/s** | **55.1 tok/s @ B=8**, **103.5 tok/s @ B=16**, **171.3 tok/s @ B=32** |
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
| Rust server returns backend unavailable | Decode custom-call bodies are guarded, not executable yet | Finish Mosaic NVFP4 custom-call implementation |

---

## 6. Option A — Use the Rust OpenAI-compatible API server

Run it on the TPU VM:

```bash
cd $HOME/runs/$SHA/v3
cargo run --release -p rvllm-serve --bin rvllm-server -- \
  --model-dir /dev/shm/m2-nvfp4 \
  --host 0.0.0.0 \
  --port 8080 \
  --batch-size 8 \
  --prompt-len 20 \
  --decode-steps 256
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
  -d '{"model":"minimax-m2.7-nvfp4","prompt_token_ids":[1,2,3,4],"max_tokens":256}'
```

Until tokenizer/chat-template support lands in Rust, pass `prompt_token_ids`
directly. The server owns request validation and the Rust M2 plan surface; it
returns a guarded backend-unavailable response until the executable decode
custom calls are present.

---

## 7. Option B — Browser smoke client

A minimal single-file HTML that hits the Rust API above. Until tokenizer support
is ported, it sends comma-separated token IDs and prints the JSON response.

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
<p><input id="in" placeholder="token ids, e.g. 1,2,3,4" /></p>
<script>
const log = document.getElementById('log');
const input = document.getElementById('in');

input.addEventListener('keydown', async (e) => {
  if (e.key !== 'Enter') return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const ids = text.split(',').map(x => Number.parseInt(x.trim(), 10)).filter(Number.isFinite);
  log.innerHTML += `<span class="u">prompt_token_ids:</span> ${ids.join(',')}\n`;
  log.scrollTop = log.scrollHeight;

  const r = await fetch('http://localhost:8080/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model: 'MiniMax-M2.7-NVFP4', prompt_token_ids: ids, max_tokens: 256}),
  });
  const j = await r.json();
  log.innerHTML += `<span class="a">server:</span> ${JSON.stringify(j, null, 2)}\n\n`;
  log.scrollTop = log.scrollHeight;
});
</script>
</body>
</html>
```

1. Run the API server on the TPU VM (Section 6).
2. Keep the gcloud SSH port-forward open.
3. Open `chat.html` in any browser. Type comma-separated token IDs, press enter.

---

## 8. Option C — egui native client

The repo already has `v3/crates/rvllm-swarm-egui/` — that's the swarm
controller for GPU workloads. For a chat GUI against the TPU API, reuse that
crate's HTTP client and point it at the Rust server from Section 6. Minimal
chat-only egui:

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

Current bottleneck: small-batch MoE still launches per-active-expert NVFP4
matmuls. B=8/B=16/B=32 avoid padded all-to-all and empty expert matmuls now,
but B=1 still needs a better kernel.
Known paths that would move the needle:

1. **Mosaic/MLIR fused NVFP4 matmul kernel**. The Rust custom-call ABI is
   pinned; the next step is the executable TPU body.
2. **Tokenizer/chat-template port** into Rust so `/v1/chat/completions` accepts
   normal OpenAI `messages` instead of low-level `prompt_token_ids`.
3. **Longer Rust PPL/coherence gate** once decode custom calls execute.
4. **Async collective flags** (`LIBTPU_INIT_ARGS`) — Gemma4 uses these for
   ~5% gain at larger batch sizes. Free to add.
5. **EAGLE-3 speculative decode** — port the existing scaffold to Rust after
   target-model decode is executable.

See `tpu/harness/M2_PERF_ADVISOR_SPEC.md` for the 16-agent perf analysis.
The full run JSONs are checked in under `tpu/out/m2/full_equiv_bb800cc21/`.

---

## 10. Files you'll actually touch

```
tpu/harness/
├── M2_SETUP_GUIDE.md          # this file
├── PYTHON_LEGACY.md           # legacy Python/JAX quarantine map
├── deploy_m2_tpu.sh           # one-shot TPU deploy
├── pull_cache_from_hf.sh      # fast re-boot from cached env
├── push_cache_to_hf.sh        # save compile cache to HF
├── m2_checkpoint_schema/      # ground-truth tensor names + real_schema.md
│   ├── REAL_SCHEMA.md
│   ├── model.safetensors.index.json
│   └── tensor_names_canonical.txt
└── *.py                       # legacy JAX reproduction/reference only
```

Rust entry points:
- Bench/PPL/gen loop: `v3/crates/rvllm-xla/src/bin/m2_rust_prefill_decode.rs`
- API: `v3/crates/rvllm-serve/src/main.rs` (`rvllm-server`)
- Fused NVFP4 ABI: `v3/crates/rvllm-fused/src/m2_nvfp4.rs`
