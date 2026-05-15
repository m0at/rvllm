# rvllm-serve

OpenAI-compatible HTTP frontend over `rvllm-runtime`. Drives the
Gemma 4 31B FP8 engine on a single Hopper GPU (H100).

## Build

GPU host (Hopper, sm_90):

```bash
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-serve
```

Off-GPU compile check (no cuda):

```bash
cargo check -p rvllm-serve --manifest-path v3/Cargo.toml
```

## Run

```bash
RVLLM_MODEL_DIR=/home/ubuntu/gemma4-31b-solidsf-merged \
RVLLM_KERNELS_DIR=/workspace/runs/<sha>/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/runs/<sha>/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/runs/<sha>/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/runs/<sha>/rvllm/kernels/sm_90/policy.json \
RVLLM_PORT=8080 \
RVLLM_MAX_NUM_SEQS=4 \
RVLLM_SERVED_MODEL_NAME=gemma4-31b-solidsf \
./target/release/rvllm-server
```

## Env vars

| var | required | default | purpose |
| --- | --- | --- | --- |
| `RVLLM_MODEL_DIR` | yes | — | HF snapshot dir with `config.json`, `tokenizer.json`, `chat_template.jinja`, `*.safetensors` |
| `RVLLM_KERNELS_DIR` | yes | — | rvllm PTX kernels dir (output of `kernels/build.sh`) |
| `RVLLM_CUTLASS_SO` | SM90 only | `/dev/null` | path to `libcutlass_kernels.so` |
| `RVLLM_FA3_SO` | SM90 only | `/dev/null` | path to `libfa3_kernels.so` |
| `RVLLM_POLICY` | SM90 only | `/dev/null` | path to `policy.json` (CUTLASS autotune) |
| `RVLLM_HOST` | no | `0.0.0.0` | bind host |
| `RVLLM_PORT` | no | `8080` | bind port |
| `RVLLM_MAX_NUM_SEQS` | no | `4` | concurrency cap; excess requests queue (no 429) |
| `RVLLM_SERVED_MODEL_NAME` | no | `gemma4-31b-solidsf` | id exposed by `GET /v1/models` |
| `RVLLM_DRY_RUN` | no | unset | skip engine init; only answer `/health` + `/v1/models` |
| `RVLLM_ARENA_GB` | no | auto | HBM arena size override (auto = free - 512 MiB) |

No fallbacks: a missing required var aborts startup with a clear error.

## Endpoints

```
GET  /health                  -> 200 "ok"
GET  /v1/models               -> { "object":"list", "data":[ ... ] }
POST /v1/chat/completions     -> OpenAI-compatible (stream or non-stream)
```

## curl examples

Non-streaming:

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma4-31b-solidsf",
    "messages": [
      {"role": "user", "content": "what is the capital of France?"}
    ],
    "max_tokens": 64,
    "stream": false
  }'
```

Streaming (SSE):

```bash
curl -N -sS http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma4-31b-solidsf",
    "messages": [
      {"role": "user", "content": "write a haiku about GPUs"}
    ],
    "max_tokens": 64,
    "stream": true
  }'
```

Health:

```bash
curl -sS http://localhost:8080/health
```

## Model format

`Gemma4Bringup` accepts HuggingFace-format snapshots directly: it
parses `config.json` + the safetensors shards under `RVLLM_MODEL_DIR`.
The loader supports two layouts:

1. **BF16 weights** (`*.safetensors` with `bfloat16` dtype). The
   loader CPU-quantizes to FP8 E4M3 at startup; the clamp-rate gate
   refuses to proceed if any tensor exceeds 10 ppm clamp.
2. **Pre-quantized FP8 weights** (per-channel `*.weight_scale`
   sidecars; e.g. `RedHatAI/gemma-4-31B-it-FP8-Dynamic`). Uploaded
   directly with cuBLASLt per-channel scales — no runtime CPU quant.

For `/home/ubuntu/gemma4-31b-solidsf-merged` (the LoRA-merged tune,
BF16), the BF16 path runs at load time. No pre-quant tool is
required.

## Architecture

```
                 axum (tokio)                       std::thread
client --HTTP--> handler --mpsc:: EngineReq-->  engine worker
                  | semaphore                       (owns Gemma4Bringup,
                  | size = MAX_NUM_SEQS              CUDA stream, arena)
                  |
                  <--mpsc:: TokenEvent--   ────────── streaming generate
```

- The worker thread is a plain `std::thread` because the Gemma 4
  bringup binds CUDA modules and a stream to the calling thread —
  living on the tokio executor would let work-stealing run kernels
  against the wrong context.
- The HTTP semaphore size (`RVLLM_MAX_NUM_SEQS`) is the maximum
  number of chat completions in flight at once. Excess requests
  block in `acquire().await` and queue.
- Each request gets its own `mpsc::UnboundedSender<TokenEvent>`;
  streaming SSE simply forwards `Token`/`Finish` events.

## Smoke test (no GPU)

```bash
RVLLM_DRY_RUN=1 cargo run --release -p rvllm-serve
curl -sS http://localhost:8080/health
curl -sS http://localhost:8080/v1/models
```

`/v1/chat/completions` will still fall through to the engine and
return an error in dry-run mode — that's intentional.
