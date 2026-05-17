# solidSF Agent Serving Runbook

This document describes the production-shaped rvLLM setup used for the
solidSF agents model endpoint. It is intentionally operational: enough detail
to rebuild, deploy, verify, and reason about the service without depending on
private chat history.

## Purpose

solidSF agents use a stock Gemma 4 31B instruction model served by rvLLM behind
an OpenAI-compatible API. The deployment goal is:

- Rust-only serving path.
- No vLLM in production.
- No Python in the request path.
- FP8 Gemma 4 31B weights on a single H100 SXM 80GB.
- 256K advertised model context.
- Greedy decode only.
- At least four simultaneous agent users admitted before backpressure.
- A clear paid-plan message when all agent seats are busy.
- A CAD harness prompt that produces real solidSF feature-tree operations,
  not fake meshes or wrapper scripts.

The public model id is:

```text
gemma4-31b-solidsf
```

The public OpenAI-compatible endpoint is:

```text
https://llm.solidsf.com/v1/chat/completions
```

## Architecture

```text
agents.solidsf.com
  -> solidSF agent backend / terminal websocket
  -> https://llm.solidsf.com
  -> rvllm-server
  -> single Gemma 4 CUDA engine owner thread
  -> H100 SXM 80GB
```

The HTTP server accepts concurrent TCP connections, but CUDA execution is owned
by one engine thread. That is deliberate. The Gemma 4 runtime currently has a
single safe engine owner; requests are admitted into a bounded in-flight set and
then serialized through the engine.

This preserves the best single-user behavior while still letting several users
wait inside the server instead of immediately seeing a capacity error.

## Production Contract

The server exposes:

```text
GET  /health
GET  /status
GET  /v1/models
POST /v1/chat/completions
```

`/status` is intentionally small and safe to expose:

```json
{
  "object": "rvllm.status",
  "model": "gemma4-31b-solidsf",
  "max_model_len": 262144,
  "max_num_seqs": 1,
  "max_inflight_requests": 4,
  "in_flight_requests": 0
}
```

`max_num_seqs=1` means rvLLM uses the single-sequence Gemma 4 generation path.
`max_inflight_requests=4` means four agent calls may be admitted at once. The
fifth live request gets a `429` OpenAI-style error.

The busy message is user-facing and intentionally commercial:

```text
All 4 agent seats are in use at the moment. Please subscribe to a paid plan to help us expand the number of people who can use agents concurrently.
```

## Decode Policy

Only greedy decoding is supported in this server path. Client requests must set:

```json
{"temperature": 0}
```

Any non-zero temperature returns a `400 invalid_request_error`:

```text
only greedy decoding is supported; set temperature to 0
```

The solidSF agent backend also strips sampling knobs such as `top_p`, `top_k`,
`min_p`, `presence_penalty`, and `frequency_penalty` before forwarding to rvLLM.

## Environment

A production service should export at least:

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/workspace/rvllm/kernels/sm_90:${LD_LIBRARY_PATH:-}"
export RUST_LOG=info

export RVLLM_MODEL_DIR=/workspace/models/gemma-4-31B-it-FP8-Dynamic
export RVLLM_KERNELS_DIR=/workspace/rvllm/kernels
export RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so
export RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so
export RVLLM_FA_FALLBACK_SO=/workspace/rvllm/kernels/sm_90/libfa_sm89_kernels.so
export RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json

export RVLLM_SERVED_MODEL_NAME=gemma4-31b-solidsf
export RVLLM_SYSTEM_PROMPT_FILE=/workspace/rvllm/v3/prompts/solidsf-agent-system.md
export RVLLM_ARENA_GB=72
export RVLLM_NUM_BLOCKS=8192
export RVLLM_BATCH_PREFILL=1
export RVLLM_MAX_INFLIGHT_REQUESTS=4
```

Run shape:

```bash
/workspace/rvllm/v3/target/release/rvllm-server \
  --host 127.0.0.1 \
  --port 8080 \
  --max-model-len 262144
```

## Systemd Shape

The service should be managed by systemd, not by an orphan shell process.

```ini
[Unit]
Description=rvLLM Gemma 4 31B FP8 OpenAI API server
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/workspace/rvllm
ExecStart=/home/ubuntu/serve_rvllm.sh
Restart=always
RestartSec=10
TimeoutStartSec=30min
TimeoutStopSec=120
KillSignal=SIGTERM
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

Use `SIGTERM` for clean restarts. A previous deployment used `SIGINT`, which
could leave an orphan rvLLM process holding port 8080 while systemd repeatedly
failed to start a second copy.

Healthy service checks:

```bash
systemctl show rvllm-server.service -p MainPID -p ActiveState -p SubState -p NRestarts
curl -fsS http://127.0.0.1:8080/status
pgrep -af 'vllm|rvllm'
```

Expected:

```text
ActiveState=active
SubState=running
NRestarts=0
```

The `pgrep` output should show `rvllm-server` and no vLLM server process.

## Build And Deploy

Build on the H100 host:

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cd /workspace/rvllm/v3
cargo build --release -p rvllm-serve --bin rvllm-server --features cuda
```

Restart only the inference service:

```bash
sudo systemctl restart rvllm-server.service
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8080/status
    break
  fi
  sleep 1
done
```

Do not terminate unrelated agents, websocket bridges, CAD services, or frontend
containers when only rvLLM changed.

## Smoke Tests

Model list:

```bash
curl -fsS https://llm.solidsf.com/v1/models
```

Simple generation:

```bash
curl -fsS https://llm.solidsf.com/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gemma4-31b-solidsf",
    "messages":[{"role":"user","content":"Reply exactly: ok"}],
    "max_tokens":8,
    "temperature":0
  }'
```

Expected assistant content:

```text
ok
```

Status:

```bash
curl -fsS https://llm.solidsf.com/status
```

Expected key fields:

```json
{
  "max_model_len": 262144,
  "max_num_seqs": 1,
  "max_inflight_requests": 4
}
```

## Concurrency Test

This script sends five simultaneous requests. Four should be admitted. One
should return `429 server_busy` with the paid-plan seat message.

```bash
node - <<'NODE'
const endpoint = 'https://llm.solidsf.com/v1/chat/completions';

async function one(i) {
  const t0 = Date.now();
  const r = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma4-31b-solidsf',
      messages: [{ role: 'user', content: `Seat test ${i}: write solidSF exactly 24 times.` }],
      max_tokens: 30,
      temperature: 0
    })
  });
  const text = await r.text();
  let json = null;
  try { json = JSON.parse(text); } catch {}
  return { i, status: r.status, ms: Date.now() - t0, error: json?.error ?? null };
}

const promises = Array.from({ length: 5 }, (_, i) => one(i + 1));
await new Promise(resolve => setTimeout(resolve, 150));
const mid = await fetch('https://llm.solidsf.com/status').then(r => r.json());
const rows = await Promise.all(promises);
console.log(JSON.stringify({ mid, rows }, null, 2));
NODE
```

Known-good behavior:

```json
{
  "mid": {
    "in_flight_requests": 4,
    "max_inflight_requests": 4
  },
  "rows": [
    {"status": 200},
    {"status": 200},
    {"status": 200},
    {"status": 200},
    {
      "status": 429,
      "error": {
        "type": "server_busy",
        "message": "All 4 agent seats are in use at the moment. Please subscribe to a paid plan to help us expand the number of people who can use agents concurrently."
      }
    }
  ]
}
```

The exact request index that gets `429` is race-dependent. The invariant is
four admitted, one rejected.

## CAD Harness Prompt

The default system prompt lives at:

```text
v3/prompts/solidsf-agent-system.md
```

For CAD mode, the prompt teaches the stock model to emit a real solidSF feature
tree through `artist_cad_replay`. The intended shape is:

```json
{
  "method": "artist_cad_replay",
  "params": {
    "source": {
      "version": 1,
      "units": "mm",
      "name": "part",
      "operations": [
        {
          "id": "base_sketch",
          "kind": "sketch",
          "host": "XY",
          "entities": [
            { "kind": "rect", "center": [0, 0], "size": [50.8, 50.8] }
          ]
        },
        {
          "id": "base",
          "kind": "extrude",
          "sketch": "base_sketch",
          "depth": 3.175
        },
        {
          "id": "hole_1",
          "kind": "hole",
          "diameter": 6.35,
          "depth": 3.175,
          "position": [19.05, 19.05, 3.175],
          "direction": [0, 0, -1]
        }
      ]
    },
    "commit": true,
    "strict": true,
    "validate": true
  }
}
```

The prompt explicitly forbids:

- `solid_*` wrapper calls.
- `solid_create_part`.
- `solid_create_sketch`.
- `solid_add_rectangle`.
- `solid_add_circle`.
- `solid_extrude`.
- `solid_extrude_cut`.
- Python, shell, mesh-only, STL-only, Three.js, or fake preview code.

Final CAD smoke target:

- response contains `artist_cad_replay`;
- response contains valid JSON inside `<tool_call>...</tool_call>`;
- no `solid_*` string appears;
- operations contain one sketch, one extrude, and four holes for the plate test;
- sketch host is `XY`;
- hole positions are 3D `[x,y,z]`.

## Performance Notes

The current production server is intentionally a single engine owner path with
bounded in-flight request admission. It does not yet batch four active users
into one multi-sequence decode kernel. The practical result:

- one user gets the normal single-request latency path;
- four users can be admitted and wait inside rvLLM;
- the fifth user gets an immediate commercial capacity message;
- no unbounded hidden queue builds up.

Longer prompts that cross the Gemma sliding-window threshold can still trigger
slower prefill paths. Keep the default system prompt compact. If CAD or harness
instructions grow, rerun the CAD smoke and check prompt token count plus TTFT in
the service logs.

## Troubleshooting

### `/health` works but `/status` is 404

An old binary is still serving. Check:

```bash
ss -ltnp | grep 8080
readlink /proc/<pid>/exe
```

If the executable path ends with `(deleted)`, kill the orphan and restart
systemd.

### systemd keeps restarting with `cuMemAlloc_v2 AllocFailed`

Usually a second rvLLM process is already holding the H100 memory. Stop the
service, kill the orphan, then start the service once.

### Non-zero temperature fails

Expected. This server path is greedy only. Send `temperature: 0`.

### CAD response drifts into wrapper scripts

Check `RVLLM_SYSTEM_PROMPT_FILE` and rerun the CAD smoke. The prompt should
mention `artist_cad_replay` and forbid `solid_*`.

### vLLM appears in process list

That is not the intended production path. Stop it and verify only
`rvllm-server` is bound to the private upstream port.
