# MiniMax-M2.7-NVFP4 on TPU v6e-8 — Deployment Runbook

This is the operator runbook for deploying `lukealonso/MiniMax-M2.7-NVFP4`
(230B total / 10B active, NVFP4-quantized MoE) on a single Google Cloud
TPU v6e-8 slice using the rvllm JAX harness.

The canonical deploy script is `tpu/harness/deploy_m2_tpu.sh`. This document
explains what it does, how to prepare your environment before running it,
how much it costs, how to verify the run, and how to recover from common
failures.

Nothing here edits the model, kernels, or harness code. All code edits happen
locally and the script ships a SHA-pinned tarball to the TPU VM.

---

## 1. Prerequisites

### 1.1 Tooling (local machine)

Required CLIs on the deploy host:

- `gcloud` — authenticated against a project that has TPU v6e quota in
  `us-east5-b` (see section 1.3). Verify with `gcloud auth list`.
- `huggingface-cli` — authenticated with a token that has access to
  `lukealonso/MiniMax-M2.7-NVFP4`. Verify with `huggingface-cli whoami`.
- `git` — repo must be on a clean commit (the script refuses to deploy a
  dirty tree unless `ALLOW_DIRTY=1` is set).

### 1.2 gcloud auth

```
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT>
gcloud config set compute/zone us-east5-b
```

### 1.3 TPU v6e quota

v6e-8 is a single-host 8-chip slice (256 GB HBM total). You need quota for
**8 chips of `TPUV6E`** in `us-east5-b`. Request quota here:

- Quota page: https://console.cloud.google.com/iam-admin/quotas
- Filter: `Service = TPU API`, `Metric = TPUV6E`, `Region = us-east5`.

Confirm visibility:

```
gcloud compute tpus tpu-vm accelerator-types list \
  --zone=us-east5-b --filter='name~v6e'
```

If nothing is returned, request quota and wait for approval (usually same day).

### 1.4 HuggingFace token

```
huggingface-cli login          # paste a write-enabled or read token
# or
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

If `HF_TOKEN` is exported in your shell, `deploy_m2_tpu.sh` forwards it to the
remote `huggingface-cli download` step.

---

## 2. Cost estimate

Google Cloud list prices for TPU v6e in `us-east5` (as of 2026-04):

| SKU             | On-demand        | Spot (preemptible) |
|-----------------|------------------|--------------------|
| v6e-8 (1 slice) | **$21.60 / hr**  | **~$10 / hr**      |
| v6e-4           | $10.80 / hr      | ~$5 / hr           |
| v6e-1 (1 chip)  | $2.70 / hr       | ~$1.25 / hr        |

Full run budget for a deploy + smoke (model download dominates, ~40 min
at 130 GB / ~60 MB/s to the HF CDN):

| Stage                         | Wall time | On-demand cost |
|-------------------------------|-----------|----------------|
| VM create + boot              | 5 min     | $1.80          |
| Python/zig deps install       | 4 min     | $1.44          |
| HF model download (130 GB)    | 40 min    | $14.40         |
| zig build + tests             | 2 min     | $0.72          |
| Smoke run (16 tokens)         | 1 min     | $0.36          |
| **Total one-time bring-up**   | **~52 min** | **~$18.72**  |

Steady-state inference / benchmarking is $21.60/hr on-demand. Spot cuts that
roughly in half but can be preempted. **Always delete the VM immediately
after benchmarking** — see section 6.

---

## 3. Step-by-step

### 3.1 Local checkout

```
cd ~/rvllm
git status              # must be clean
git rev-parse HEAD      # note the SHA; this is what ships to the box
```

### 3.2 Dry-run the deploy script first

```
./tpu/harness/deploy_m2_tpu.sh --dry-run
```

This prints every command the script will run without executing anything.
Read the output and confirm PROJECT / ZONE / TPU_NAME are what you expect.

### 3.3 Real deploy

```
export PROJECT=my-gcp-project
export ZONE=us-east5-b
export TPU_NAME=rvllm-m2
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx     # optional if already logged in
./tpu/harness/deploy_m2_tpu.sh
```

The script does, in order:

1. Checks `gcloud`, `huggingface-cli`, `git` are installed.
2. Checks v6e quota visibility in the zone (warns, does not abort, if empty).
3. Creates TPU VM `rvllm-m2` with accelerator type `v6e-8`, SW version
   `v2-alpha-tpuv6e`. Skips create if the VM already exists.
4. Polls until the VM state is `READY`.
5. Creates `/workspace/runs/<SHA>` on the remote.
6. Builds a `git archive` tarball from local HEAD, uploads via
   `gcloud compute tpus tpu-vm scp`, unpacks into the run dir, writes a
   `REVISION` file with the full SHA.
7. Installs Python deps (`jax[tpu]`, `safetensors`, `huggingface_hub`,
   `tokenizers`, `ml_dtypes`, `numpy`) and zig 0.13.0.
8. Builds the zig project (`zig build -Doptimize=ReleaseFast && zig build test`).
9. Downloads `lukealonso/MiniMax-M2.7-NVFP4` to `/workspace/models/m2-nvfp4`.
10. Runs a 16-token smoke generation via `tpu/harness/m2_tpu_infer.py`.

Every step echoes the command it is about to run, prefixed with `>>`.
`set -euo pipefail` is in effect so any step failing aborts the whole
deploy — fix the root cause locally, commit, and re-run the script.

---

## 4. Troubleshooting

### `ERROR: required tool not found: gcloud`

Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install

### `ERROR: PROJECT env var is empty`

Either `export PROJECT=…` or run `gcloud config set project …` once.

### `WARNING: no v6e accelerator types visible`

You are missing quota or you picked the wrong zone. The `--dry-run`
output prints the chosen zone; cross-check the quota page from section 1.3.

### `QUOTA_EXCEEDED` on `tpus tpu-vm create`

Quota request not yet approved, or another VM of yours in the region is
already consuming v6e chips. `gcloud compute tpus tpu-vm list --zone=us-east5-b`
shows current usage.

### Create hangs in `CREATING` forever

Regional capacity pressure. Try again in 10–30 minutes or switch zone via
`ZONE=us-central2-b ./deploy_m2_tpu.sh` (note: v6e availability is limited;
check `accelerator-types list` per zone).

### `git archive` error: ambiguous HEAD

You are not in the rvllm repo root or the repo is in an unusual state
(detached / shallow). Run `git status` and `git rev-parse HEAD` to confirm.

### SSH fails with `Permission denied`

First-time SSH to a new TPU VM generates and pushes keys; it can take 30s
after `READY`. Retry the script; step-idempotency will skip the create
and jump to SSH.

### zig build fails

The zig toolchain version is pinned to 0.13.0 on the remote. If agents 2–5
require a different version, edit the `ZIG_VER=0.13.0` line in the install
heredoc in `deploy_m2_tpu.sh`.

### `huggingface-cli download` fails 401 / 403

Your HF token is missing, expired, or lacks access to the gated repo.
Export `HF_TOKEN` locally and re-run. Check access at
https://huggingface.co/lukealonso/MiniMax-M2.7-NVFP4.

### Out of disk on remote

The default boot disk at `--version=v2-alpha-tpuv6e` is usually 100 GB.
The 130 GB NVFP4 model will not fit on a 100 GB boot disk. Attach a
separate persistent data disk of 300 GB and mount at `/workspace` before
re-running. Example:

```
gcloud compute disks create rvllm-m2-data \
  --zone=us-east5-b --size=300GB --type=pd-balanced
gcloud compute tpus tpu-vm attach-disk rvllm-m2 \
  --zone=us-east5-b --disk=rvllm-m2-data --mode=read-write
# then on the VM: sudo mkfs.ext4 /dev/sdb && sudo mkdir -p /workspace \
#                 && sudo mount /dev/sdb /workspace
```

### Smoke run crashes with `could not allocate HBM`

Expected if another process is holding chips on the VM. `sudo lsof /dev/accel*`
or reboot the VM (`gcloud compute tpus tpu-vm stop … && … start …`).

### `import m2_tpu_infer` fails because agent X wrote a bad module

Fix locally, commit, redeploy. Never edit on the server — per the global
deploy protocol the remote tree is disposable.

---

## 5. Verification checklist

After `deploy_m2_tpu.sh` returns 0, confirm the following on the VM via
`gcloud compute tpus tpu-vm ssh rvllm-m2 --zone=us-east5-b`:

- [ ] `cat /workspace/runs/<SHA>/REVISION` matches your local
      `git rev-parse HEAD`.
- [ ] `du -sh /workspace/models/m2-nvfp4` reports ~130 GB.
- [ ] `ls /workspace/runs/<SHA>/zig/zig-out/lib/librvllm_zig.so` exists.
- [ ] `python3 -c "import jax; print(jax.devices())"` lists **8** TPU devices.
- [ ] The smoke step (step 9) printed a coherent completion for prompt
      `"Hello"` and at least 16 tokens of output.
- [ ] Throughput sanity on a B=1 decode smoke: target ≥ 40 tok/s at context
      2048. This is a soft bound — NVFP4 on-the-fly dequant in JAX should
      push past 60 tok/s once warm. If you see <10 tok/s, suspect the fused
      dequant is materializing; run `jax.make_jaxpr` on the forward step.
- [ ] No `WARNING: capacity overflow` messages from the MoE dispatch at
      batch sizes up to 32.
- [ ] Log lines identify the run: the startup should print the SHA from
      `REVISION` and `__file__` resolution for `m2_tpu_infer`.

Longer benchmarks are agent 16's job — see `tpu/harness/m2_bench.py`.

---

## 6. Teardown

**Important:** v6e-8 at $21.60/hr compounds fast. Tear down immediately
after the benchmarks you need.

```
gcloud compute tpus tpu-vm delete rvllm-m2 \
  --zone=us-east5-b --project=$PROJECT --quiet
```

If you created a separate data disk:

```
gcloud compute disks delete rvllm-m2-data \
  --zone=us-east5-b --project=$PROJECT --quiet
```

Double-check zero spend:

```
gcloud compute tpus tpu-vm list --zone=us-east5-b
gcloud compute disks list --filter='zone:us-east5-b AND name~rvllm'
```
