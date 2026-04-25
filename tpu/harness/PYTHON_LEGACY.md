# M2 Python Legacy Harnesses

The MiniMax-M2 Python/JAX files in this directory are legacy bring-up and
reproduction harnesses. They are no longer the runtime direction.

Use the Rust path for new M2 work:

| Need | Rust entry point |
|---|---|
| Checkpoint schema / tensor discovery | `rvllm_loader::M2CheckpointIndex` |
| Safetensors payload reads | `rvllm_loader::M2SafetensorsReader` |
| Batched prefill/decode/PPL/gen loop | `v3/crates/rvllm-xla/src/bin/m2_rust_prefill_decode.rs` |
| Decode graph / flat arena ABI | `rvllm_xla::m2_decode_graph` and `rvllm_xla::M2WeightArenaPlan` |
| NVFP4 custom-call ABI | `rvllm_fused::M2Nvfp4CustomCallAbi` |
| OpenAI-compatible API surface | `v3/crates/rvllm-serve/src/main.rs` (`rvllm-server`) |

Keep these Python files only for reference until the Rust custom-call decode
graph is fully executable on TPU:

| File | Status |
|---|---|
| `m2_full_bench.py` | Legacy JAX reproduction of measured B=8/B=16/B=32 tok/s and PPL gates. |
| `m2_synth_bench.py` | Legacy JAX forward graph; do not add new runtime behavior here. |
| `m2_real_bench.py` | Legacy Python/NumPy loader; replaced by Rust checkpoint index + safetensors reader. |
| `m2_api_server.py` | Legacy Python server; replaced by `rvllm-server`. |
| `m2_tpu_infer.py` | Legacy single-prompt smoke path. |
| `nvfp4_loader.py` | Legacy ModelOpt NVFP4 reader; replaced by `rvllm_loader`. |
| `nvfp4_jax_ops.py` / `nvfp4_matmul_pallas.py` | Correctness references for Rust/Mosaic work. |
| `m2_moe*.py`, `m2_attention.py`, `m2_kv_cache.py` | Legacy JAX graph components and benchmark references. |

Rules:

1. Do not add new M2 serving, scheduling, PPL, generation, or checkpoint-loading
   behavior in Python.
2. If a Python harness is used, label the result as legacy JAX reproduction.
3. New performance work belongs in Rust crates under `v3/crates/`.
4. Delete these files only after the Rust TPU decode path reproduces the legacy
   PPL/coherence gate and the B=8/B=16/B=32 throughput sweep.
