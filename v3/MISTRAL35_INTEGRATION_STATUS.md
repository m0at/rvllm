# Mistral 3.5 integration status

Snapshot for the `rusty_sm121_mistral` branch, after 10 incremental
loop iterations of scaffolding work. The integration spec lives in
`mistral-35-integration.md` (repo root); this file tracks what's
landed in code vs what still needs the GPU/CUTLASS work.

## What's landed (Rust + design, fully tested)

| Step | Deliverable | Tests | Code |
|---|---|---|---|
| 1 | `ModelFamily {Auto, Qwen36, Gemma4, Mistral35}` enum, `--model-family` CLI flag, env `RVLLM_MODEL_FAMILY`, **shared resolver** in `rvllm-serve/src/family.rs` (replaces the duplicate Qwen-vs-Gemma probe). Explicit selection asserts match — no silent fall-through. | 6 | family.rs |
| 2 | `Mistral35Arch::from_dir` in `rvllm-loader/src/mistral35_arch.rs`. Detects the marker triple, parses the dense-decoder + YaRN + Pixtral blocks, **fails on `mscale_all_dim != 0.0`** per the known checkpoint correction, asserts `silu` activation + integer GQA ratio. | 8 | mistral35_arch.rs |
| 3 (Rust) | `Nvfp4LinearWeight` + `Nvfp4LinearShape` + `Mistral35WeightInventory` in `rvllm-loader/src/mistral35_weights.rs`. Walks the safetensors index header-only, validates every NVFP4 linear's dtype + shape (`weight_packed U8 [N,K/2]`, `weight_scale Fp8E4M3 [N, K/16]`, `weight_global_scale F32 [1]`), counts the 88×7 = 616 packed/scale/global + 434 vision BF16 + 4 projector BF16 tensors expected by the spec. | 7 | mistral35_weights.rs |
| 4 (Rust ABI) | NVFP4 fn-pointer types (`Nvfp4GemmSm120Fn`, `Nvfp4GemmSm120WorkspaceFn`, `Nvfp4Sm120SfaBytesFn`, `Nvfp4Sm120PrepSfaFn`) + `CutlassSm120Lib`/`CutlassBackend` extension with `nvfp4_active`, `require_nvfp4`, `launch_nvfp4_gemm`, `launch_nvfp4_prep_sfa`. Distinct ABI from FP8-blockscale; m=1 decode kernel auto-selected when present. | 3 | rvllm-cutlass/src/lib_so.rs |
| 5 (scaffold) | `Mistral35Bringup::load` in `rvllm-runtime/src/mistral35_bring_up.rs`. End-to-end startup chain: parse arch → walk safetensors index → `validate_mistral35_inventory` → `CutlassBackend::require_nvfp4`. `cuda_worker.rs::spawn_cuda_worker` dispatches `ModelFamily::Mistral35` here; per-request error is typed and references the missing kernel. | 3 | mistral35_bring_up.rs |
| 6 | `KvDecodeStrategy` gate documenting that Mistral's GQA=12 exceeds the existing NVFP4-KV decode kernel caps (`MAX_GQA_DECODE=4`, `MAX_GQA_SPLIT=8`) → routes to `PerHeadFallback`. Strategy logged at startup with the concrete kernel-side fix path. | 4 | mistral35_bring_up.rs |
| 7 (host preprocess) | `Mistral35PreprocessConfig` + `Mistral35Patches` + `preprocess_mistral35_pixtral` in `rvllm-runtime/src/vision_preprocess.rs`. CLIP-style normalisation, longest-edge=1540, factor-28 rounding, tiny-image snap-up. **Predictor in `vision_fetch.rs::predict_mistral35_num_tokens` matches byte-for-byte** so the worker's predicted-vs-actual num_tokens check stays satisfied. | 12 | vision_preprocess.rs, vision_fetch.rs |
| 8 | `reasoning_effort: Option<String>` on `ChatCompletionRequest`, validated in `reject_v1_unsupported_chat` (Mistral accepts `none`/`high`; other families reject with a clear 400). `collect_vision_items` dispatches `predict_mistral35_num_tokens` for Mistral. `TokenizerHandle::render_chat_with_vision` is **VisionArch-aware** — Mistral's `image_token_index=10` is gated per family because `10` is `\n` in many other tokenisers. | (covered above) | chat.rs, handlers.rs, tokenize.rs |
| 9 (plan + gates) | `v3/MISTRAL35_BATCHED_PREFILL_PLAN.md` mapping Qwen phases 4b/5/6/7/8 onto Mistral's dense NVFP4 decoder (no MoE, no linear-attn, only 2 truly Mistral-specific kernels needed: NVFP4 GEMM + YaRN-RoPE-and-NVFP4-KV-write). `BatchedPrefillConfig` + 5 env gates + `outer_loop_deleted` derived flag. | 2 | mistral35_bring_up.rs |
| 10 | Workspace-wide build + test sweep green under default features and `cuda,gb10`. DAG invariant test fixed (rvllm-serve reaches Mistral arch through rvllm-runtime re-export, not a direct rvllm-loader edge). Profile template at `v3/MISTRAL35_PROFILE_TEMPLATE.env`. | (full sweep) | this doc |

## What still needs GPU work

The remaining work is genuine CUDA / CUTLASS engineering that needs
hardware iteration to validate — outside the safe scope of a blind
dynamic-loop without GPU runs:

### A. CUTLASS NVFP4 GEMM kernel (Step 4 GPU half)

`kernels/cutlass_nvfp4_gemm_sm120.cu` is a documented skeleton.
The implementation needs:

1. CUTLASS 4.4.2 `CollectiveBuilder<Sm120, ...>` parameterised for
   FP8 × NVFP4 → BF16/F16, with the activation FP8 prep helper.
2. m=1 specialisation for decode — the prefill kernel handles m=1
   correctly, the specialised path is a perf optimisation only.
3. Shape coverage for every Mistral linear:
   `q/o: 12288×12288`, `k/v: 1024×12288`, `gate/up: 28672×12288`,
   `down: 12288×28672`. Each gets a cosine + benchmark gate.
4. Scale-prep kernel (`cutlass_nvfp4_gemm_sm120_prep_sfa`) that
   per-token-quantises BF16/F16 activations to FP8 E4M3 and stages
   the corresponding `[m, K/16]` SFA scratch.

Once the source is in, append it to `SOURCES=` in
`kernels/build_cutlass_sm120_so.sh`, rebuild, and Mistral
`require_nvfp4()` flips green.

### B. Mistral35Bringup CUDA forward path (Step 5 GPU half)

Mirrors `Gemma4Bringup` (~7000 LOC) and `Qwen36Bringup` (~9000 LOC)
but for the dense NVFP4 decoder:

- HBM arena layout: 128B NVFP4 weights + paged NVFP4 KV (block_size
  32, ~1024 blocks) + activations + scratch + workspace.
- Per-layer forward: RMSNorm → q/k/v NVFP4 GEMM → YaRN-RoPE +
  NVFP4-KV-write → FA2 NVFP4 prefill → o NVFP4 GEMM → residual →
  RMSNorm → gate/up NVFP4 GEMM → SiLU → down NVFP4 GEMM → residual.
- Greedy-only sampling at first; non-greedy returns clean API
  errors per the spec.
- Cancellation boundaries before vision, after prefill, every
  decode step (mirror Gemma).
- Round-26 / 27 stream-fence invariant ported forward — populate
  `pos_cl_region` / `context_lens` / `positions` via on-stream
  kernel, never `Region::copy_from_host`.

### C. YaRN RoPE + NVFP4 KV write kernel (Step 5 helper)

`kernels/fused_yarn_rope_nvfp4kv_mistral.cu`: ~80% of the existing
`fused_rope_partial_nvfp4kv` body, only the angle table changes
to YaRN's `mscale × ramp(beta_fast, beta_slow)` per-frequency
correction (`original_max=4096`, `factor=64`, `mscale_all_dim=0.0`).

### D. Raise NVFP4 KV decode GQA caps (Step 6 GPU half)

In `kernels/flash_attention_nvfp4kv*.cu` and
`flash_attention_split_decode_nvfp4kv*.cu`: bump
`MAX_GQA_DECODE` (4 → 16) and `MAX_GQA_SPLIT` (8 → 16). Register
impact bounded; covered in `MISTRAL35_BATCHED_PREFILL_PLAN.md §
Kernel reuse map`. Must validate Gemma 4 + Qwen 3.6 cosine stays
byte-identical post-bump.

### E. Pixtral vision GPU forward (Step 7 GPU half)

`forward_mistral35_vision`: 48-layer ViT (head_dim=104, patch=14,
2D RoPE, BF16 attn+MLP) → patch-merger `[1664, 6656]` → projector
`linear_1 [12288, 1664]` → GELU → `linear_2 [12288, 12288]`.
Reuses the existing splice mechanism (Qwen/Gemma path) with
Mistral's image-token id 10.

## Workspace test counts (post-Step-10)

```
cargo test --workspace --features cuda,gb10
→ 33 test binaries, 404 passed, 0 failed
```

Across:
- `rvllm-cutlass` 12 (`--features cuda`)
- `rvllm-loader` 64
- `rvllm-runtime` 58
- `rvllm-serve` 131 lib + 29 integration
- plus `rvllm-core` / `rvllm-mem` / `rvllm-kernels` / `rvllm-fused`
  / `rvllm-attention` / `rvllm-graph` / `rvllm-metadata` /
  `rvllm-sampling` / `rvllm-invariants` test crates unchanged.

Net new tests landed by this branch: **~40**, distributed across
arch parsing, weight inventory, Pixtral host preprocess, NVFP4
CUTLASS ABI, family resolver, KV decode strategy, batched-prefill
config, bring-up validation, and reasoning_effort API surface.

## Branch-shape summary

10 commits on `rusty_sm121_mistral`, each gated by build-green +
tests-green + zero regressions on Gemma 4 / Qwen 3.6 paths.
Rust-side ABI for every Mistral 3.5 surface is in place; the GPU
work (CUTLASS kernel + bring-up forward + GQA cap raise +
Pixtral forward) is the remaining work-stream for an iteration
with deliberate GPU validation cycles.
