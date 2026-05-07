# Mistral 3.5 integration status

Snapshot for the `rusty_sm121_mistral` branch after **25 incremental
loop iterations**. The integration spec lives in
`mistral-35-integration.md` (repo root); this file tracks what's
landed in code vs what still needs the GPU/CUTLASS work.

## What's landed (Rust + design + math + harness, fully tested)

### Steps 1-10 (initial scaffolding pass — iterations 1-7)

| Step | Deliverable | Tests | Code |
|---|---|---|---|
| 1 | `ModelFamily {Auto, Qwen36, Gemma4, Mistral35}` enum, `--model-family` CLI flag, env `RVLLM_MODEL_FAMILY`, **shared resolver** in `rvllm-serve/src/family.rs` (replaces the duplicate Qwen-vs-Gemma probe). Explicit selection asserts match — no silent fall-through. | 6 | family.rs |
| 2 | `Mistral35Arch::from_dir` in `rvllm-loader/src/mistral35_arch.rs`. Detects the marker triple, parses the dense-decoder + YaRN + Pixtral blocks, **fails on `mscale_all_dim != 0.0`** per the known checkpoint correction, asserts `silu` activation + integer GQA ratio. | 8 | mistral35_arch.rs |
| 3 | `Nvfp4LinearWeight` + `Nvfp4LinearShape` + `Mistral35WeightInventory` in `rvllm-loader/src/mistral35_weights.rs`. Walks the safetensors index header-only, validates every NVFP4 linear's dtype + shape (`weight_packed U8 [N,K/2]`, `weight_scale Fp8E4M3 [N, K/16]`, `weight_global_scale F32 [1]`), counts the 88×7 = 616 packed/scale/global + 434 vision BF16 + 4 projector BF16 tensors expected by the spec. | 7 | mistral35_weights.rs |
| 4 (Rust ABI scaffolding) | NVFP4 fn-pointer types + `CutlassSm120Lib`/`CutlassBackend` extension with `nvfp4_active`, `require_nvfp4`, `launch_nvfp4_gemm`, `launch_nvfp4_prep_sfa`. Distinct ABI from FP8-blockscale; m=1 decode kernel auto-selected when present. | 3 | rvllm-cutlass/src/lib_so.rs |
| 5 (bring-up scaffolding) | `Mistral35Bringup::load`. End-to-end startup chain: parse arch → walk safetensors index → `validate_mistral35_inventory` → `CutlassBackend::require_nvfp4`. `cuda_worker.rs::spawn_cuda_worker` dispatches `ModelFamily::Mistral35` here; per-request error is typed and references the missing kernel. | 3 | mistral35_bring_up.rs |
| 6 | `KvDecodeStrategy` gate documenting that Mistral's GQA=12 exceeds the existing NVFP4-KV decode kernel caps (`MAX_GQA_DECODE=4`, `MAX_GQA_SPLIT=8`) → routes to `PerHeadFallback`. Strategy logged at startup with the concrete kernel-side fix path. | 4 | mistral35_bring_up.rs |
| 7 (host preprocess) | `Mistral35PreprocessConfig` + `Mistral35Patches` + `preprocess_mistral35_pixtral` in `rvllm-runtime/src/vision_preprocess.rs`. CLIP-style normalisation, longest-edge=1540, factor-28 rounding, tiny-image snap-up. **Predictor in `vision_fetch.rs::predict_mistral35_num_tokens` matches byte-for-byte** so the worker's predicted-vs-actual num_tokens check stays satisfied. | 12 | vision_preprocess.rs, vision_fetch.rs |
| 8 | `reasoning_effort: Option<String>` on `ChatCompletionRequest`, validated in `reject_v1_unsupported_chat` (Mistral accepts `none`/`high`; other families reject with a clear 400). `collect_vision_items` dispatches `predict_mistral35_num_tokens` for Mistral. `TokenizerHandle::render_chat_with_vision` is **VisionArch-aware** — Mistral's `image_token_index=10` is gated per family because `10` is `\n` in many other tokenisers. | (covered above) | chat.rs, handlers.rs, tokenize.rs |
| 9 (plan + gates) | `v3/MISTRAL35_BATCHED_PREFILL_PLAN.md` mapping Qwen phases 4b/5/6/7/8 onto Mistral's dense NVFP4 decoder. `BatchedPrefillConfig` + 5 env gates + `outer_loop_deleted` derived flag. | 2 | mistral35_bring_up.rs |
| 10 | Workspace-wide build + test sweep green under default features and `cuda,gb10`. DAG invariant test fixed (rvllm-serve reaches Mistral arch through rvllm-runtime re-export, not a direct rvllm-loader edge). Profile template at `v3/MISTRAL35_PROFILE_TEMPLATE.env`. | (full sweep) | this doc |

### Step 4 deepening — CUDA-side ABI + standalone-compiled .cu (iterations 8-11)

Each `.cu` file below standalone-compiles against CUTLASS 4.4.2 for
`sm_121a` via `nvcc 13.x -O3 -arch=sm_121a -std=c++17
--expt-relaxed-constexpr`. The sources are intentionally **not yet
listed** in `kernels/build_cutlass_sm120_so.sh::SOURCES` so the
running `libcutlass_sm120.so` stays byte-identical for Gemma 4 /
Qwen 3.6; appending all three to `SOURCES=` and rebuilding flips
`CutlassBackend::nvfp4_active()` to true at startup.

| Deliverable | Code | Notes |
|---|---|---|
| NVFP4 × NVFP4 → BF16 GEMM (port of CUTLASS example 79a) | `kernels/cutlass_nvfp4_gemm_sm120.cu` | Block-scaled tensor-core MMA on Sm120; global F32 scale folded into epilogue alpha; m=1 decode-specialised symbol present as alias today. |
| Activation prep: BF16/F16 → NVFP4-packed + natural-layout E4M3 SFA | `kernels/cutlass_nvfp4_prep_act_sm120.cu` | Block-wide `__shfl_xor` absmax over 16-lane subwarp; `peak/6` → E4M3 scale; lanes 0..7 each emit one packed byte. |
| SFA layout transform: natural row-major `[m, K/16]` → CUTLASS Sm120 interleaved | inside `cutlass_nvfp4_gemm_sm120.cu` | Pure index permutation through the cute Layout produced by `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA`. |
| Rust chain ABI | `rvllm-cutlass/src/lib_so.rs` | New fn-pointer types `Nvfp4Sm120PrepActFn` / `Nvfp4Sm120SfaTransformFn` / `Nvfp4Sm120NaturalSfaBytesFn`, dispatchers on `CutlassSm120Lib` + `CutlassBackend`, plus `launch_nvfp4_prep_sfa_chain` convenience that runs prep_act → sfa_transform back-to-back. `nvfp4_active` / `require_nvfp4` gate on the chain (3 + 6 → 3 = 6 required symbols). |

### Step 5 deepening — pure-Rust math reference + CI determinism (iterations 12-22)

Every per-layer step has a pure-Rust deterministic reference. The
future CUDA forward will diff against these via the cosine harness
below.

| Deliverable | Code | Tests |
|---|---|---|
| `LayerScratchBudget::natural` + backend-aware `with_backend` | mistral35_bring_up.rs | 3 |
| `CutlassBackend` retained on `Mistral35Bringup` so the forward path doesn't re-open the .so per request | mistral35_bring_up.rs | (covered above) |
| YaRN math: `yarn_inv_freq`, `yarn_mscale`, correction-range, ramp interp/extrap blend | mistral35_yarn.rs | 10 |
| YaRN cos/sin precompute: `YarnRopeTables` + `build_yarn_rope_tables` (mscale folded in) + `Mistral35Bringup::build_rope_tables` | mistral35_yarn.rs / mistral35_bring_up.rs | 5 |
| RoPE apply: `apply_rope_pair` (canonical Llama/Mistral half-split rotation), `apply_rope_at` indexed by `pos` | mistral35_yarn.rs | 5 |
| Per-layer math refs: `silu`, `silu_inplace`, `silu_mul`, `rms_norm`, `softmax_row` | mistral35_math.rs | 12 |
| Single-token decoder layer: `mistral_layer_step` composes RMSNorm → Q/K/V f32 matmul → YaRN-RoPE → KV append → GQA-aware attention → O proj → residual → post-attn-RMSNorm → gate/up → silu_mul → down → residual. f32 weights (separate from production NVFP4) so cosine validation runs against a deterministic baseline. | mistral35_layer_ref.rs | 4 |
| Probe binary: `probe-mistral35-layer-ref` exercises the layer ref on a synthetic seed and writes per-position f32 LE dumps + final KV cache + checksums summary. **Byte-deterministic across runs** (verified). | bin/probe_mistral35_layer_ref.rs | (verified by integration test below) |
| Cosine harness: `tools/cmp_mistral35_layer_ref.py` walks every `*.bin`, loads as f32 LE, reports per-file cosine + max-abs-error, exits 1 with first-drift bisect pointer when below threshold. | tools/cmp_mistral35_layer_ref.py | (drift-detection live verified) |
| CI determinism gate: `tests/mistral35_layer_ref_determinism.rs` runs the layer ref twice with the same seed + byte-compares; runs the probe binary twice + byte-compares; asserts different seeds yield different outputs. | tests/mistral35_layer_ref_determinism.rs | 3 |

### Workspace test counts (post iteration 22)

```
cargo test --workspace --features cuda,gb10
→ 41 test binaries / 450 passed / 0 failed
```

Net new tests landed by this branch: **~110**, distributed across
arch parsing, weight inventory, Pixtral host preprocess, NVFP4
CUTLASS ABI (chain), family resolver, KV decode strategy, batched-
prefill config, bring-up validation, scratch budgets, YaRN math
(inv_freq + mscale + tables + apply_rope), per-layer math (silu /
rms_norm / softmax), single-token layer composition, probe binary
determinism, and `reasoning_effort` API surface.

## What still needs GPU work

These remain the genuine CUDA / CUTLASS engineering pieces that
need hardware iteration cycles a 25-min dynamic loop cannot safely
do without continuous validation:

### A. Wire the .cu sources into `kernels/build_cutlass_sm120_so.sh::SOURCES`

Three files standalone-compile already:

```
kernels/cutlass_nvfp4_gemm_sm120.cu     (GEMM + SFA-transform, 226 KB .o)
kernels/cutlass_nvfp4_prep_act_sm120.cu (activation prep,         36 KB .o)
```

Append both to the build script's `SOURCES=(...)`, rebuild via
`bash kernels/build_cutlass_sm120_so.sh sm_121a`, run the existing
Gemma 4 / Qwen 3.6 cosine smokes to confirm no regression, then
verify Mistral startup gets past `require_nvfp4`.

### B. Mistral35Bringup CUDA forward path (Step 5 GPU half)

Mirrors `Gemma4Bringup` (~7000 LOC) and `Qwen36Bringup` (~9000
LOC) but for Mistral's dense NVFP4 decoder. Sub-deliverables:

- HBM arena layout for 128B NVFP4 weights + paged NVFP4 KV (block
  size 32, ~1024 blocks) + activations + scratch + workspace.
- Per-layer forward implementation. The CPU reference in
  `mistral35_layer_ref.rs` documents the exact step ordering;
  port it to CUDA using the existing rvllm-fused / rvllm-attention
  kernels for RMSNorm + SiLU + softmax + FA2 NVFP4 prefill, plus
  the new NVFP4-CUTLASS chain for the seven projections.
- Greedy-only sampling at first; non-greedy returns clean API
  errors per the spec.
- Cancellation boundaries before vision, after prefill, every
  decode step (mirror Gemma).
- Round-26 / 27 stream-fence invariant: populate `pos_cl_region`
  / `context_lens` / `positions` via on-stream kernel, never
  `Region::copy_from_host`.
- `RVLLM_MISTRAL35_LAYER_DUMP_DIR=...` instrumentation to write
  per-(layer, position) hidden states matching the
  `cmp_mistral35_layer_ref.py` file naming convention.

### C. YaRN RoPE + NVFP4 KV write fused kernel

`kernels/fused_yarn_rope_nvfp4kv_mistral.cu`. ~80% of the existing
`fused_rope_partial_nvfp4kv` body, only the angle table changes
to YaRN's per-frequency interp/extrap blend. The pure-Rust
`mistral35_yarn::yarn_inv_freq` is the byte-comparable reference;
the CPU `apply_rope_pair` is the per-element correctness baseline.

### D. Raise NVFP4 KV decode GQA caps (Step 6 GPU half)

In `kernels/flash_attention_nvfp4kv*.cu` and
`flash_attention_split_decode_nvfp4kv*.cu`: bump
`MAX_GQA_DECODE` (4 → 16) and `MAX_GQA_SPLIT` (8 → 16). Register
impact bounded; covered in `MISTRAL35_BATCHED_PREFILL_PLAN.md §
Kernel reuse map`. Gemma 4 + Qwen 3.6 cosine MUST stay
byte-identical post-bump.

### E. Pixtral vision GPU forward (Step 7 GPU half)

`forward_mistral35_vision`: 48-layer ViT (head_dim=104, patch=14,
2D RoPE, BF16 attn+MLP) → patch-merger `[1664, 6656]` → projector
`linear_1 [12288, 1664]` → GELU → `linear_2 [12288, 12288]`.
Reuses the existing splice mechanism (Qwen/Gemma path) with
Mistral's image-token id 10. The host-side preprocess
(`preprocess_mistral35_pixtral`) is already in place.

## Branch-shape summary

25 commits on `rusty_sm121_mistral`, each gated by build-green +
tests-green + zero regressions on Gemma 4 / Qwen 3.6 paths. Every
Rust-callable surface for Mistral 3.5 is in place and CI-locked
(determinism integration test). The CPU baseline + cosine
validation harness is operational. The GPU work-stream remaining
is concentrated in five well-bounded kernel-implementation tasks
above; each has its CPU-reference numpy diff already wired so the
implementation cycle is predictable.
