# Mistral Medium 3.5 NVFP4 sm_121 Integration Prompt

You are implementing first-class support for Mistral Medium 3.5 NVFP4 in rvllm. The target model is any configured Hugging Face model directory matching `Mistral3ForConditionalGeneration` / `mistral3`, including the NVFP4 compressed-tensors checkpoint published as `zdy1995love/Mistral-Medium-3.5-128B-NVFP4`.

This is a native Rust/CUDA implementation. Do not introduce runtime sidecars, helper services, Python-serving paths, external preprocess workers, or non-vLLM API surfaces. The server remains a vLLM-compatible OpenAI API server.

## Hard Requirements

- Support this model path only on `CompileTarget::Sm121`.
- Build and load CUDA kernels for `sm_121a`.
- Require a configured sm_121 kernel directory and a sm_121-compatible `libcutlass_sm120.so`.
- Require `CutlassBackend::SoSm120`; missing required CUTLASS symbols are startup errors.
- Keep all model, kernel, cache, and output locations configurable. Do not hardcode local paths in code, tests, or defaults.
- Keep the 128B text decoder weights packed in NVFP4. Never expand the full decoder to BF16 or FP8 for serving.
- Use native Rust for config parsing, model selection, safetensors metadata, loading, request handling, image preprocessing orchestration, scheduling, and runtime control.
- Use CUDA/CUTLASS for GPU compute. Offline Python reference tools may exist only for validation; serving must not depend on them.

## Model Selection

Add explicit model-family selection to serve and CUDA-worker configuration.

Required public configuration:

- CLI: `--model-family auto|qwen36|gemma4|mistral35`
- Env fallback: `RVLLM_MODEL_FAMILY`
- Default: `auto`

Add a typed enum:

```rust
pub enum ModelFamily {
    Auto,
    Qwen36,
    Gemma4,
    Mistral35,
}
```

Selection rules:

- If the user explicitly selects `mistral35`, require Mistral config markers and fail on mismatch.
- If the user explicitly selects `gemma4` or `qwen36`, require matching config markers and fail on mismatch.
- If `auto`, detect using model config markers and log the selected family plus the exact markers used.
- Never silently fall through to another family after explicit selection.
- `/v1/models` must report the configured served model id, not a hardcoded family label.

Expected startup usage shape:

```bash
rvllm serve --model <configured-model-dir-or-id> --model-family mistral35
```

The model path, kernel path, CUTLASS path, cache path, and served model id must all stay configurable through existing config mechanisms or explicit flags/env vars.

## Architecture Parser

Add native Rust modules following existing architecture patterns:

- `mistral35_arch`
- `mistral35_weights`
- `mistral35_load`
- `mistral35_bring_up`
- `mistral35_layer_exec`

Repo-specific edit map:

- Add the model-family enum in `v3/crates/rvllm-serve/src/config.rs` or a small shared module exported by `rvllm-serve`.
- Add the CLI flag in `v3/crates/rvllm-serve/src/main.rs` and pass it through `ServerConfig` into `CudaWorkerConfig`.
- Replace the current Qwen-vs-Gemma-only vision detection in `main.rs` with a shared resolver that returns both `ModelFamily` and `VisionArch`.
- Extend `v3/crates/rvllm-serve/src/router.rs::VisionArch` with `Mistral35`.
- Extend `v3/crates/rvllm-serve/src/openai/handlers.rs::collect_vision_items` to dispatch `predict_mistral35_num_tokens`.
- Extend `v3/crates/rvllm-serve/src/tokenize.rs::render_chat_with_vision` so image token id `10` participates in the placeholder expansion path.
- Split the current `Gemma4EnginePaths` naming into a generic path struct or add a sibling `Mistral35EnginePaths`; do not keep routing Mistral through a Gemma-named type in new code.
- Add the CUDA-worker branch in `v3/crates/rvllm-serve/src/cuda_worker.rs::spawn_cuda_worker`, using explicit `ModelFamily` selection before auto-detection.

The current `main.rs` computes `vision_arch` as Qwen or Gemma only. Replace that with one resolver used by both HTTP setup and CUDA-worker startup so the tokenizer, image-token predictor, and worker all agree on the selected family. Do not duplicate detection logic in two places.

Detect and validate:

- `architectures[0] == "Mistral3ForConditionalGeneration"`
- `model_type == "mistral3"`
- `quantization_config.format == "nvfp4-pack-quantized"`

Parse decoder config from the model's `config.json`:

- `num_hidden_layers = 88`
- `hidden_size = 12288`
- `intermediate_size = 28672`
- `num_attention_heads = 96`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `vocab_size = 131072`
- `max_position_embeddings = 262144`
- `rms_norm_eps = 1e-5`
- `hidden_act = "silu"`
- `tie_word_embeddings = false`

Parse YaRN RoPE exactly:

- `rope_theta = 1000000.0`
- `rope_type = "yarn"`
- `original_max_position_embeddings = 4096`
- `factor = 64.0`
- `beta_fast = 4.0`
- `beta_slow = 1.0`
- `mscale = 1.0`
- `mscale_all_dim = 0.0`

Fail startup if `mscale_all_dim` is not `0.0`, because the public checkpoint had a known config correction for this field.

Parse vision config:

- `model_type = "pixtral"`
- `hidden_size = 1664`
- `num_hidden_layers = 48`
- `num_attention_heads = 16`
- `head_dim = 104`
- `intermediate_size = 8192`
- `patch_size = 14`
- `image_size = 1540`
- `num_channels = 3`
- `rope_theta = 10000.0`
- top-level `image_token_index = 10`
- processor `spatial_merge_size = 2`

## Weight Loading

Implement a mixed-format loader. The Mistral checkpoint is not Gemma AWQ and not the existing Qwen/Gemma FP8 format.

Dense BF16 tensors:

- `model.language_model.embed_tokens.weight` `[131072, 12288]`
- `lm_head.weight` `[131072, 12288]`
- `model.language_model.norm.weight` `[12288]`
- all decoder norms
- all vision tower tensors
- all multi-modal projector tensors

Packed text linears:

- `<linear>.weight_packed` U8
- `<linear>.weight_scale` F8_E4M3
- `<linear>.weight_global_scale` F32 `[1]`

Validate and load every expected decoder linear:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

Expected logical shapes:

- q/o: `12288 x 12288`
- k/v: `1024 x 12288`
- gate/up: `28672 x 12288`
- down: `12288 x 28672`

Expected compressed-tensors layout:

- packed FP4 bytes store two E2M1 values per byte.
- per-16-element scales are E4M3.
- scale shape is `[N, K / 16]`.
- global scale shape is `[1]`.

Loader requirements:

- Upload packed weights and scales directly.
- Keep decoder projections in NVFP4 form.
- Do not allocate temporary full-model BF16/FP8 decoder copies.
- Use strict named errors for missing tensors, dtype mismatches, and shape mismatches.
- Print a startup memory budget summary derived from actual loaded tensor sizes and configured KV/context settings.

Implementation details from existing loaders:

- Follow the `LoadCtx` pattern in `v3/crates/rvllm-loader/src/qwen36_load.rs`: mmap all safetensors shards once, build a `BTreeMap<String, (usize, TensorEntry)>`, expose `must_get`, `bytes_of`, and typed upload helpers.
- Keep Mistral's loader independent from `load_multiformat.rs`; that path assumes FP8/F16/AWQ shapes and should not learn compressed NVFP4 as a generic fallback.
- Add `Nvfp4Weight` and `Nvfp4LinearWeight` structs rather than overloading `Fp8Weight`.
- Store device pointers as absolute device pointers, matching Qwen36's newer loader convention.
- Upload BF16 dense weights through a reusable BF16-to-F16 or BF16-preserving path. Choose one internal activation/storage convention and keep it explicit in the struct names.
- Add a `model.language_model` prefix constant for Mistral; validate it from tensor names but do not hardcode a local model directory.
- Validate tensor counts as a loader smoke check:
  - 616 decoder `weight_packed` tensors.
  - 616 decoder `weight_scale` tensors.
  - 616 decoder `weight_global_scale` tensors.
  - 434 BF16 vision tensors.
  - 4 BF16 projector tensors.
- Add shape helpers:
  - `q_rows = num_attention_heads * head_dim`
  - `kv_rows = num_key_value_heads * head_dim`
  - `packed_cols = k / 2`
  - `scale_cols = k / 16`
  - reject any `k % 16 != 0`.

## sm_121 CUTLASS/CUDA Projection Backend

Add required sm_121 NVFP4 projection support to the CUTLASS shared library and Rust loader.

Required symbols in the configured sm_121 CUTLASS library:

- NVFP4 GEMM workspace query.
- scale-factor staging size query.
- scale-factor preparation kernels.
- optimized `m=1` decode projection.
- batched prefill projection.

Production projection contract:

- Input activations are dynamically quantized to FP8 E4M3.
- Weights remain NVFP4 packed with E4M3 per-16 scales and F32 global scale.
- Compute uses Blackwell FP8 x NVFP4 CUDA/CUTLASS tensor-core path.
- Output is BF16 or F16, matching the surrounding runtime convention.
- Small-M decode and batched prefill must both use the production CUDA/CUTLASS backend from day one.

Use existing local CUDA building blocks:

- `nvfp4_utils.cuh`
- `nvfp4_mma_frag_pack.cuh`
- existing sm_121 NVFP4 KV kernels
- existing `CutlassBackend::SoSm120` dynamic loading pattern

Do not add a serving fallback that expands decoder weights or moves projections to CPU. Reference/dequant paths are allowed only behind test/probe binaries and must not be reachable in production serving.

Repo-specific CUTLASS integration:

- Extend `v3/crates/rvllm-cutlass/src/lib_so.rs::CutlassSm120Lib` with distinct NVFP4 function pointer types. Do not reuse the existing FP8 blockscale ABI; Mistral's NVFP4 weight scales are `[N, K/16]`, not the current FP8 `[N/128, K/128]` blockscale shape.
- Add symbol resolution beside the current `cutlass_fp8_gemm_blockscale_sm120` and helper symbols.
- Add Rust wrappers on `CutlassBackend` equivalent to `launch_fp8_gemm_blockscale_sm120`, but named for Mistral NVFP4 so call sites cannot pass FP8-block tensors by mistake.
- Extend `kernels/build_cutlass_sm120_so.sh` `SOURCES=(...)` to include the new NVFP4 projection source. The script already auto-detects `sm_121a`; keep that behavior.
- Keep `LoadedModule` and shared-library handles alive for every resolved kernel function. The CUDA worker comments already document that dropping a module invalidates function handles; mirror that pattern in `Mistral35Bringup`.
- Preallocate all workspaces during bring-up:
  - activation FP8 buffer for max prefill batch.
  - activation scale/SFA staging.
  - CUTLASS workspace.
  - per-layer projection scratch.
  - lm_head scratch.
- On sm_121, run `cuMemPrefetchAsync_v2` after loading weights and after allocating persistent KV regions, following the Gemma bring-up pattern.
- Startup must log whether the NVFP4 projection symbols are active. For Mistral, absence is fatal.

## Decoder Runtime

Implement `Mistral35Bringup` and `mistral35_layer_exec` as a dense decoder path.

Per-request flow:

1. Reset per-request KV/cache state.
2. Run optional vision forwards for attached images.
3. Tokenize/render prompt through existing server machinery.
4. Expand image placeholders and prepare splice metadata.
5. Prefill prompt.
6. Decode step-by-step until stop, length, or cancellation.

Per-layer decoder flow:

1. Embedding gather.
2. RMSNorm.
3. NVFP4 q/k/v projections.
4. YaRN RoPE on Q/K.
5. Fused RoPE + NVFP4 KV write.
6. NVFP4 KV attention.
7. NVFP4 output projection.
8. Residual.
9. RMSNorm.
10. NVFP4 gate/up projection.
11. SiLU.
12. NVFP4 down projection.
13. Residual.

Final flow:

1. Final RMSNorm.
2. lm_head.
3. Sampling or explicit rejection of unsupported sampling modes.

Initial serving may be greedy-only if the current runtime returns chosen tokens rather than logits. If so, non-greedy request parameters must return clear API errors. Do not silently ignore sampling parameters.

Worker integration details:

- Mirror `run_one` in `v3/crates/rvllm-serve/src/cuda_worker.rs`, but call `Mistral35Bringup::run_generate`.
- Keep per-token streaming through the same `GenerateEvent::Token` callback flow used by Gemma.
- Convert worker errors into `GenerateEvent::Error`, then let existing handlers shape API errors.
- Reuse `GenerateRequest.vision_items` and `GenerateRequest.vision_slots`; do not add a second multimodal request type.
- Make `Mistral35Bringup::run_generate` accept:
  - prompt ids.
  - max new tokens.
  - stop token ids.
  - sampling config.
  - cancellation flag.
  - token callback.
  - vision splice list.
- Use the same cancellation boundaries as Gemma: before vision, after prefill, and each decode step.
- Restore transient arena scratch after each request, but keep persistent weights, KV allocation, CUTLASS workspaces, and module handles above the scratch checkpoint.

## NVFP4 KV Attention

Mistral attention shape:

- `num_heads = 96`
- `num_kv_heads = 8`
- GQA ratio `12`
- `head_dim = 128`

Requirements:

- NVFP4 KV is mandatory by default for Mistral.
- Implement or extend NVFP4 attention so GQA ratio 12 is a first-class tested path.
- Do not reuse assumptions from lower-ratio GQA kernels without validation.
- KV layout must remain compatible with existing paged NVFP4 cache conventions:
  - packed K/V: `[blocks, block_size, kv_heads, head_dim / 2]`
  - scales: `[blocks, block_size, kv_heads, head_dim / 16]`
- Disable prefix-cache reuse for requests containing image embeddings until cache keys include image identity and splice metadata.

## Pixtral Vision

Reuse the existing OpenAI-compatible image request pipeline and prefill splice mechanism. Add a Mistral/Pixtral-specific native Rust preprocessing and CUDA forward path.

Preprocessing config:

- RGB conversion.
- resize longest edge to `1540`.
- patch size `14`.
- rescale `1 / 255`.
- mean `[0.48145466, 0.4578275, 0.40821073]`.
- std `[0.26862954, 0.26130258, 0.27577711]`.
- spatial merge `2`.

Preprocessing requirements:

- Implement host-side token-count prediction matching the actual preprocessing.
- Reject zero-token images.
- Reject images exceeding configured pixel/token limits.
- No hardcoded image dimensions beyond model config defaults.

Pixtral forward:

- patch conv `[1664, 3, 14, 14]`
- pre norm
- 48 ViT blocks
- Pixtral 2D RoPE for head dim `104`
- BF16 attention and MLP weights
- patch merger `[1664, 6656]`
- projector norm `[1664]`
- projector `linear_1 [12288, 1664]`
- GELU
- projector `linear_2 [12288, 12288]`

Output:

- `[num_image_tokens, 12288]` BF16/F16 embeddings ready for the existing prefill splice.
- Extend image placeholder handling to Mistral image token id `10`.

Repo-specific vision changes:

- Add `predict_mistral35_num_tokens(width, height)` in `v3/crates/rvllm-serve/src/openai/vision_fetch.rs`.
- Add `preprocess_mistral35_pixtral` in `v3/crates/rvllm-runtime/src/vision_preprocess.rs`, parallel to existing Qwen and Gemma functions.
- Add a `Mistral35Patches` output type containing normalized patch data, resized dimensions, patch grid, merged grid, and predicted soft-token count.
- Keep request caps wired through the existing `RVLLM_VISION_MAX_IMAGES`, `RVLLM_VISION_MAX_TOTAL_BYTES`, and `RVLLM_VISION_MAX_TOTAL_TOKENS` logic.
- Add `forward_mistral35_vision(&[u8]) -> VisionForwardOutput`, reusing `qwen36_bring_up::VisionForwardOutput` or moving that struct to a shared runtime module.
- Use existing vision kernels where shape-compatible, but add separate kernels for Pixtral-specific 2D RoPE and patch merge if Gemma/Qwen semantics differ.
- The worker must compare predicted `VisionItem.num_tokens` to `VisionForwardOutput.num_tokens` and return an error on mismatch, as the existing Qwen/Gemma paths do.

Pixtral token prediction policy:

- Resize preserving aspect ratio so longest edge is at most the configured longest edge.
- Round resized width and height to valid patch/merge multiples.
- Compute patch grid as `grid_h = resized_h / patch_size`, `grid_w = resized_w / patch_size`.
- Compute merged grid as `merged_h = ceil_or_exact(grid_h / spatial_merge_size)` and `merged_w = ceil_or_exact(grid_w / spatial_merge_size)` according to the upstream processor behavior verified by fixtures.
- `num_tokens = merged_h * merged_w`.
- Encode the exact rounding rule in tests before wiring serving.

## API Behavior

Keep the public surface vLLM-compatible:

- `/v1/models`
- `/v1/chat/completions`
- existing health/readiness endpoints if present

Request behavior:

- Accept OpenAI-compatible image content parts.
- Add Mistral `reasoning_effort` parsing:
  - allowed: `"none"`, `"high"`
  - default: `"none"`
- Stop tokens must include resolved `</s>` id `2`.
- Reject unsupported request parameters explicitly.
- Do not add unrelated public endpoints.

Specific API edits:

- Add `reasoning_effort: Option<String>` to `ChatCompletionRequest`.
- Validate it in `reject_v1_unsupported_chat` or a Mistral-specific validator:
  - allow missing, `"none"`, and `"high"` for Mistral.
  - reject other values with `invalid_request_error`.
  - reject or ignore by documented policy for non-Mistral families; explicit rejection is preferred.
- Include `reasoning_effort` in request dumps if request dumping is enabled.
- Extend reserved text marker validation to include the Mistral image marker string if it can collide with placeholder expansion.
- Extend `TokenizerHandle` with a model-family-aware image-token list rather than hardcoding Qwen/Gemma/Mistral ids inside a local closure.

## Tests

Model selection tests:

- explicit `--model-family mistral35` accepts a valid configured Mistral model directory.
- explicit `--model-family mistral35` rejects Gemma/Qwen directories.
- explicit Gemma/Qwen selections reject Mistral directories.
- `auto` logs selected family.
- `/v1/models` reports the configured served model id.

Native Rust tests:

- config detection and parsing.
- YaRN fields including `mscale_all_dim == 0.0`.
- safetensors header/tensor shape validation.
- loader rejects missing NVFP4 tensors with named errors.
- loader never allocates a full BF16/FP8 decoder copy.

CUDA/CUTLASS tests:

- configured sm_121 CUTLASS library resolves all required NVFP4 symbols.
- projection cosine for `m = {1, 2, 8, 64, 256}`.
- projection benchmarks for Mistral q/k/v/o/gate/up/down shapes.
- GQA-12 NVFP4 attention correctness.
- long-context smoke at 4k, 32k, then configured target context.

Vision tests:

- Pixtral preprocessing fixtures for square and non-square images.
- Pixtral forward substep cosine checks.
- image-text chat smoke through the OpenAI API.
- request with image splice bypasses prefix-cache reuse.

Server tests:

- text-only chat.
- image chat.
- `reasoning_effort = "none"`.
- `reasoning_effort = "high"`.
- unsupported parameters return clear errors.
- explicit cancellation returns cleanly.

Repo-targeted unit tests to add:

- `rvllm-serve` config tests:
  - CLI/env parsing for `--model-family`.
  - explicit family mismatch fails before worker startup.
- `rvllm-serve/src/router.rs` tests:
  - `VisionArch::Mistral35` selection from a minimal config fixture.
- `rvllm-serve/src/openai/vision_fetch.rs` tests:
  - Mistral token prediction for square, wide, tall, tiny, and max-edge images.
  - caps still apply with Mistral.
- `rvllm-serve/src/tokenize.rs` tests:
  - image token id `10` expands to `VisionItem.num_tokens`.
  - literal reserved marker in text is rejected before expansion.
- `rvllm-loader` tests:
  - construct a tiny safetensors-header fixture with Mistral tensor names and validate shape checking without loading real 80 GB weights.
  - ensure `Nvfp4LinearWeight` rejects scale shape `[N, K/128]` for Mistral.
- `rvllm-cutlass` tests:
  - missing NVFP4 symbols in `SoSm120` produce a typed startup error.
  - FP8 blockscale symbols alone are not considered sufficient for Mistral.

## Performance Acceptance

The implementation is complete only when:

- It starts on the configured 128 GB sm_121 machine without hardcoded paths.
- Decoder weights remain packed NVFP4 throughout serving.
- Decode and prefill use CUDA/CUTLASS production kernels.
- Mistral GQA-12 attention runs through the NVFP4 KV path.
- Text-only chat works through the OpenAI-compatible API.
- Image-text chat works through the OpenAI-compatible API.
- Startup logs include selected model family, selected model id, sm_121 target, loaded kernel directory, loaded CUTLASS library, max context, KV dtype, and memory budget.
- Bench output reports decode tokens/s, prefill tokens/s, vision latency, and memory high-water mark.
