# CLAUDE.md — rvllm-serve

Repo-local guidance. Pair with `~/CLAUDE.md` (system-wide context:
services, profiles, brain ecosystem, model paths) and
`llm_instructions_sm121.md` (algorithmic picture of the NVFP4
pipeline). This file covers what's specific to working **inside
this tree**.

## Active branch

`rusty_sm121_vision` — the only branch you should push to.

Forked from `rusty_sm121_inference_server` (now the merge target /
fallback). The older `rusty_sm121_nvfp4` and `rusty_sm121` branches
are frozen — do not cherry-pick from them.

## Layout

```
v3/                        Cargo workspace (all binaries + libraries)
  crates/
    rvllm-core/            errors, IDs, RvllmError
    rvllm-mem/              CUDA arena (HBM + Unified for GB10)
    rvllm-cutlass/         cuBLASLt + CUTLASS SM120 FP8 GEMM bindings
    rvllm-attention/       FA2/FA3 wrappers, paged decode launchers
    rvllm-fused/           fused kernels (RoPE+pack, RMSNorm+QKV, …)
    rvllm-metadata/        config.json + tokenizer config
    rvllm-graph/           CUDA Graph capture/replay
    rvllm-sampling/        sampling + repetition penalty
    rvllm-loader/          weight loading (fp8-block, NVFP4)
    rvllm-runtime/         engine, scheduler, bring-up (Llama,
                           Qwen3-VL, Gemma4)
    rvllm-serve/           OpenAI HTTP server (axum), tokenize,
                           chat templates, vision-aware admission
kernels/                    raw .cu sources + sm_121 PTX output
  build.sh                  rebuilds PTX + manifest.json (ALWAYS
                            re-run after a kernel edit)
  build_cutlass_sm120_so.sh CUTLASS .so build (use sm_121a on GB10,
                            sm_120a on RTX 5090 / 6000 Blackwell)
  sm_121/                   per-arch PTX + manifest + libcutlass.so
```

## Build commands

```bash
cd v3 && cargo build --release --bin rvllm-server --features cuda,gb10
bash kernels/build.sh sm_121
```

**Both `cuda` and `gb10` features are required** on GB10. Without
`gb10` the bring-up takes the FA3-SM90 path and crashes with
`Fa3SoMissing`.

After every kernel edit, **re-run `kernels/build.sh sm_121`** —
otherwise the next service start picks the stale PTX. Manifest drift
between binary `RVLLM_BUILD_REVISION` and `manifest.json` is a real
class of bug; the rule is: always rebuild PTX as the LAST step
after the final commit in a chain.

## Runtime profile + service

Active profile lives at `/home/r00t/.rvllm/active-profile.env`
(symlink). Available profiles in `/home/r00t/.rvllm/profiles/`:

- `mobile-31b-rvllm.env`         — Gemma 4 31B fp8-block (default)
- `mobile-31b-rvllm-nvfp4.env`   — Gemma 4 31B NVFP4 KV
- `mobile-qwen-rvllm.env`        — Qwen 3.6 35B-A3B fp8 + vision
- `combo-*`, `creative-*`, `work-*` — Rusty mode-switch variants

Switch + restart:
```bash
sudo ln -sfn /home/r00t/.rvllm/profiles/<profile>.env \
             /home/r00t/.rvllm/active-profile.env
sudo systemctl restart rvllm-serve
```

## Native multimodal vision (Qwen3-VL + Gemma4)

Both vision towers run end-to-end as native Rust+CUDA inside this
process — no Python sidecar. `image_url` parts on
`/v1/chat/completions` go straight from the OpenAI handler through
the same process to the GPU.

Per-request flow:
1. **Admission** (`crates/rvllm-serve/src/openai/handlers.rs ::
   collect_vision_items`): fetches each `image_url` (data: URI or
   http(s)) under bounded caps —
   `RVLLM_VISION_MAX_IMAGES` (default 8),
   `RVLLM_VISION_MAX_TOTAL_BYTES` (64 MiB),
   `RVLLM_VISION_MAX_TOTAL_TOKENS` (8192). Per-fetch hard timeout
   5 s + 20 MiB cap. Literal `<|image|>` / `<|image_pad|>` markers
   in user/assistant text are rejected at admission so they cannot
   collide with the post-render token-id splice scan.
2. **Tokenize** (`crates/rvllm-serve/src/tokenize.rs ::
   render_chat_with_vision`): renders the chat template, then
   expands each image-pad token (Qwen `248056`, Gemma `258880`) to
   `vision_items[i].num_tokens` copies and emits
   `VisionSlot{token_start, num_tokens, vision_item_idx}`.
3. **GPU pre-pass** (`crates/rvllm-serve/src/cuda_worker.rs`):
   per-image `Qwen36Bringup::forward_qwen_vision` /
   `Gemma4Bringup::forward_gemma_vision`. Each loop checks
   `req.cancelled` per image. Output stays device-side
   (`VisionForwardOutput.data`).
4. **Splice** in the prefill embed step:
   `crates/rvllm-runtime/src/qwen36_bring_up.rs ::
   forward_qwen36_decode` / `gemma4_bring_up.rs ::
   run_generate` copy each output into `residual_ptr` at
   `slot.token_start * row_bytes` after `EmbeddingGatherLaunch` and
   before `F16ToBf16Launch`. Vision-bearing requests force
   `common_prefix_len = 0` and the chunked-prefill batch path
   (clean error on F16-KV).

Architecture dispatch: `VisionArch` (router.rs) is resolved at
startup from `Qwen36Arch::from_dir(model_dir)` —
`Some(_) → Qwen36`, else `Gemma4`. `forward_gemma_vision` is gated
by `#[cfg(feature = "cuda")]`; the default/mock build does not pull
in cudarc.

### Vision kernel set (sm_121, all f16)

In `kernels/`: `vit_pos_emb_lookup_2d`, `vit_pos_embed_interp`
(Qwen bilinear), `vit_rotary_2d` (Qwen cat-trick),
`vit_rotary_gemma4_2d` (Gemma per-chunk), `vit_avgpool` +
`_to_f32`, `vit_standardize` + `_f32_to_f16`, `extract_head`,
`scatter_heads`, `transpose_heads_v`, `transpose_2d`,
`gelu_tanh_mul`, `silu_mul`, `scale_inplace`,
`softmax_row_f32_to_f16`, `vector_add`, `vnorm`. cuBLASLt:
`f16_gemm_f32_batched_strided` + `bf16_gemm_f32_batched_strided`.

**INVARIANT** in the batched-strided wrappers
(`crates/rvllm-cutlass/src/cublaslt.rs`): cuBLAS internally swaps
a/b inside `cublasLtMatmul`, so the wrappers apply caller
`stride_a` to internal `layout_b` (which holds `a_*16`) and vice
versa — verified, **do not "fix" the swap**.

bf16 sibling kernels + `rmsnorm_inplace_bf16_gbf16` are committed
but NOT wired into the forward (Phase-3 attempt drifted at
`blk0_out` cos = 0.76 and was reverted; building blocks kept for
the next debug pass — see `v3/GEMMA_VISION_AUDIT.md`).

### Correctness methodology

Layer-by-layer f16 dumps via `RVLLM_QWEN36_VIT_*_DUMP` /
`RVLLM_GEMMA4_VIT_DUMP_DIR` vs HF reference dumps, compared
row-cosine. Qwen tower is byte-faithful per layer (cos = 0.9999).
Gemma is byte-faithful through block 13 and drifts to mean
cos = 0.9974 by block 26 / 0.9969 at post_projection — pure f16
compound + sqrt(1152) ≈ 33.94 saturation in the pooler. The
pooler→standardize bridge runs in f32 explicitly to recover the
saturated rows (`crates/rvllm-runtime/src/gemma4_bring_up.rs`,
commit b2969c6).

### E2E smoke (re-run after any vision-touching change)

```bash
B64=$(base64 -w0 /tmp/ball.png)
curl -s http://127.0.0.1:8010/v1/chat/completions -d '{
  "model":"<gemma-4-31b-it|qwen3-6-35b-a3b>",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'$B64'"}},
    {"type":"text","text":"Was zeigt das Bild?"}]}],
  "max_tokens":80,"temperature":0.2}'
# Qwen   → "Das Bild zeigt einen orangefarbenen Ball."
# Gemma  → "Das Bild zeigt einen orangefarbenen Kreis auf einem hellblauen Hintergrund."
```

Reserved-marker rejection (must return 400):
```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -d \
  '{"model":"...","messages":[{"role":"user","content":"hi <|image|> bye"}],"max_tokens":5}'
# → error.code = "reserved_marker_in_text"
```

## Other docs in this tree

- `llm_instructions_sm121.md` — NVFP4 + FP8 algorithmic picture
- `best_configs_sm121.md` — sweep-validated NVFP4 quality knobs
- `parameters_for_nvfp4_sm121.md` — env-knob reference
- `fp8_gemm_debug_spec.md` — per-shape FP8 GEMM correctness notes
- `v3/GEMMA4_SPEC.md` / `GEMMA4_IMPLEMENTATION.md` — Gemma 4 layer
  shapes, weight names, KV variation
- `v3/GEMMA_VISION_AUDIT.md` — layer-by-layer Gemma ViT cosine
  audit + bf16-wiring debug plan
- `v3/GB10_SPEC.md` — GB10 hardware quirks (sm_121 caveats, FA3
  unavailability)
- `CONTRIBUTING.md` — upstream workflow

## Known pitfalls

- **`cargo` cwd**: every cargo invocation must be from `v3/`, not
  the repo root. Otherwise: `could not find Cargo.toml`.
- **PTX manifest drift**: rebuild `kernels/build.sh sm_121` AFTER
  the final commit in a chain, including manifest-only commits.
- **F16-KV + vision**: vision-bearing requests force the
  chunked-prefill batch path; F16-KV is incompatible there and
  raises a clean error. Use FP8 or NVFP4 KV for vision profiles.
- **Manager-wide systemd env**: stale
  `systemctl set-environment RVLLM_*=…` entries leak into
  rvllm-serve even when unit + profile are clean. Check
  `systemctl show-environment` and `systemctl unset-environment`
  if a behaviour persists across config changes.
- **bf16 vision forward**: kernels are committed, wiring is not.
  Don't enable until the per-sub-step debug plan in
  `v3/GEMMA_VISION_AUDIT.md` is run through.
