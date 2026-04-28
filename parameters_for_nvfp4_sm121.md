# rvllm Runtime Parameters — sm_121 (GB10 / DGX Spark)

Every knob is read from the process env at startup or per-request. Values
shown as `default → recommended for our case`. "Our case" = Gemma 4 31B
(fp8-block weights), 14k–17k zeroclaw persona prompts, Lenovo PGX (sm_121).

---

## 1. KV-cache dtype dispatch

Resolution order (first truthy wins): `RVLLM_NVFP4_KV` → `RVLLM_F16_KV` →
`RVLLM_FP8_KV` → explicit `RVLLM_NVFP4_KV=0` → default NVFP4.

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_NVFP4_KV` | unset (NVFP4) | Force NVFP4 KV cache (4-bit packed + per-16-elem E4M3 microscale, ~4.5 bits/elem). | leave unset; default already NVFP4 |
| `RVLLM_FP8_KV` | unset | Force FP8 E4M3 KV cache (8 bits/elem + per-slot f32 scale). | unset (full FP8 broke WHO with Korean glitches) |
| `RVLLM_F16_KV` | unset | Force F16 KV. **Routes attention through `Fa3SoMissing` on sm_121** — broken on this hardware. | leave unset / 0 |
| `RVLLM_NVFP4_HYBRID_GLOBAL_FP8` | 0 | Force the 10 global-attention layers to FP8 KV; sliding stays NVFP4. | **0** (testing showed no win on weather; hybrid adds Korean glitches) |
| `RVLLM_NVFP4_HYBRID_SLIDING_FP8` | 0 | Force the 50 sliding-window layers to FP8 KV; globals stay NVFP4. | **0 for pure NVFP4**, 1 for fallback if NVFP4 quality regresses |
| `RVLLM_FP8_KV_LAYERS` | "" | Comma-list of layer indices forced to FP8 (e.g. `0,30,59`). Generalizes the global hybrid. | empty |

---

## 2. NVFP4 quantization quality knobs

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_NVFP4_HADAMARD` | 0 | Apply per-layer signed Hadamard rotation R = H · diag(D) to Q+K post-RoPE. Flattens outliers; Q·K^T invariant under matched rotation. | **1** |
| `RVLLM_NVFP4_HADAMARD_V` | 0 | Also rotate V (separate sign vector reused from K). Companion `hadamard_unrotate_f16` strips R^T from attn_out before O-proj. **Only works in combination with the 6-candidate MSE V-policy** (cycle 56 step 12). | **1** (under 6-candidate MSE) |
| `RVLLM_PER_TOKEN_Q_SCALE` | 0 | Compute fresh per-(token, head) Q scale at rope time → write `q_scale_cache[token,head]` → FA decoder reads same. Required when Hadamard rotates Q (rotated Q saturates static scalar). | **1** |
| `RVLLM_Q_SCALE` | 0.1 | Static fallback Q descale used when per-token-Q-scale is off. Hadamard-rotated Q saturates at 0.1; raise to 2.0. | 2.0 |
| `RVLLM_NVFP4_SCALE_POLICY` | amax6 | Default K and V scale policy: `amax6` (peak/6, range-preserving, OCP baseline) or `mse` (6-candidate search post cycle 56). Falls back to per-side env if K/V-specific not set. | mse |
| `RVLLM_NVFP4_K_SCALE_POLICY` | (fallback) | Override K policy. K affects softmax routing (Q·K^T) — range-preserving avoids miscloning attention to wrong tokens. | **amax6** |
| `RVLLM_NVFP4_V_SCALE_POLICY` | (fallback) | Override V policy. V is averaged via softmax weights; outlier-aware MSE search picks the best of 6 candidates `{peak/6, peak/4, peak/5, second/6, second/4, second/5}` (cycle 56 step 12). | **mse** |
| `RVLLM_NVFP4_STOCH_ROUND_V` | 0 | Stochastic rounding for V quantization (unbiased noise; averages out under attention sums). Empirically regressed short-context. | **0** |
| `RVLLM_NVFP4_SPLIT_KV` | 1 | Use paged_attention_v2-style split decode kernel for long context. +75% on 15k-ctx. | **1** |

---

## 3. Prefill / batching

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_BATCH_PREFILL` | 0 | Enable chunked batch prefill. Without it, prefill goes per-token (slow). | **1** |
| `RVLLM_UNIFIED_PREFILL` | 1 | Use the unified prefill kernel (one launch per chunk, MMA path). Falls back to per-qi loop when off. | **1** |
| `RVLLM_UNIFIED_PREFILL_MMA` | 1 | Route Q·K^T through the sm_121a FP8 tensor-core MMA. Off = scalar FMA reference. | **1** |
| `RVLLM_PREFILL_CHUNK_SIZE` | 0 (single-shot) | Tokens processed per prefill kernel call. Empirical hard cliff at ~288 — values 256 OK, 384+ produce garbage cycles even with our quality fixes. **Root cause not pinned** (suspected FP-accumulation-order dependence in MMA reductions). | **256** |
| `RVLLM_NUM_BLOCKS` | 1024 | KV cache logical blocks. Each block holds `RVLLM_BLOCK_SIZE` tokens × num_kv_heads × head_dim. | 1024 |
| `RVLLM_BLOCK_SIZE` | 32 | Tokens per KV block. | 32 |
| `RVLLM_ARENA_GB` | 40 | HBM arena (model weights + scratch + KV cache) in GiB. Gemma 4 31B fp8-block live-set ≈ 35 GiB. | **60** (covers max prompt + workspace) |

---

## 4. Residual chain dtype (cycle 53–55)

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_RESIDUAL_BF16` | 1 | Embedding gather widens f16 → bf16 in-place; every residual touchpoint reads/writes bf16; narrow at projection input via `bf16_to_f16_sat`. Math stays f32. Recovers 3 mantissa bits of dynamic range vs pre-cycle-54. | **1** |
| `RVLLM_BF16_NATIVE_QKV_FAST_PATH` | 0 | Experimental: skip the bf16→f16 narrow, run QKV GEMV with bf16 input directly. Long-ctx regression (WHO@17k gibberish). | **0** |
| `RVLLM_BF16_NATIVE_FULL_CHAIN` | 0 | Wholesale bf16 through all GEMMs. Cycle 55 step 19 reverted — even short-ctx breaks. | **0** |

---

## 5. FP8 GEMM dispatch

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_FP8_GEMM_CUTLASS_SM120` | 0 | Route M ≥ 128 FP8 GEMMs through CUTLASS 4.5-dev sm_120 blockwise FP8 kernel. ~10% cold-prefill win on 700-token prompts. | **1** |
| `RVLLM_FP8_DECODE_GQA` | 0 | Use the FP8 GQA decode kernel when num_q_per_kv > 1. Fp64 harness not yet ported, kept opt-in. | 0 (opt-in once validated) |
| `RVLLM_AWQ_PREFILL_LOOP` | 0 | Force AWQ M>1 prefill through the per-token GEMV loop instead of the WMMA GEMM. Debug only; production runs FP8 weights. | 0 |

---

## 6. Decode-time corrections

| Env | Default | What it does | Best for our case |
|---|---|---|---|
| `RVLLM_TOOL_CALL_OPEN_BIAS` | 0.0 | Add this f32 to logit of token 48 (`<|tool_call>` open-tag) at every decode step. Corrects cycle-53 long-ctx margin compression for tool-eligible prompts. | **4.0** |
| `RVLLM_REPETITION_PENALTY` | 1.0 | HF-convention penalty. `1.0` = off; `>1.0` divides positive logits / multiplies negative logits of recently-emitted tokens. Adds ~7 ms/step DtoH+HtoD on Gemma 4 vocab=263168. | **1.05** |
| `RVLLM_REPETITION_PENALTY_WINDOW` | 64 | Look-back window in decoded tokens. | 64 |
| `RVLLM_REPETITION_PENALTY_MIN_COUNT` | 1 | Penalize only tokens that appeared this many times in the window. `2` = avoid blunt punishment of common German function words. | **2** |
| `RVLLM_REPETITION_GUARD_N` | 20 | Abort decode when the last N decoded tokens are all the same id. | 20 |
| `RVLLM_REPETITION_CYCLE_K` | 32 | Window for the cycle-aware guard (catches multi-token attractors). | 32 |
| `RVLLM_REPETITION_CYCLE_MAX_FRAC` | 0.5 | Abort if any single id covers ≥ this fraction of the cycle window. | 0.5 |
| `RVLLM_REPETITION_CYCLE_MAX_UNIQUE` | 5 | Abort if cycle window contains ≤ this many distinct ids. | 5 |
| `RVLLM_NO_SOFTCAP` | 0 | Disable Gemma's logit softcap (`30 * tanh(logit/30)`). Diagnostic only. | 0 |

---

## 7. Diagnostics (PROD must be 0)

| Env | Default | Cost | Comment |
|---|---|---|---|
| `RVLLM_DUMP_LOGITS` | 0 | stream-fence + DtoH 1 MiB + host-sort vocab on EVERY decode step | OFF in prod |
| `RVLLM_DUMP_TOPK_LOGITS` | 0 | similar; first decode step only | OFF in prod |
| `RVLLM_DUMP_REQUEST_DIR` | unset | per-request JSON disk write | OFF in prod |
| `RVLLM_DBG_LAYER` | unset | inserts cuStreamSynchronize + per-layer DtoH probes for the first 2 layers | OFF in prod |
| `RVLLM_NVFP4_SHADOW_F16` | 0 | per-layer f16 shadow KV write + Q snapshot (~2 GiB extra arena) | OFF in prod |
| `RVLLM_DISABLE_PREFIX_CACHE` | 0 | skip prefix-cache init (every request re-prefills full prompt) | OFF in prod |

---

## 8. Server / paths

| Env | Default | Comment |
|---|---|---|
| `RVLLM_BIND` | `127.0.0.1:8080` | HTTP listener |
| `RVLLM_MODEL_DIR` | required | HF model directory (config.json, tokenizer, safetensors) |
| `RVLLM_MODEL_ID` | dirname | Advertised on `/v1/models` |
| `RVLLM_KERNELS_DIR` | required for cuda | Per-arch PTX + manifest.json |
| `RVLLM_CUTLASS_SM120_SO` | sibling of kernels_dir | CUTLASS sm_120 blockwise FP8 GEMM `.so` (built via `kernels/build_cutlass_sm120_so.sh sm_121a`) |
| `RVLLM_QUEUE_DEPTH` | 8 | Worker queue depth (admission control = 429 when full) |
| `RVLLM_MAX_TOKENS_CAP` | 4096 | Hard cap on per-request `max_tokens` |
| `RVLLM_REQUEST_TIMEOUT_SECS` | 300 | Per-request wall-clock cap. Worker can't be cancelled mid-`run_generate`; this returns 504 to the client and flips the cancel flag for next-token boundary. |

---

## Notes specific to sm_121

- **Hadamard signs are SplitMix32-finalized**, NOT FNV-1a (the latter collapses
  to a stride-2 [+1,-1,+1,-1,...] pattern).
- **CUTLASS arch suffix**: `sm_121a` (NOT `sm_120a`) on DGX Spark. Wrong suffix
  triggers `CUTE_INVALID_CONTROL_PATH` because
  `CUTLASS_ARCH_CONDITIONAL_OR_FAMILY(1210)` is false.
- **K-norm gamma is a near-scalar** (≈ 0.1221 ≈ 1/√67). Gemma 4 absorbs the
  attention-scale 1/√d into the QK-norm gamma weights at training time, which
  is why the kernel call uses `attn_scale = 1.0`.
- **Sliding window = 1024 tokens.** Sliding layers attend only to last 1024 K
  positions. Global layers see full context.
- **Per-(token, head) Q scale is required** when `RVLLM_NVFP4_HADAMARD=1`
  because rotated Q grows up to √D per channel and saturates the static scalar.
