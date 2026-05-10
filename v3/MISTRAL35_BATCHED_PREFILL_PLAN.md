# Mistral 3.5 batched-prefill plan

**Status:** design / scaffolding (Step 9-partial of `mistral-35-integration.md`).
**Goal:** lay out which prefill phases of the Qwen 3.6 batched plan
(`v3/QWEN_BATCHED_PREFILL_PLAN.md`) port directly to Mistral 3.5
and which need new kernels, so when the GPU forward path
(`Mistral35Bringup::run_generate`) is implemented, the batched-from-
day-one invariant survives.

## Mistral 3.5 vs Qwen 3.6 — structural diff

| Aspect | Qwen 3.6 35B-A3B | Mistral 3.5 128B |
|---|---|---|
| Layers | 40 (30 linear + 10 full) | 88 (all dense full-attn) |
| Attention | hybrid Gated-DeltaNet + full | full only |
| MoE | 256 experts, top-8, shared expert | none — dense gate/up/down MLP |
| GQA ratio | 8 | **12** (96 / 8 KV) |
| Weight format | FP8 fp8-block `[N/128, K/128]` | **NVFP4** `[N, K/16]` (per-row × per-16-K) |
| KV cache | NVFP4 KV (fp8 fallback) | **NVFP4 KV mandatory** |
| RoPE | full rotary, theta=1e7 | **YaRN**, theta=1e6, factor=64, orig_max=4096 |
| Vision | Qwen3VL ViT | Pixtral 48L hd=104 |

Headline: Mistral is structurally **simpler** than Qwen (no MoE,
no linear attention, no hybrid-pattern bookkeeping) but uses a
**different weight format** at every projection. The Qwen plan's
phases 4b (linear-attn batching) and 6 (MoE / shared-expert
batching) drop. Phases 5 (full-attn prefill), 7 (outer-loop
deletion), and 8 (CUDA Graph capture) port directly.

## Per-layer prefill flow

For an `[N, D]` prompt buffer (`D = hidden_size = 12288`), one
batched layer is a strict 13-step pipeline:

```
 0. RMSNorm(input)                                    → 1 launch
 1. Q/K/V projections (NVFP4 GEMM, M=N)               → 3 launches
                                                       (or 1 if fused QKV)
 2. YaRN RoPE on Q + fused RoPE-and-NVFP4-KV-write    → 2 launches
                                                       (Q rotated in-place,
                                                        K rotated + NVFP4-
                                                        packed in one kernel
                                                        like Gemma 4 path)
 3. NVFP4-KV FA2 prefill (GQA-12)                     → 1 launch
 4. O projection (NVFP4 GEMM, M=N)                    → 1 launch
 5. residual add                                      → 1 launch
 6. RMSNorm(post-attn)                                → 1 launch
 7. gate/up projection (NVFP4 GEMM, M=N)              → 2 launches
                                                       (or 1 if fused
                                                        gate||up)
 8. SiLU + elementwise-mul (gate ⊙ silu(up))          → 1 launch
 9. down projection (NVFP4 GEMM, M=N)                 → 1 launch
10. residual add                                      → 1 launch
```

≈11–13 launches/layer × 88 layers ≈ **1k–1.2k launches per
prefill** regardless of `N`. That's the layer-major budget.

The legacy per-token loop (which `Mistral35Bringup::run_generate`
will explicitly NOT use) would be `N × ≈13 × 88 ≈ 1.1k × N`
launches — at `N = 256` that's ~280k launches, an order-of-
magnitude waste even before fence overhead.

## Kernel reuse map

### Reusable as-is (no Mistral variant needed)

| Kernel | Path | Notes |
|---|---|---|
| `rmsnorm_inplace_*` | `kernels/rmsnorm_*` | f16 + bf16 variants exist; pick the one matching the chosen activation dtype. |
| `silu_mul` | `kernels/silu_mul.cu` | gate ⊙ silu(up) — same shape as Qwen/Gemma. |
| `vector_add` | `kernels/vector_add.cu` | residual adds. |
| `EmbeddingGatherLaunch` | `rvllm-runtime` | per-prompt embed-table lookup; no model-specific math. |
| `argmax` / sampler | `kernels/argmax.cu`, `rvllm-sampling` | unchanged. |
| FA2 prefill NVFP4 | `kernels/flash_attention_unified_prefill_nvfp4kv.cu` | **no MAX_GQA cap on the prefill side** (verified — only the decode kernels carry the 4/8 limit). GQA-12 should fit; needs end-to-end cosine validation against a reference dump when the Mistral model directory becomes available. |

### Need new variants (Mistral-specific)

| Kernel | Why | Where to put it |
|---|---|---|
| **NVFP4 GEMM** (M=any) | Mistral's per-row × per-16-K NVFP4 weight scale grid is incompatible with the Gemma 4 / Qwen 3.6 fp8-block `[N/128, K/128]` ABI. Step 4 already has the Rust-side fn-pointer types (`Nvfp4GemmSm120Fn`); the `.cu` kernel skeleton is in `kernels/cutlass_nvfp4_gemm_sm120.cu`. | CUTLASS .so |
| **YaRN RoPE + NVFP4 KV write (fused)** | Existing `fused_rope_partial_nvfp4kv` (Gemma) computes RoPE with a single theta + partial-rotary factor. Mistral YaRN needs the per-frequency `mscale × ramp(beta_fast, beta_slow)` correction (`original_max=4096`, `factor=64`), and the resulting per-token Q/K rotated activations feed the same NVFP4 pack stage. | new `kernels/fused_yarn_rope_nvfp4kv_mistral.cu` (~80% of the code is the existing `fused_rope_partial_nvfp4kv` body, only the angle table changes). |
| **NVFP4 KV decode at GQA=12** | Existing fused / split decode kernels cap at GQA ≤ 4 / 8 (see `KvDecodeStrategy` gate, `mistral35_bring_up.rs`). Per-head fallback works for prefill+decode but pays `12×` duplicated K/V load on every decode tile. | raise `MAX_GQA_DECODE` in `flash_attention_nvfp4kv*.cu` and `MAX_GQA_SPLIT` in `flash_attention_split_decode_nvfp4kv*.cu` to ≥ 12. Register impact bounded: `q_reg[16][8]` = 512 B/thread × 128 threads = 64 KB per block, well under sm_121's 255-reg limit. |

### Drop entirely (no Mistral analogue)

- All MoE-related kernels: `router_gemv_batched_*`, `topk_softmax_batched_*`, `fp8_gemv_blockwise_wpr_native_*_indirect_batched_topk*`, `shared_gate_dot_sigmoid_*`, `scaled_add_*_devw_batched_topk*`. The Qwen `RVLLM_QWEN36_BATCH_MOE_*` env gates have no Mistral counterpart.
- Linear-attn (`gated_delta_rule_prefill_*`, `conv_state_advance_batched_*`). Mistral has no recurrent state.

## Phase plan

Phases mirror the Qwen rollout structure so the same audit tools
(`v3/tools/cmp_qwen36_prefill_layers.py` and friends) generalise.

### Phase M-A — projections batched over N (NVFP4 GEMM)

Replace the `M=1` GEMV path with the `M=N` CUTLASS NVFP4 GEMM
for **all seven projections** (q, k, v, o, gate, up, down) inside
the layer body. Single launch per projection per layer. Gates:

```
RVLLM_MISTRAL35_BATCH_PROJ_PREFILL=1     // all NVFP4 projections
RVLLM_MISTRAL35_FUSED_QKV=1              // optional: fused [Q || K || V]
                                          // weight layout if loader stages
                                          // a fused tensor (Mistral
                                          // checkpoint ships them split,
                                          // so this is opt-in)
```

Audit: per-(layer, phase) hidden-state dump comparable to Qwen's
`RVLLM_QWEN36_DUMP_DIR`. Every layer × phase × token must hit
`cos = 1.000000` against a reference per-token greedy run.

### Phase M-B — RoPE + NVFP4 KV write batched

The fused YaRN RoPE + NVFP4 pack kernel runs once per layer over
all `N` tokens. Q rotated in-place, K rotated and packed into the
paged NVFP4 cache slots in the same kernel.

```
RVLLM_MISTRAL35_BATCH_ROPE_PREFILL=1
```

### Phase M-C — FA2 NVFP4 prefill batched (GQA-12)

The existing `flash_attention_unified_prefill_nvfp4kv` kernel
already runs `[N, num_heads]` blocks. The GQA mapping inside the
kernel is a runtime division (`head_idx / (num_heads /
num_kv_heads)`); GQA=12 should fit without source changes.
Validation hook:

```
RVLLM_MISTRAL35_BATCH_FULL_PREFILL=1
```

### Phase M-D — outer-loop deletion

When phases A/B/C are all on, `Mistral35Bringup::run_generate`
runs strictly layer-major: no `for tok in 0..N` chain at the
top level. Mirrors Qwen's Phase 7 collapse.

### Phase M-E — CUDA Graph capture (decode)

Same scope as the parked Qwen Phase 8 (`v3/QWEN_BATCHED_PREFILL_
PLAN.md`). Capture the steady-state decode loop body, replay each
step. Only attempted after the prefill phases land green.

## Round-26 stream-fence invariant — port forward

Qwen learned the hard way (round-26 / 27 race) that
`pos_cl_region` / `context_lens` / `positions` per-token slot
ids must be populated by a kernel on the worker stream rather
than `Region::copy_from_host`. Mistral's bring-up adopts the same
pattern from day one — `cuMemcpyHtoD_v2` on the legacy default
stream races the non-blocking compute stream and produces token-
major non-determinism that masks as a quantisation cliff.

## Bench targets

Reference targets (taken on the same GB10 host as the Qwen
numbers):

* N = 22 prompt: prefill TTFT < 350 ms.
* N = 256 prompt: prefill TTFT < 4 s.
* Decode steady-state: comparable to the Gemma 4 31B fp8 path
  (~3–5 tok/s) once the GQA-12 decode kernel cap is raised.

These are placeholder bounds — once the kernel set is in, the
bench harness (`v3/scripts/bench_sm121.sh`) gets a Mistral
profile entry and the numbers turn into pass/fail gates for
phase rollout.

## Open questions

1. **Fused QKV weight layout** — the Mistral checkpoint ships
   `q_proj`, `k_proj`, `v_proj` separately. Synthesising a fused
   `[q_dim + 2 × kv_dim, hidden]` packed weight at load time
   would save 2 of 3 launches per layer per token, but doubles
   the loader's NVFP4 staging code. Defer until baseline phases
   are stable and the bench shows projection launches dominate.
2. **NVFP4 weight layout for `down_proj`** — the only projection
   with `K = intermediate_size = 28672`. K/16 = 1792 scale
   columns, ≥ the K/2=14336 packed columns of the q-proj path.
   The CUTLASS sm_120 NVFP4 kernel must validate against this
   shape range when the kernel implementation lands; current
   ABI (`Nvfp4GemmSm120Fn`) takes `m, n, k` so the path is
   uniform — only kernel-internal tile shapes need verification.
3. **Pixtral splice** — same architecture as Qwen/Gemma vision
   (host-side preprocess → device patch tensor → ViT forward →
   splice into prefill embed buffer at `slot.token_start`).
   Vision-bearing requests force `common_prefix_len = 0` and
   the chunked-prefill batch path, identical to the Gemma 4
   invariant. No new infrastructure needed; just the Pixtral
   GPU forward (host side already done in
   `vision_preprocess::preprocess_mistral35_pixtral`).

---

## 2026-05-10 status / measured baseline

`RVLLM_MISTRAL35_TIMING=1` on a 383-token German chat-templated
prompt + 64-token greedy decode:

```
prompt_tokens=383  decode_steps=63
prefill_ms=171691.8  prefill_tok_per_s=2.23   (86 % of request)
decode_ms=29887.1    decode_tok_per_s=2.11    (14 % of request)
```

Two facts the timing pins down:

* Prefill tok/s ≈ decode tok/s. `prefill_token` and `decode_token`
  both call `forward_smoke_q_proj_inner` at M=1, so prefill is
  literally per-token decode. No batched path is currently
  reachable — the gates documented at line 1294 in
  `mistral35_bring_up.rs` are stubs that abort if set.
* Decode at 2.11 tok/s ≈ 70 % of the unified-memory bandwidth
  ceiling for streaming the 67 GB W4 weight set per token. The
  `mistral35_w4a16_gemv_bf16` kernel (default-on path, line 1678)
  is already the Marlin-style fused dequant + GEMV — no separate
  bf16 weight scratch, no cublasLt round-trip. Codex review #4 was
  written against the legacy `RVLLM_W4A16_GEMV=0` opt-out path.

Conclusion: the only large per-request lever left is layer-major
batched prefill. Decode is at hardware limit for single-stream;
further decode wins need continuous batching or speculative decode
(out of scope here).

## Concrete next phases (in commit-sized chunks)

### Phase A — scaffolding, fall-through (next commit)

* `forward_smoke_chunk(tokens: &[u32], pos_start: i32,
  compute_logits_at: Option<usize>) -> Result<SmokeStageDump>` —
  new entry point. Initial implementation: per-token loop of
  `forward_smoke_q_proj_inner`, returning the dump at
  `compute_logits_at` (defaults to T-1).
* `RVLLM_MISTRAL35_BATCH_PREFILL` env gate. Default OFF. When ON,
  `generate` calls `forward_smoke_chunk(prompt, 0, Some(T-1))`
  instead of the per-token prefill loop.
* No kernel changes. Greedy smoke must stay byte-identical with
  the gate ON because the fallback path is per-token.

### Phase B — batched embed + RMSNorm + residual

`mistral35_embed_gather_t_bf16`, batched RMSNorm + residual launch
shapes. Cheap kernels, mostly establishes the `[T, hidden]`
buffer layout.

### Phase C — batched W4A16 path (the big win)

For each layer's 7 projections, run M=T `bf16_gemm_f32` against a
freshly dequantized weight tile (the legacy
`nvfp4_dequant_weights_bf16` + `cublaslt::bf16_gemm_f32` path
already exists for M=1; M=T is just a call-site change). The
fused M=1 GEMV stays as the decode fast path.

### Phase D — batched RoPE + KV write

T-strided RoPE + `mistral35_kv_cache_write_t_bf16`.

### Phase E — causal prefill attention

`mistral35_attn_prefill_bf16` — fused FA-2 forward, GQA-aware,
one CTA per kv_head tile across T queries.

### Phase F — flip default, retire legacy gate

Once the byte-identical equivalence test passes for prompts of
length 1, 16, 128, 383, flip `RVLLM_MISTRAL35_BATCH_PREFILL` to
default ON. Acceptance gate: prefill tok/s ≥ 8 (≥ 3.5× current),
greedy "Hallo!" + 64-tok German GQA decode + vision E2E all
byte-identical. **Done — `7de5fb0`.**

---

## Decode side — fused FA-decode + NVFP4-KV (codex review #3)

### Status

Two CUDA kernels landed as opt-in building blocks; full
end-to-end NVFP4-KV decode path is multi-session.

* `mistral35_fa_decode_gqa_bf16` (`9ffe799`, `8d21e39`) —
  fused FlashAttention-2 m=1 decode. Online softmax in
  registers, scores/probs never in DRAM, Q loaded once,
  K/V tiles streamed per kv_head. Currently 19 % slower
  than the split-kernel default; left opt-in via
  `RVLLM_MISTRAL35_FA_DECODE=1`. Correctness verified —
  text "Hallo!", 64-tok German GQA, vision orange-ball E2E
  all byte-identical to baseline. Warp-shuffle reduction
  + UB fix landed in `8d21e39`.

* `mistral35_kv_cache_write_nvfp4_bf16` (this commit) —
  converts BF16 K/V row to NVFP4 packed nibbles + E4M3
  per-16-element scale, writes at cache slot `pos`.
  3.55× smaller storage (0.5625 vs 2 bytes/elem). Building
  block; not yet wired into the production write path.

### What's still missing for end-to-end NVFP4-KV

1. **NVFP4 KV cache buffers** — allocate `packed`
   `[max_pos, n_kv, head_dim/2]` + `scale`
   `[max_pos, n_kv, head_dim/16]` per layer, gated on
   `RVLLM_MISTRAL35_NVFP4_KV=1`.
2. **Write dispatch** — when the gate is on, switch
   `attention_step`'s KV-write call to the new NVFP4 kernel
   instead of `mistral35_kv_cache_write_bf16`.
3. **FA-decode NVFP4 read variant** — second variant of
   `mistral35_fa_decode_gqa_bf16` that reads NVFP4 packed +
   scale, dequant-in-kernel inside the K/V tile load. Same
   online-softmax body, ~80 LOC delta.
4. **Equivalence check** — confirm
   `dequant(write_nvfp4(bf16_x))` returns the BF16 K/V used
   downstream (cosine ≥ 0.999 per slot, by far the most
   important validation step). Use the existing weight-side
   `nvfp4_dequant_weights_bf16` decode logic as the
   ground-truth reference for the FP4 LUT.
5. **Acceptance gate** — vision orange-ball + 64-tok German
   GQA byte-identical (allowing for the ~0.1 % NVFP4 quant
   noise; if drift is too high, the K-scale policy may need
   the existing `mse` variant rather than the plain
   `amax/6` used in the write kernel today).

### Why it's worth doing

At past_len ≈ 400 (current Mistral 3.5 workloads):
* BF16 KV bytes per decode token: ≈ 70 MB across 88 layers
* NVFP4 KV bytes:                   ≈ 20 MB
* Weight bytes (unchanged):         ≈ 530 MB
* Total decode-bandwidth saving:    ≈ 8 %

At past_len ≈ 4000 (long-context decoding):
* BF16 KV bytes: ≈ 700 MB     →  total 1230 MB
* NVFP4 KV bytes: ≈ 200 MB    →  total 730 MB
* Saving:        ≈ 40 % decode tok/s

So the win is small at current short-context use but
transformative for long-context decoding — the regime where
KV bandwidth surpasses weight bandwidth.

---

## 2026-05-10 — refined scope after kernel audit + per-stage timing

`RVLLM_MISTRAL35_LAYER_TIMING=1` per-token forward (87 layers):

  norm + QKV + RoPE   58.4 ms   13.1 %
  attention            9.2 ms    2.1 %
  O proj + post-norm  48.3 ms   10.9 %
  MLP                329.7 ms   74.0 %

So Phase C should land MLP first (gate / up / down at M=T) — that
captures three quarters of the per-token time on its own. Then
QKV + O at M=T captures another 24 %. Attention at 2 % is barely
worth a fused prefill kernel.

### Kernel-readiness inventory

Walked the kernel set against the [T, hidden] flow this iteration.
Most of the building blocks are already batch-capable:

| Kernel | Status | Notes |
|---|---|---|
| `embedding_gather_bf16` | ✅ batched | grid = (num_tokens, 1, 1) |
| `RmsnormInplaceLaunch` | ✅ batched | num_tokens param, grid stride |
| `vector_add_bf16` | ✅ batched-equivalent | 1-D over `n = T * hidden` |
| `silu_mul_bf16` | ✅ batched-equivalent | 1-D over `n = T * intermediate` |
| `nvfp4_dequant_weights_bf16` | ✅ weight-only | M-independent |
| `cublasLt bf16_gemm_f32` | ✅ M param native | already takes M, n, k |
| `mistral35_w4a16_gemv_bf16` | ⊘ M=1 only (correct) | stays as decode fast path |
| `f32_to_bf16` cast | ⚠ hardcoded `count = n` | needs `count = m*n` for M>1 |
| `rope_split_half_bf16` | ⚠ per-call num_tokens param | call sites loop, easy fix |
| `kv_cache_write_bf16` | ⚠ per-call slot | call sites loop, easy fix |
| **causal prefill attention** | ❌ no kernel | the only real new-kernel blocker |

### Pragmatic intermediate — Phase C-light (no new attention kernel)

Build a chunked path that batches everything *except* attention:

  embed_gather_t_bf16 → [T, hidden]
  for layer in 0..N:
      rmsnorm           num_tokens=T
      qkv  M=T          legacy dequant + bf16_gemm_f32
      rope_t            existing kernel × T (still cheap)
      kv_write_t        existing kernel × T (still cheap)
      for t in 0..T:    PER-QUERY ATTENTION LOOP
          attention_step at past_len = pos_start + t + 1
      o    M=T          legacy path
      residual + rmsnorm
      gate, up   M=T
      silu_mul          n = T * intermediate
      down  M=T
      residual

Attention per-query loop costs ≈ 2 % of per-token time × T calls
≈ 2 % of total. Net of the M=T GEMM win (74 % MLP + 24 % QKV/O
moving from bandwidth-bound GEMV to GEMM-throughput): expected
prefill 3–5× speedup matches the original plan without writing a
new attention kernel.

### Concrete TODO list, in commit-sized chunks

1. **Bump `out_f32_scratch` to T·N_max·4 bytes** at bring-up.
   Pick `T_max` = the existing `RVLLM_MAX_TOKENS_CAP` (=64) or a
   profile-knob; rejecting longer chunks at the entry point.
2. **Generalise `f32_to_bf16` call site to count = M*N** in the
   legacy `gemm` helper.
3. **Wire MLP at M=T** (gate/up/down). Smallest reachable speedup:
   the 74 % bucket. Test on a 4-token prompt with the chunk path,
   byte-compare logits at position T-1 vs the per-token loop.
4. **Wire QKV + O at M=T**. Adds the 24 % bucket.
5. **Per-query attention loop** in the chunk path. Reuse the
   existing M=1 attention kernels.
6. **Embed gather + post-norm + LM head row pick**. The "outer
   shell" of the forward, mostly already batched-friendly.
7. **Acceptance suite**: token-equivalence at prompt lengths
   1 / 4 / 16 / 64 / 372, vision E2E unchanged.
8. **Phase E (later, optional)**: causal prefill attention
   kernel. Only worth doing if attention's 2 % share becomes the
   new bottleneck once the GEMVs are batched.
