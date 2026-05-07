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
