# Unified FP8-KV prefill for sm_121 — port of vLLM Triton kernel

**Branch:** `rusty_sm121_unified_prefill` (off `rusty_sm121`)
**Goal:** close the TTFT gap to vLLM. On a 1836-token Gemma 4 31B prompt
we measure vLLM 1.8 s vs rvllm 61 s (≈34×). The dominant cost is
`gemma4_layer_exec::Gemma4Phase::Prefill` running a
`PagedDecodeFp8Launcher` per prompt token per layer (≈110 k attention
launches for 1836 tokens × 60 layers). Replace that loop with one
multi-Q FA2 prefill kernel launch per layer — the pattern vLLM uses via
`kernel_unified_attention_2d` (Triton).

## Why not the pre-existing FA2 prefill kernel

Commit `1f05836` landed a dedicated FA2 FP8 prefill kernel on sm_121;
commit `55491e9` dropped it. The kernel diverged numerically from the
per-token decode path that `rvllm-ppl` validates (see
`PagedPrefillFp8Launcher::Fa2Ptx` arm in `prefill.rs`, the "numerically
identical" comment). This port starts fresh with bit-parity to the
decode reference as a hard gate, not an aspiration.

## Algorithm (straight port of `kernel_unified_attention_2d`)

```
grid = (total_num_q_blocks, num_kv_heads)
block = FA2_THREADS (= 128)

per program:
  seq_idx       = find_seq_idx(cu_seqlens_q, program_id(0), BLOCK_Q)
  q_block_local = program_id(0) - cu_seqlens_q[seq_idx]/BLOCK_Q - seq_idx
  kv_head_idx   = program_id(1)
  context_len   = seq_lens[seq_idx]  (= full sequence; includes the prompt)
  query_len     = cu_seqlens_q[seq_idx+1] - cu_seqlens_q[seq_idx]
  context_prefix= context_len - query_len  (= 0 for single-shot prefill)

  query_pos[BLOCK_M] = q_block_local * BLOCK_Q + m/num_queries_per_kv
  query_head[BLOCK_M] = kv_head_idx*num_queries_per_kv + m%num_queries_per_kv

  load Q[BLOCK_M][head_dim] from query_ptr (FP8, dequant with per-(seq,head) scale)

  M[BLOCK_M] = -inf
  L[BLOCK_M] =  1
  acc[BLOCK_M][head_dim] = 0

  for tile in 0..num_tiles:
      seq_offset[TILE_SIZE] = tile*TILE_SIZE + t
      tile_mask  = seq_offset < max_seq_prefix_len
      phys_block = block_tables[seq_idx][seq_offset / BLOCK_SIZE]
      slot       = phys_block*BLOCK_SIZE + (seq_offset % BLOCK_SIZE)

      K[head_dim][TILE_SIZE] = key_cache[slot][kv_head_idx][:]        (FP8)
      V[TILE_SIZE][head_dim] = value_cache[slot][kv_head_idx][:]       (FP8)
      k_scale[TILE_SIZE] = k_scale_cache[slot][kv_head_idx]            (f32)
      v_scale[TILE_SIZE] = v_scale_cache[slot][kv_head_idx]            (f32)

      S[BLOCK_M][TILE_SIZE] = scale * (Q dot K)
                            * k_scale[None,:]                          # per-slot K
      if USE_SOFTCAP: S = softcap * tanh(S/softcap)
      mask:
        causal:          seq_offset[None,:] <= context_prefix + query_pos[:,None]
        sliding (opt):   (context_prefix + query_pos[:,None]) - seq_offset[None,:] < SLIDING_WINDOW

      # online softmax (flash-attention 2)
      m_j[BLOCK_M] = max(M, max_over_tile(S))
      P[BLOCK_M][TILE_SIZE] = exp(S - m_j)
      l_j[BLOCK_M] = sum_over_tile(P)
      alpha[BLOCK_M] = exp(M - m_j)
      acc *= alpha
      L    = L*alpha + l_j
      M    = m_j
      acc += (P * v_scale[None,:]) dot V                              # per-slot V

  out[BLOCK_M][head_dim] = acc / L  (cast to f16)
```

This is line-for-line the `kernel_unified_attention_2d` body, adjusted
for rvllm's data layouts. The mapping is:

| vLLM Triton (KV_QUANT_MODE=3)       | rvllm current                                   |
|-------------------------------------|-------------------------------------------------|
| `q, k, v`  FP8 E4M3                 | `scratch.q_fp8`, `scratch.k_cache`, `scratch.v_cache` |
| per-(blk, slot, head) `k_scale_cache` f32 | `scratch.k_scale_cache`, `[slot*num_kv_heads+kvh]` |
| per-(blk, slot, head) `v_scale_cache` f32 | `scratch.v_scale_cache` (same layout)           |
| per-(seq, head) `q_descale` (tensor)     | `scratch.q_scale_cache[seq*num_heads + h]`       |
| `block_tables[num_seqs, max_blocks_per_seq]` i32 | `meta.block_tables` (same)                      |
| `cu_seqlens_q[num_seqs+1]` i32           | `scratch.cu_seqlens_q` (to be prepared)         |
| `seq_lens[num_seqs]` i32                  | `scratch.context_lens`                         |
| `SLIDING_WINDOW` constexpr                | `dims.sliding_window` per-layer in Rust         |
| `USE_SOFTCAP` constexpr                   | Gemma 4 has no attn softcap → `false`           |
| `scale` softmax scale                    | `dims.attn_scale`                              |

## Parameter choices for Gemma 4 31B

For every sliding and global layer `BLOCK_M = 16`, `BLOCK_Q = BLOCK_M /
num_queries_per_kv`, chosen so one program covers 16 Q rows of a single
KV head:

| layer kind | q_heads | kv_heads | q/kv | head_dim | BLOCK_M | BLOCK_Q | TILE_SIZE | sliding |
|------------|---------|----------|------|----------|---------|---------|-----------|---------|
| sliding    | 32      | 16       | 2    | 256      | 16      | 8       | 32        | 1024    |
| global     | 32      |  4       | 8    | 512      | 16      | 2       | 16        | 0       |

TILE_SIZE picked to stay inside the sm_121 99 KB dynamic-smem opt-in
cap. Keeping K/V in smem as FP8 (1 B/elem) rather than dequantised
(4 B/elem) gives headroom:

| thing      | sliding (head=256, TILE=32)         | global (head=512, TILE=16)      |
|------------|-------------------------------------|---------------------------------|
| Q (f32)    | 16 × 256 × 4 = 16 KB                | 16 × 512 × 4 = 32 KB            |
| K (FP8)    | 256 × 32 × 1 = 8 KB                 | 512 × 16 × 1 = 8 KB             |
| V (FP8)    | 32 × 256 × 1 = 8 KB                 | 16 × 512 × 1 = 8 KB             |
| S (f32)    | 16 × 32 × 4 = 2 KB                  | 16 × 16 × 4 = 1 KB              |
| acc (f32)  | 16 × 256 × 4 = 16 KB                | 16 × 512 × 4 = 32 KB            |
| M, L, reduce | ~1 KB                            | ~1 KB                           |
| **total**  | **≈ 51 KB**                         | **≈ 82 KB**                     |

Fits. `acc` can live in registers (16 rows × `dims_per_thread` elems per
thread) but the arithmetic is simpler if it starts in smem; optimise
later if needed.

## Scope (what's in / out)

**In.** Causal mask, sliding window, FP8 E4M3 KV with per-slot scales,
per-(seq, head) Q scale, single-batch (`num_seqs = 1`), head_dim
256 / 512. Output f16.

**Out.** ALiBi, sinks, multimodal prefix, query-query bias, softcap —
Gemma 4 doesn't use any of these in attention (logit softcap is on the
LM head, not inside attention). Leave as `constexpr false` branches
that the compiler drops; add when another model needs them.

Multi-batch (`num_seqs > 1`) is out-of-scope for v1 because rvllm-serve
runs single-request today; adding it is a later change to the program
index → (seq, q_block) map, not a kernel rewrite.

## Correctness gate

Byte-identical to the current decode-per-qi reference on three prompt
lengths (256, 1024, 2500 tokens). `rvllm-ppl chunk=128` stays at 2.56 ±
noise. A dedicated Python harness (`v3/tools/fa2_unified_prefill_check.py`)
runs the single-layer kernel against a NumPy reference and reports
max-abs-err; first-token parity runs through `rvllm-serve` against the
live model.

## Non-goals for this branch

* `num_seqs > 1` — out of scope as above.
* Mixed prefill + decode in one launch (vLLM's chunked prefill) — needs
  a scheduler rework first.
* 3D (segmented) kernel — 2D is enough up to ~4 k tokens, which is all
  rvllm-serve advertises (`RVLLM_MAX_TOKENS_CAP=4096`).
* FP8 output (`USE_FP8` in vLLM) — our attention output is f16, the
  O-proj quantises downstream.

## Phases

* **A.** Design doc (this file) + kernel skeleton + Rust launcher stubs.
  Compiles. **In this commit.**
* **B.** Kernel body — Q load, tile loop, softmax, V dot, epilogue.
  Target: numerical parity on a single layer in the Python harness.
* **C.** Wire into `gemma4_layer_exec::Gemma4Phase::Prefill` as the
  default path for `num_tokens >= 2`; keep the decode-per-qi loop
  behind `RVLLM_UNIFIED_PREFILL=0` for bisect.
* **D.** Live-model validation — first-token parity on 3 prompt
  lengths via rvllm-serve, `rvllm-ppl` regression bound, TTFT
  measurement vs vLLM.
* **E.** Cherry-pick / rebase onto `rusty_sm121_inference_server` so
  rvllm-serve picks it up.

## File layout

| new / changed file                                                    | phase |
|-----------------------------------------------------------------------|-------|
| `v3/UNIFIED_PREFILL_SPEC.md`                                          | A     |
| `kernels/flash_attention_unified_prefill.cu`                          | A/B   |
| `v3/crates/rvllm-attention/src/lib.rs` (Fa2PtxKernels fn ptr)         | A     |
| `v3/crates/rvllm-attention/src/prefill.rs` (Fa2Ptx arm)               | A/B   |
| `v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs` (Phase::Prefill)   | C     |
| `v3/tools/fa2_unified_prefill_check.py`                               | B     |
