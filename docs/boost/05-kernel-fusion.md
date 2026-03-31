# 05: Kernel Fusion

## Kernel Fusion Opportunities in rvLLM: A Complete Analysis

## 1. Complete Kernel Trace for One Decode Step

Reference model: Qwen2.5-1.5B (hidden=1536, intermediate=8960, num_heads=12, num_kv_heads=2, head_dim=128, 28 layers), f16, batch size N=1 decode.

### 1.1 Current Maximally-Fused T=1 Decode Path (gpu_layer.rs, line 527+)

Per-layer kernel launches in the **f16 fused decode path** (the current production path for M=1):

| Step | Operation | Kernel(s) | Launches | Allocs |
|------|-----------|-----------|----------|--------|
| 1+2 | Fused add+norm+QKV GEMV | `fused_cute_add_norm_qkv_gemv` | 1 | 2 |
| 3 | QKV bias add (if model has bias) | `add_bias_broadcast_f16` | 1 | 0 |
| 4+5 | Fused RoPE + KV cache write | `fused_rope_cache_f16_kernel` | 1 | 0 |
| 6 | Paged decode attention | `paged_attention_v2_f16kv` | 1 | 1 |
| 7 | O-projection (cuBLAS hgemm) | cuBLAS `cublasHgemm` | 1 | 1 |
| 8+9 | Fused add+norm+gateup GEMV | `fused_cute_add_norm_gateup_gemv` | 1 | 2 |
| 10 | Fused SiLU*mul+down GEMV | `fused_cute_silu_down_gemv` | 1 | 1 |
| **Subtotal per layer** | | | **7** (8 with bias) | **7** |
| **28 layers** | | | **196** (224 with bias) | **196** |

Layer 0 differs: uses `fused_cute_norm_qkv_gemv` (no residual add from previous layer).

Post-layer operations:

| Step | Operation | Kernel(s) | Launches | Allocs |
|------|-----------|-----------|----------|--------|
| Final | Fused residual+RMSNorm | `fused_residual_rmsnorm_f16` | 1 | 2 |
| LM head | Fused LM head argmax pass 1 | `fused_lm_head_argmax_f16_kernel` | 1 | 2 |
| LM head | Fused LM head argmax pass 2 | `fused_lm_head_argmax_reduce_kernel` | 1 | 1 |
| **Post-layer total** | | | **3** | **5** |

Pre-step overhead:

| Step | Operation | Count |
|------|-----------|-------|
| Metadata upload | Packed memcpy_htod | 1 |
| Embedding lookup | `embedding_gather_f16` | 1 |
| Output download | memcpy_dtoh (i32 token IDs) | 1 |

**Grand total per decode step (Qwen2.5-1.5B, M=1 fused path):**

| Metric | Count |
|--------|-------|
| Kernel launches | 200 (bias model: 228) |
| GPU allocations | 201 |
| HtoD memcpy | 1 |
| DtoH memcpy | 1 |

### 1.2 Current T>1 Batched Decode / Prefill Path (cuBLAS GEMMs)

Per-layer kernel launches for the **scratch-buffer batched path** (num_tokens > 1):

| Step | Operation | Kernel(s) | Launches |
|------|-----------|-----------|----------|
| 1 | Fused residual+RMSNorm (or plain RMSNorm for L0) | `fused_residual_rmsnorm_f16` | 1 |
| 2 | QKV GEMM (fused weight) | cuBLAS hgemm | 1 |
| 2b | Deinterleave QKV (if T>1) | `deinterleave_qkv_f16` | 1 |
| 3 | QKV bias (3 separate launches for split layout) | `add_bias_f16` x3 | 3 |
| 4 | RoPE (in-place) | `rotary_embedding_f16` | 1 |
| 5 | KV cache write | `reshape_and_cache_f16` | 1 |
| 6 | Attention (FA or paged) | attention kernel | 1 |
| 7 | O-projection GEMM | cuBLAS hgemm | 1 |
| 8 | Fused residual+RMSNorm | `fused_residual_rmsnorm_f16` | 1 |
| 9a | GateUp GEMM | cuBLAS hgemm | 1 |
| 9b | SiLU*Mul | `silu_mul_interleaved_f16` | 1 |
| 10 | Down-projection GEMM | cuBLAS hgemm | 1 |
| **Subtotal per layer (bias model)** | | | **14** |
| **28 layers** | | | **392** |

With CUTLASS available, steps 7-8 become `cutlass_oproj_residual` + `rms_norm_f16` (still 2 kernels), and steps 9a-9b become `cutlass_gateup_silu` (1 kernel). This reduces to **11 per layer, 308 total**.

### 1.3 Kernel Launch Categorization (T=1 Fused Path, 28 Layers)

| Category | Launches per layer | Total (28L) | Percentage |
|----------|--------------------|-------------|------------|
| Fused norm+GEMV kernels | 2 | 56 | 28% |
| Fused SiLU+GEMV kernels | 1 | 28 | 14% |
| Fused RoPE+cache kernels | 1 | 28 | 14% |
| cuBLAS GEMM (O-proj) | 1 | 28 | 14% |
| Attention | 1 | 28 | 14% |
| Bias add | 1 | 28 | 14% |
| Post-layer (norm+LM head) | - | 4 | 2% |
| **Total** | 7-8 | **200-228** | 100% |

## 2. Current Fusions: What Is Already Fused

### 2.1 Fusion Inventory

| Fusion | Kernel Name | Ops Fused | HBM Saved | Launch Saved | Path |
|--------|-------------|-----------|-----------|-------------|------|
| RMSNorm + QKV GEMV | `fused_cute_norm_qkv_gemv` | 2 | hidden * 2B (normed state not materialized) | 1 | T=1 decode |
| Add + RMSNorm + QKV GEMV | `fused_cute_add_norm_qkv_gemv` | 3 | hidden * 4B (residual intermediate + normed) | 2 | T=1 decode L1+ |
| Add + RMSNorm + QKV bias GEMV | `fused_cute_add_norm_qkv_bias_gemv` | 4 | hidden * 4B + qkv_dim * 2B | 3 | T=1 bias models |
| Add + RMSNorm + GateUp GEMV | `fused_cute_add_norm_gateup_gemv` | 3 | hidden * 4B | 2 | T=1 decode |
| SiLU * Mul + Down GEMV | `fused_cute_silu_down_gemv` | 3 | intermediate * 2B (activated output not materialized) | 2 | T=1 decode |
| RoPE + KV cache write | `fused_rope_cache_f16_kernel` | 2 | kv_dim * 2B (roped K not stored to HBM before cache write) | 1 | All decode |
| Residual add + RMSNorm | `fused_residual_rmsnorm_f16` | 2 | hidden * 2B | 1 | T>1 path, final norm |
| O-proj GEMM + residual add | `cutlass_oproj_residual` (CUTLASS) | 2 | hidden * 2B (residual add folded into GEMM epilogue) | 1 | T>1 with CUTLASS |
| GateUp GEMM + SiLU * Mul | `cutlass_gateup_silu` (CUTLASS) | 3 | intermediate * 4B (gate+up intermediate not materialized separately) | 1 | T>1 with CUTLASS |
| O-proj + Add + Norm + GateUp GEMV | `fused_cute_oproj_add_norm_gateup_gemv` | 5 | Entire O-proj output not materialized to HBM | 4 | T=1 small models only |
| Fused LM head + argmax | `fused_lm_head_argmax_f16_kernel` | 2 | vocab_size * 4B (logits not materialized) | 1 | T=1 greedy |
| QKV GEMM + bias (CUTLASS) | `cutlass_qkv_bias` | 2 | qkv_dim * 2B | 1 | T>1 with CUTLASS |

### 2.2 Speedup Estimates from Current Fusions

For Qwen2.5-1.5B (hidden=1536, intermediate=8960, qkv_dim=2048, head_dim=128):

**Per-layer HBM savings (T=1 fused path vs unfused):**
- Add+Norm+QKV GEMV: eliminates 1536 * 4 = 6,144 bytes read+write = 12.3 KB round-trip
- Add+Norm+GateUp GEMV: eliminates 1536 * 4 = 6,144 bytes = 12.3 KB round-trip
- SiLU+Mul+Down GEMV: eliminates 8960 * 2 = 17,920 bytes = 17.9 KB round-trip
- RoPE+Cache: eliminates 2 * 256 * 2 = 1,024 bytes = 1.0 KB round-trip
- Cross-layer residual fusion: eliminates 1536 * 2 = 3,072 bytes = 3.1 KB

**Per-layer launch savings:** 5-6 launches saved. At ~5 us per launch, that is 25-30 us per layer, 700-840 us per step.

**Total per-step:** 700-840 us from launch overhead alone + HBM savings of approximately 43 KB per layer = 1.2 MB total across 28 layers. At 2 TB/s bandwidth, that saves ~0.6 us -- negligible at M=1 but significant at higher batch sizes.

The dominant benefit is **launch overhead elimination**, not bandwidth, at M=1.

## 3. Missing Fusion Opportunities

### 3.1 T=1 Decode Path Remaining Unfused Operations

The current T=1 path has 7 launches per layer. The unfused operations are:

1. **QKV bias add** (1 launch) -- separate from the fused norm+QKV GEMV
2. **O-projection** (1 cuBLAS GEMM launch) -- separate from the mega-fused oproj+add+norm+gateup
3. **Attention** (1 launch) -- inherently isolated by its reduction nature

### 3.2 T>1 Batched Path Remaining Unfused Operations

The T>1 path has 11-14 launches per layer. Unfused gaps:

1. **RMSNorm output -> QKV GEMM input**: HBM round-trip of hidden * T * 2 bytes
2. **QKV GEMM output -> Deinterleave -> RoPE**: 2-3 separate kernels
3. **RoPE -> KV cache write**: separate kernels (not using fused_rope_cache for T>1)
4. **Attention output -> O-proj GEMM input**: HBM round-trip of q_dim * T * 2 bytes
5. **O-proj -> residual add -> RMSNorm -> GateUp GEMM**: 3 operations, only partially fused
6. **GateUp output -> SiLU*Mul -> Down GEMM input**: 2-3 operations
7. **Down GEMM output -> next layer's residual**: HBM round-trip of hidden * T * 2 bytes

## 4. Detailed Fusion Opportunity Analysis

### 4.1 Fusion Opportunity F1: Residual + RMSNorm + GEMM Megafusion (T>1)

**Current state:** For T>1, the sequence is:
1. `fused_residual_rmsnorm_f16` (elementwise): write `normed[T, hidden]` to HBM
2. `cuBLAS hgemm` (GEMM): read `normed[T, hidden]` from HBM

**Proposed fusion:** Fold the residual add and RMSNorm into the GEMM's prologue. The CUTLASS framework supports custom prologues via `VisitorCompute` in the EVT (Epilogue Visitor Tree) system, but SM90 also supports custom mainloop modifications.

**Implementation approach:**
- Use CUTLASS 3.x `CollectiveMainloop` with a custom A-operand load that performs in-register residual add + RMSNorm before feeding data to WGMMA.
- Each threadblock tile loads a [TileM, K] chunk from A. Before the WGMMA, threads perform: `a_tile[m][k] = rmsnorm(residual[m][k] + attn_out[m][k], weight[k], eps)`.
- The reduction across K for RMSNorm requires a full-row scan first. For a tile-based GEMM, this requires a two-pass scheme: pass 1 computes `rms_scale` per row, pass 2 applies normalization and feeds WGMMA.
- Alternative: pre-compute `rms_scale[T]` in a tiny 1-block kernel (T elements), then fuse only the elementwise `x * weight * rms_scale` into the GEMM prologue.

**HBM savings:**
- Eliminated: `normed[T, hidden]` write (T * 1536 * 2 bytes) + read (T * 1536 * 2 bytes)
- For T=128: 128 * 1536 * 4 = 786 KB per fusion site
- Two sites per layer (pre-attn norm->QKV, post-attn norm->GateUp): 1.57 MB per layer
- 28 layers: **44 MB per step**

**Microseconds saved:**
- At 2 TB/s: 44 MB / 2 TB/s = 22 us
- Plus 1 kernel launch saved: 5 us
- **Total: ~27 us per step** at T=128

**Register pressure:** High. RMSNorm requires storing the `rms_scale` per row in registers (1 float per row in the tile, typically 128 rows = 128 floats = 512 bytes). This is feasible with TileM=128 but reduces the number of pipeline stages the mainloop can buffer.

**Shared memory:** The RMSNorm reduction needs hidden_size floats in shared memory for the sum-of-squares. For hidden=1536, that is 6 KB -- well within the CUTLASS mainloop's shared memory budget.

**JIT compiler changes:** Add a new `FusionOp::GemmPrologue { norm: bool, residual: bool }` to the IR. The codegen needs to emit CUTLASS CollectiveBuilder with custom prologue A-load. The template engine needs a new template `fused_norm_gemm_prologue.cu.template`.

**Priority: MEDIUM.** The savings scale with T but at T=1 (the latency-critical decode) the GEMV path already fuses these. Most impactful for batched decode (T=8-128).

### 4.2 Fusion Opportunity F2: Attention Output + Residual + Norm

**Current state (T=1):**
- Attention writes `attn_out[q_dim]` to HBM
- cuBLAS reads `attn_out[q_dim]` for O-projection, writes `oproj[hidden]` to HBM
- `fused_cute_add_norm_gateup_gemv` reads `oproj[hidden]` and `residual[hidden]`

The mega-fused kernel `fused_cute_oproj_add_norm_gateup_gemv` already fuses steps 7-9 (O-proj + add + norm + gateup) for small models where O-proj weights fit in L2. For 7B+ models, the O-proj weight matrix exceeds L2 capacity, making the redundant O-proj computation in each block catastrophically slow.

**Current state (T>1):**
- `cutlass_oproj_residual` fuses O-proj GEMM + residual add
- Separate `rms_norm_f16` follows

**Proposed fusion:** Fuse the O-proj GEMM epilogue with both residual add AND RMSNorm.

**CUTLASS epilogue fusion approach:**
- The `cutlass_oproj_residual` already uses `LinearCombination` with alpha=1, beta=1 to fuse `D = A@B + residual`.
- Extend: after the epilogue computes each output tile row, immediately compute partial sum-of-squares. After all tiles for a row are accumulated, compute `rms_scale` and apply normalization.
- This is a **streaming RMSNorm epilogue**: each CTA computes its tile's contribution to sum-of-squares, atomicAdd to a global counter, last-CTA-standing performs the final normalization.

**HBM savings:**
- Eliminated: `oproj_with_residual[T, hidden]` write + read before RMSNorm
- T=128: 128 * 1536 * 4 = 786 KB per layer, 28 layers = 22 MB

**Microseconds saved:** ~11 us (bandwidth) + 5 us (launch) = ~16 us per step

**Complexity:** HIGH. Streaming reductions in GEMM epilogues require careful synchronization. The CUTLASS EVT framework supports custom visitors, but an RMSNorm visitor with cross-CTA reduction is non-trivial. The alternative is a "split-accumulate" scheme where each CTA writes partial sum-of-squares to a global buffer, and a lightweight fixup kernel computes the final norm.

**JIT compiler changes:** New CUTLASS kernel template with custom `EpilogueVisitor` implementing the residual add + partial RMSNorm accumulation. The dispatch module needs a new pattern `OprojResidualNorm`.

**Priority: LOW-MEDIUM.** The benefit is moderate and the implementation is complex. Better to prioritize F5 and F6 first.

### 4.3 Fusion Opportunity F3: RoPE + KV Cache Write + Attention

**Current state (T=1):** `fused_rope_cache_f16_kernel` already fuses RoPE + KV cache write. Attention is a separate launch.

**Proposed fusion:** Fuse RoPE application, KV cache write, and the start of attention into one kernel.

**Analysis:** This is fundamentally infeasible for paged attention. The attention kernel must read the ENTIRE KV cache (all previous tokens) -- not just the current token's K/V. The cache write for the current token and the attention computation over all past tokens are different computational patterns:

- Cache write: scatter 1 token's K/V to paged slots (O(1) per head)
- Attention: read ALL cached K/V blocks, compute softmax(Q@K^T)@V (O(context_len) per head)

The only viable sub-fusion is merging the current-token's QK dot product into the cache write kernel (compute `q @ k_current` while writing k to cache), avoiding one cache read for the current token's key. This saves `num_heads * head_dim * 2` bytes per token -- trivial.

**HBM savings:** Negligible (~6 KB for Qwen2.5-1.5B)

**Priority: SKIP.** Not worth the complexity. The attention kernel is already highly optimized and functions as a natural fusion barrier (full-tensor reduction over KV cache).

### 4.4 Fusion Opportunity F4: GEMM Epilogue Fusions (CUTLASS-Style)

**Current state:** Two CUTLASS epilogue fusions exist:
1. `cutlass_oproj_residual`: O-proj + residual add
2. `cutlass_gateup_silu`: GateUp GEMM + SiLU*Mul

**Missing epilogue fusions:**

#### F4a: QKV GEMM + Bias Add Epilogue
**Current:** `cutlass_qkv_bias` exists but only for T=1 (interleaved layout). For T>1, bias is added with separate kernel(s).

**Proposed:** Extend `cutlass_qkv_bias` to T>1 with deinterleaved output layout. The CUTLASS epilogue already supports bias addition via the `LinearCombinationBias` functor. The challenge is the output layout transformation (interleaved to split Q/K/V).

**HBM savings:** T * qkv_dim * 2 bytes (bias intermediate eliminated)
- T=128: 128 * 2048 * 2 = 524 KB per layer = 14.7 MB total
- At 2 TB/s: 7.3 us saved

**JIT changes:** New CUTLASS template with `EpilogueBias` and scatter-store epilogue for split layout.

**Priority: MEDIUM.** Moderate savings, moderate complexity.

#### F4b: Down-Projection GEMM + Residual Add Epilogue
**Current:** Down GEMM writes to HBM, next layer's fused_residual_rmsnorm reads it back and adds the residual.

**Proposed:** Fuse the residual add into the Down-projection GEMM epilogue (identical pattern to `cutlass_oproj_residual`). Output `D = down_gemm_out + residual`. The subsequent RMSNorm then reads `D` once instead of reading both `down_out` and `residual` separately.

**HBM savings:** T * hidden * 2 bytes (residual read eliminated from next operation)
- T=128: 128 * 1536 * 2 = 393 KB per layer = 11 MB total
- At 2 TB/s: 5.5 us saved + 1 launch = ~10 us

**JIT changes:** Clone the `cutlass_oproj_residual` template with different matrix dimensions.

**Priority: MEDIUM-HIGH.** Easy to implement (copy existing pattern), moderate savings.

#### F4c: GateUp GEMM + SiLU*Mul + Down-Projection Megafusion
**Current:** `cutlass_gateup_silu` writes `activated[T, intermediate]` to HBM. Down-proj GEMM reads it.

**Proposed:** This is a GEMM -> elementwise -> GEMM pattern. True fusion (keeping the activated intermediate in registers) is generally not possible because the intermediate dimension (8960) is too large to hold in any on-chip storage. However, a **streamK-pipelined approach** could overlap the GateUp epilogue's SiLU*Mul computation with the Down-proj GEMM's data loads:

- GateUp GEMM produces tiles of `[TileM, 2*intermediate]` 
- SiLU*Mul reduces each tile to `[TileM, intermediate]`
- Down-proj GEMM consumes tiles of `[TileM, intermediate]`

With CUTLASS's persistent kernel and stream-K decomposition, these could be pipelined on the same SM cluster. However, this requires fundamentally different kernel scheduling.

**HBM savings:** T * intermediate * 2 bytes
- T=128: 128 * 8960 * 2 = 2.29 MB per layer = 64 MB total
- At 2 TB/s: 32 us saved -- this is the SINGLE LARGEST bandwidth saving.

**Complexity:** VERY HIGH. Requires CUTLASS 3.x persistent kernel with inter-CTA communication.

**Priority: HIGH value, HIGH difficulty.** This is the most impactful T>1 fusion but requires deep CUTLASS expertise. Recommend implementing as a CUTLASS EVT "fused_two_gemm" pattern.

### 4.5 Fusion Opportunity F5: Cross-Layer Fusion (Pipeline Layer N Output into Layer N+1 Input)

**Current state:** The T=1 fused path already implements cross-layer fusion: each layer returns `(residual, mlp_out)`, and the next layer's fused norm+GEMV kernel reads both to compute `residual + mlp_out` inline. This is a well-optimized pattern.

For T>1, the scratch double-buffering system (`residual_a/b`, `down_a/b`) achieves zero-copy cross-layer communication. The `fused_residual_rmsnorm_f16_into` function reads the previous layer's `down` buffer and the current `residual` buffer.

**Remaining opportunity:** The O-projection output from layer N is stored to HBM and read back by the same layer's post-attention norm. With the Down-proj epilogue residual fusion (F4b), layer N's Down output would already include the residual add, and the next layer's pre-attention norm would read a single buffer instead of two.

**Deeper cross-layer fusion -- overlapping layers on different SM clusters:**
On Hopper (SM90), the hardware supports up to 16 SM clusters. In principle, layer N could begin its attention phase on cluster 0-7 while layer N-1 is completing its MLP phase on cluster 8-15. This requires:
- Persistent kernels that stay resident across layers
- Inter-cluster communication via L2 or global memory
- Careful synchronization (named barriers or atomics)

**HBM savings from full cross-layer pipelining:**
- Each layer boundary currently materializes `residual[T, hidden]` + `mlp_out[T, hidden]` = 2 * T * hidden * 2 bytes
- With pipelining: only `normed[T, hidden]` crosses the boundary via L2/smem
- Savings: T * hidden * 2 bytes per layer = 128 * 1536 * 2 = 393 KB per layer

**Priority: LOW for T=1 (already done), MEDIUM for T>1 persistent kernel approach.**

### 4.6 Fusion Opportunity F6: Bias + RoPE Fusion for T>1

**Current state (T>1):** Bias add is 3 separate kernel launches (Q, K, V biases for split layout). RoPE is another separate launch. KV cache write is yet another.

**Proposed:** Fuse all three: bias add + RoPE + KV cache write into a single kernel. This is a direct extension of `fused_rope_cache_f16_kernel` that adds bias application before RoPE rotation.

**HBM savings:**
- Eliminated: qkv_dim * T * 2 bytes (bias intermediate not materialized)
- T=128: 128 * 2048 * 4 = 1.05 MB per layer = 29 MB total (read+write)

**Microseconds saved:** 29 MB / 2 TB/s = 14.5 us + 3 launches * 5 us = 29.5 us per step

**Register pressure:** Minimal. Bias add is one fmadd per element. RoPE needs 2 sin/cos values per half-dim thread. All fit in registers with the existing `fused_rope_cache` pattern.

**Shared memory:** None required (all elementwise).

**JIT compiler changes:** Add `FusionOp::BiasAdd` before `FusionOp::RoPE` in the `can_fuse` rules (already allowed by the IR). New kernel template `fused_bias_rope_cache.cu.template`. The `matcher.rs` already recognizes `BiasAdd + RoPE` as fusible.

**Priority: HIGH.** Low complexity, moderate savings, eliminates 3-4 kernel launches per layer.

### 4.7 Fusion Opportunity F7: RMSNorm Pre-computation for GEMM Prologue (Hybrid)

**Current state (T>1):** `fused_residual_rmsnorm` computes the full normalized output and writes it to HBM. The subsequent GEMM reads it back.

**Proposed hybrid approach:** Split into:
1. **Tiny kernel:** Compute `rms_scale[T]` only (T floats). Grid=(T,1,1), Block=(256,1,1). Each block reduces one row. Output: T-element scale vector.
2. **Modified GEMM:** CUTLASS mainloop loads tiles from residual (not normed), applies `x[m][k] * weight[k] * rms_scale[m]` inline in the A-operand load path.

This eliminates the normed intermediate entirely while keeping the GEMM's memory access pattern optimal.

**HBM savings:** T * hidden * 4 bytes (normed buffer eliminated)
- T=128: 128 * 1536 * 4 = 786 KB per fusion site, 2 sites per layer
- 28 layers: 44 MB per step, 22 us saved

**Register pressure:** LOW. The `rms_scale` is 1 float per row, broadcast across K dimension. Can be stored in a register loaded once per tile row.

**Shared memory:** The tiny pre-kernel needs hidden/256 iterations of shared-memory reduction. ~6 KB. The GEMM itself has no additional smem requirement.

**JIT compiler changes:** New CUTLASS template with custom A-operand transform. New IR node `FusionOp::ScaledLoad { scale_per_row: bool }`. The compiler emits two kernels: the scale pre-compute and the modified GEMM.

**Priority: MEDIUM-HIGH.** Clean separation of concerns, moderate implementation complexity, significant savings at high T.

### 4.8 Fusion Opportunity F8: Deinterleave QKV + RoPE + Bias Fusion (T>1)

**Current state (T>1):** When using fused QKV weight, the GEMM produces interleaved output `[T, qkv_dim]` that must be deinterleaved to `[T*Q, T*K, T*V]`. This is a separate kernel, followed by separate bias and RoPE kernels.

**Proposed:** Fuse deinterleave + bias add + RoPE + KV cache write into a single kernel. Each thread handles one element: reads from interleaved position, applies bias, applies RoPE (for Q/K), writes to the correct split position. For K/V, also writes to KV cache.

**HBM savings:**
- Eliminated: qkv_dim * T * 2 bytes (deinterleaved intermediate) + qkv_dim * T * 2 bytes (biased intermediate)
- T=128: 128 * 2048 * 8 = 2.1 MB per layer = 58.7 MB total

**Microseconds saved:** 29 us (bandwidth) + 20 us (4 launches) = ~49 us per step

**Implementation:** Straightforward elementwise kernel with index remapping. The thread grid is `(T * qkv_dim / 256, 1, 1)`. Each thread computes:
```
src_idx = deinterleave_map(global_idx)  // interleaved -> split
val = qkv_interleaved[src_idx] + bias[src_idx % qkv_dim]
if is_q_or_k(src_idx): val = apply_rope(val, ...)
if is_k_or_v(src_idx): cache_write(val, slot_mapping, ...)
output[global_idx] = val
```

**JIT compiler changes:** New fusion pattern in `matcher.rs`: `[QKVGemm, BiasAdd, RoPE, CacheWrite]`. New kernel template.

**Priority: HIGH.** Eliminates 4 kernel launches and 2 HBM round-trips per layer. Moderate complexity.

## 5. Register Pressure Constraints on Deeper Fusion

### 5.1 SM90 Register Budget

Each SM90 SM has 65,536 32-bit registers. At 256 threads per block and occupancy target of 2 blocks per SM, each thread gets 128 registers.

Current fused kernel register usage (estimated from code analysis):

| Kernel | Registers/Thread | Occupancy | Bottleneck |
|--------|-----------------|-----------|------------|
| `fused_cute_add_norm_qkv_gemv` | ~64 | 4 blocks/SM | smem (hidden*4B) |
| `fused_cute_silu_down_gemv` | ~48 | 4 blocks/SM | none |
| `fused_rope_cache_f16` | ~24 | 8 blocks/SM | none |
| `fused_residual_rmsnorm_f16` | ~32 | 4 blocks/SM | smem (hidden*4B) |
| `fused_cute_oproj_add_norm_gateup_gemv` | ~96 | 2 blocks/SM | L2 bandwidth |

### 5.2 Register Pressure for Proposed Fusions

| Proposed Fusion | Est. Registers | Impact |
|-----------------|---------------|--------|
| F1: Norm+GEMM prologue | ~48 extra (rms_scale, norm_weight tile) | Reduces GEMM occupancy from 4 to 2-3 waves |
| F4c: GateUp+SiLU+Down megafusion | ~128+ (two weight tiles simultaneously) | Exceeds budget, must use register spilling |
| F6: Bias+RoPE+cache | ~32 (trivial) | No impact |
| F7: RMS pre-compute + scaled GEMM | ~4 extra (1 float scale) | Negligible impact |
| F8: Deinterleave+bias+RoPE+cache | ~32 | No impact |

**Key constraint:** Any fusion involving two GEMV/GEMM operations (F4c) will exceed the register budget. The IR's `can_fuse` rule correctly prevents Gemv->Gemv fusion. For GEMM->GEMM fusion, the CUTLASS persistent kernel approach uses register spilling to L1 cache, which adds ~2 cycles per spill but is acceptable if the HBM savings dominate.

## 6. Shared Memory Requirements and Occupancy Tradeoffs

### 6.1 Current Shared Memory Usage

SM90 provides 228 KB of shared memory per SM (configurable L1/smem split). Default CUTLASS configuration uses 164 KB for GEMM pipelining.

| Kernel | Shared Memory | Blocks/SM | Notes |
|--------|--------------|-----------|-------|
| `fused_cute_add_norm_qkv_gemv` | hidden*4 + 32 = 6,176 B | 36 (limited by smem) | Redundant norm per block |
| `fused_cute_add_norm_gateup_gemv` | hidden*4 + 32 = 6,176 B | 36 | Same |
| `fused_cute_silu_down_gemv` | 32 B | unlimited (reg limited) | Minimal smem |
| `fused_residual_rmsnorm_f16` | hidden*4 = 6,144 B | 37 | Norm reduction |
| `paged_attention_v2_f16kv` | block_size*4 + head_dim*4 + warps*4 = ~2,624 B | 86 | Attention scratch |
| CUTLASS GEMM (128x128x64) | ~164 KB | 1 per SM | Pipelining buffers |

### 6.2 Shared Memory for Proposed Fusions

| Proposed Fusion | Additional Smem | Total Smem | Occupancy Impact |
|-----------------|----------------|------------|------------------|
| F1: Norm+GEMM prologue | +hidden*4 = 6 KB | 170 KB | Drops from 1 to 1 block/SM (already at limit) |
| F4a: QKV+bias epilogue | +0 (bias fits in registers) | 164 KB | None |
| F4b: Down+residual epilogue | +0 (residual fits in registers) | 164 KB | None |
| F6: Bias+RoPE+cache | 0 (all in registers) | 0 | None |
| F7: Scaled GEMM | +T*4 bytes scale vector | 164 KB + ~512 B | None |
| F8: Deinterleave+bias+RoPE | 0 | 0 | None |

The critical observation: **elementwise fusions (F6, F8) have zero shared memory overhead** while GEMM prologue/epilogue fusions are constrained by the existing CUTLASS pipelining budget.

## 7. HBM Bandwidth Analysis Summary

### 7.1 Per-Step Bandwidth Savings (T=128, Qwen2.5-1.5B, 28 Layers)

| Fusion | HBM Bytes Saved per Layer | Total (28L) | At 2 TB/s |
|--------|--------------------------|-------------|-----------|
| F1: Norm+GEMM prologue (x2 sites) | 1.57 MB | 44 MB | 22 us |
| F2: Attn+Residual+Norm GEMM epilogue | 786 KB | 22 MB | 11 us |
| F4a: QKV+bias epilogue | 524 KB | 14.7 MB | 7.3 us |
| F4b: Down+residual epilogue | 393 KB | 11 MB | 5.5 us |
| F4c: GateUp+SiLU+Down megafusion | 2.29 MB | 64 MB | 32 us |
| F6: Bias+RoPE+cache | 1.05 MB | 29 MB | 14.5 us |
| F7: RMS pre-compute + scaled GEMM (x2) | 1.57 MB | 44 MB | 22 us |
| F8: Deinterleave+bias+RoPE+cache | 2.1 MB | 58.7 MB | 29 us |
| **All fusions combined** | | **~287 MB** | **~143 us** |

### 7.2 Kernel Launch Savings (T=128, 28 Layers)

| Fusion | Launches Saved per Layer | Total (28L) | At 5 us/launch |
|--------|--------------------------|-------------|----------------|
| F1: Norm+GEMM prologue (x2) | 2 | 56 | 280 us |
| F4a: QKV+bias epilogue | 1-3 | 28-84 | 140-420 us |
| F4b: Down+residual epilogue | 0 (folded into existing GEMM) | 0 | 0 |
| F6: Bias+RoPE+cache | 3-4 | 84-112 | 420-560 us |
| F8: Deinterleave+bias+RoPE+cache | 4 | 112 | 560 us |
| **All fusions combined** | | **280-364** | **1,400-1,820 us** |

## 8. Arithmetic Intensity Analysis

Arithmetic intensity = FLOPs / bytes transferred. Higher is better (more compute per byte).

| Operation | FLOPs | Bytes | AI (FLOP/B) | Bound |
|-----------|-------|-------|-------------|-------|
| RMSNorm (T=128, hidden=1536) | 128*1536*5 = 983K | 128*1536*6 = 1.18M | 0.83 | Memory |
| cuBLAS GEMM QKV (128x2048x1536) | 128*2048*1536*2 = 805M | (128*1536 + 2048*1536 + 128*2048)*2 = 7.6M | 106 | Compute |
| SiLU*Mul (T=128, I=8960) | 128*8960*4 = 4.6M | 128*8960*6 = 6.9M | 0.67 | Memory |
| Fused Norm+QKV GEMM (proposed) | 983K + 805M = 806M | (128*1536*2 + 2048*1536*2 + 128*2048*2) = 7.0M | 115 | Compute |

**Key insight:** Fusing memory-bound operations (RMSNorm, SiLU, bias add) into compute-bound operations (GEMM) increases arithmetic intensity without changing the compute-bound nature. The GEMM still dominates runtime, but the overhead operations become "free" hidden within the GEMM's compute time.

## 9. JIT Compiler Changes Required

### 9.1 IR Extensions (`ir.rs`)

Current `FusionOp` enum needs these additions:

```rust
pub enum FusionOp {
    // Existing...
    
    // New for GEMM prologue/epilogue fusions
    GemmPrologue { norm: bool, residual_add: bool, scale_per_row: bool },
    GemmEpilogue { bias_add: bool, residual_add: bool, activation: Option<Activation> },
    Deinterleave { q_dim: usize, kv_dim: usize },
    CacheWrite,
    
    // GEMM types (currently only Gemv for M=1)
    Gemm,  // cuBLAS/CUTLASS GEMM for M>1
}

pub enum Activation {
    SiLU,
    GeLU,
}
```

### 9.2 Matcher Changes (`matcher.rs`)

New patterns to recognize:

```rust
// F6: Bias + RoPE + CacheWrite
if cfg.has_qkv_bias && contains_seq(ops, &[Op::BiasAdd, Op::RoPE, Op::CacheWrite]) {
    groups.push(/* fused_bias_rope_cache */);
}

// F8: Deinterleave + BiasAdd + RoPE + CacheWrite (T>1 only)  
if !pattern.is_prefill && contains_seq(ops, &[Op::QKVGemm, Op::BiasAdd, Op::RoPE, Op::CacheWrite]) {
    groups.push(/* fused_deinterleave_bias_rope_cache */);
}

// F4b: DownGemm + ElemAdd (epilogue fusion)
if contains_seq(ops, &[Op::DownGemm, Op::ElemAdd]) {
    groups.push(/* cutlass_down_residual */);
}
```

### 9.3 Codegen Changes (`codegen.rs`)

New `FusionPattern` variants:

```rust
enum FusionPattern {
    // Existing...
    BiasRoPECacheWrite,        // F6
    DeinterleaveBiasRoPECache, // F8
    GemmWithNormPrologue,      // F1
    GemmWithResidualEpilogue,  // F4b
}
```

Each needs a corresponding `emit_*` function generating CUDA C source.

### 9.4 Template Engine Changes (`compiler.rs`)

New templates needed:

```rust
templates.insert("fused_bias_rope_cache", include_str!("templates/fused_bias_rope_cache.cu.template"));
templates.insert("fused_deinterleave_bias_rope_cache", include_str!("templates/fused_deinterleave_bias_rope_cache.cu.template"));
templates.insert("cutlass_down_residual", include_str!("templates/cutlass_down_residual.cu.template"));
templates.insert("cutlass_norm_gemm_prologue", include_str!("templates/cutlass_norm_gemm_prologue.cu.template"));
```

### 9.5 Dispatch Changes (`dispatch.rs`)

New dispatch methods on `FusedLayerExecutor`:

```rust
fn try_fused_bias_rope_cache(...) -> Option<Result<()>>;
fn try_fused_deinterleave_bias_rope_cache(...) -> Option<Result<GpuBuffer>>;
```

### 9.6 Verification Changes (`verify.rs`)

Each new fused kernel needs a verification test that compares against the unfused reference. The existing `compare_outputs` function with `F16_TOLERANCE = 1e-2` is sufficient, but the Norm+GEMM prologue fusion may need `F16_STRICT_TOLERANCE = 1e-3` due to the different reduction order.

## 10. Priority Ranking

Ranked by estimated_speedup / implementation_effort:

| Rank | Fusion | Speedup (T=128) | Effort | Bang/Buck |
|------|--------|-----------------|--------|-----------|
| **1** | **F6: Bias+RoPE+cache** | 30 us + 420-560 us launch | 1-2 days | **HIGHEST** |
| **2** | **F8: Deinterleave+bias+RoPE+cache (T>1)** | 29 us + 560 us launch | 2-3 days | HIGH |
| **3** | **F4b: Down+residual CUTLASS epilogue** | 5.5 us + 0 launch (folded) | 1 day (clone oproj_residual) | HIGH |
| **4** | **F7: RMS pre-compute + scaled GEMM** | 22 us + 0 launch | 3-5 days | MEDIUM |
| **5** | **F1: Norm+GEMM prologue** | 22 us + 280 us launch | 5-7 days | MEDIUM |
| **6** | **F4a: QKV+bias CUTLASS epilogue (T>1)** | 7 us + 140 us launch | 2-3 days | MEDIUM |
| **7** | **F2: Attn+Residual+Norm GEMM epilogue** | 11 us + 5 us launch | 7-10 days | LOW |
| **8** | **F4c: GateUp+SiLU+Down megafusion** | 32 us (largest!) | 14+ days | LOW |
| **9** | F3: RoPE+cache+attention | Negligible | - | SKIP |

## 11. Expected Total Speedup from All Fusions Combined

### 11.1 T=1 Decode Path

The T=1 path is already heavily fused (7-8 launches per layer). Remaining gains:

| Source | Savings |
|--------|---------|
| F6: Bias+RoPE+cache (already fused, but bias separate) | ~5 us/layer (1 launch) |
| Mega-fused oproj+add+norm+gateup (L2-limited, small models only) | ~10 us/layer |
| **Total T=1 additional savings** | **140-420 us per step** |
| **As % of current T=1 latency (~10 ms)** | **1.4-4.2%** |

The T=1 path has diminishing returns from further fusion.

### 11.2 T>1 Batched Decode Path (T=128)

This is where the largest gains are:

| Source | Bandwidth Savings | Launch Savings | Total |
|--------|-------------------|----------------|-------|
| F6: Bias+RoPE+cache | 14.5 us | 420-560 us | ~500 us |
| F8: Deinterleave+bias+RoPE+cache | 29 us | 560 us | ~590 us |
| F4b: Down+residual epilogue | 5.5 us | 0 | 5.5 us |
| F7: Scaled GEMM (x2 sites) | 22 us | 0 | 22 us |
| F1: Norm+GEMM prologue (x2) | 22 us | 280 us | 302 us |
| F4a: QKV+bias epilogue | 7.3 us | 140-420 us | ~220 us |
| **Total** | **~100 us** | **~1,400-1,820 us** | **~1,640 us** |

Current T=128 decode step time: approximately 3.2 ms (128 tokens at 6,360 tok/s from existing benchmarks, adjusted for per-step overhead).

**Expected speedup from all fusions: ~1.64 ms / 3.2 ms = ~51% of compute time saved.**

This is overly optimistic because launch savings depend on kernel launches not being hidden by GPU compute. With CUDA graphs, launch overhead is already amortized. The more realistic estimate:

- **Without CUDA graphs:** 1,400+ us from launch overhead elimination = **~44% improvement**
- **With CUDA graphs (launches already hidden):** ~100 us from bandwidth savings = **~3% improvement**

### 11.3 Combined Expected Impact

| Scenario | Current | After All Fusions | Improvement |
|----------|---------|-------------------|-------------|
| T=1 decode, no graph | 10 ms | 9.6-9.9 ms | 1-4% |
| T=128 decode, no graph | ~3.2 ms | ~1.6-2.6 ms | 19-50% |
| T=128 decode, with CUDA graph | ~2.8 ms | ~2.7 ms | 3-5% |
| T=128 prefill | ~8 ms | ~7.0 ms | 12-15% |

**Bottom line:** Kernel fusion provides the largest gains when **CUDA graph capture is not possible** (variable batch sizes, prefill, mixed-phase execution). When CUDA graphs are active, the launch overhead is already eliminated and fusion provides only the bandwidth savings -- which are meaningful but modest.

The recommended implementation order prioritizes **F6 and F8** (bias+RoPE+cache fusions) because they are:
1. Simple elementwise kernels with no register/smem pressure
2. Beneficial in ALL execution paths (T=1 and T>1, with or without CUDA graphs)
3. Implementable in 1-3 days each
4. Eliminate 3-4 kernel launches per layer even when CUDA graphs are not used

---

### Critical Files for Implementation
- `/Users/andy/rvllm/crates/rvllm-fusion/src/matcher.rs` -- Pattern matching engine where new fusion groups (F6, F8, F4b) must be registered with their cost estimates
- `/Users/andy/rvllm/crates/rvllm-fusion/src/ir.rs` -- Fusion IR that needs new FusionOp variants (CacheWrite, Deinterleave, GemmEpilogue) and updated can_fuse rules
- `/Users/andy/rvllm/crates/rvllm-model-runner/src/gpu_layer.rs` -- The 2,202-line transformer layer implementation where every kernel launch decision is made and where new fused kernel dispatch must be wired
- `/Users/andy/rvllm/crates/rvllm-fusion/src/codegen.rs` -- CUDA C code generation that needs new emit functions for each proposed fusion pattern
- `/Users/andy/rvllm/kernels/fused_rope_cache.cu` -- The existing fused RoPE+cache kernel that serves as the template for F6 (bias+RoPE+cache) and F8 (deinterleave+bias+RoPE+cache)
