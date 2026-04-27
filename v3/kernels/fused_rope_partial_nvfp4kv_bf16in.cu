// Partial RoPE + NVFP4 paged-KV-cache write (Gemma 4).
//
// Sibling of `fused_rope_partial_fp8kv.cu` that stores KV as packed
// 4-bit values + per-16-element E4M3 microscales. Same RoPE math and
// split-half pairing as the FP8 kernel; only the quantise + write
// path changes.
//
// Differences vs the FP8 sibling:
//   * Thread-per-element (blockDim.x = head_dim) instead of
//     thread-per-pair — we need 16 cooperating threads to compute
//     one NVFP4 microscale via warp-shuffle reduction.
//   * Each thread handles both the "lo" dim (tid < half_head) and
//     the "hi" dim (tid >= half_head) of the split-half RoPE layout
//     by reading its own + its pair's input.
//   * Output fan-out:
//       key_cache_packed[cache_off + tid/2]   — half byte per thread
//       key_cache_scale [cache_off_scale + tid/16] — 1 E4M3 per 16
//     Same for V cache, plus Q FP8 output (unchanged; Q stays FP8
//     because the MMA input side of FA2 is still FP8).
//
// head_dim must be a multiple of 16 (NVFP4 block size). Gemma 4
// head_dim ∈ {256, 512} both satisfy.
//
// Q output (`q_fp8_out`) stays FP8 E4M3 with per-tensor scale.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../../kernels/nvfp4_utils.cuh"
// === HADAMARD ROTATION ===
#include "../../kernels/hadamard.cuh"
// === END HADAMARD ROTATION ===

using rvllm_nvfp4::fp4_encode;
using rvllm_nvfp4::fp4_encode_stochastic;

// Reduce the absolute-value peak across a 16-lane group within a warp.
// `lane_id` is `threadIdx.x & 31`; the group is determined by
// `lane_id >> 4` (upper 16 or lower 16 lanes of the warp).
__device__ __forceinline__ float block16_peak_abs(float v) {
    float a = fabsf(v);
    // Butterfly within 16-lane group: xor-shuffle with 8, 4, 2, 1.
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    }
    return a;
}

// Reduce the absolute-value SUM across the 16-lane group.
__device__ __forceinline__ float block16_sum_abs(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xFFFFFFFFu, a, off);
    }
    return a;
}

// Reduce an arbitrary f32 SUM across the 16-lane group.
__device__ __forceinline__ float block16_sum(float v) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// Find second-largest abs value in the 16-lane group by masking out
// the peak's lane(s) and reducing again. If multiple lanes hold the
// peak (ties), the second reduce still returns the peak magnitude —
// acceptable for our use since ties mean the distribution doesn't
// have a single outlier.
__device__ __forceinline__ float block16_second_largest_abs(
    float v, float peak)
{
    float a = fabsf(v);
    // Mask: if this lane IS the peak, contribute -1 so max ignores it.
    float contrib = (a >= peak - 1e-9f) ? -1.0f : a;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        contrib = fmaxf(contrib, __shfl_xor_sync(0xFFFFFFFFu, contrib, off));
    }
    // If all 16 lanes tied at peak, the mask knocks everything to -1 →
    // fall back to peak.
    return (contrib < 0.0f) ? peak : contrib;
}

// --- Scale-policy encoding (kernel arg `scale_policy`) ---
//   0 = amax6   : peak / 6.0  (OCP-baseline, range-preserving — outlier-insensitive)
//   1 = mse    : blockwise MSE search over 4 candidates:
//                  { peak/6, peak/4, second_largest/6, second_largest/4 }
//                e4m3-rounded, picks the scale minimizing sum-squared
//                dequant error over the 16-element block.
// Policies 0 and 1 produce DIFFERENT on-cache byte streams (different
// scale bytes + different nibbles). Re-run through the full KV cache
// fill when switching.
#define RVLLM_NVFP4_POLICY_AMAX6 0
#define RVLLM_NVFP4_POLICY_MSE   1

// Given a candidate scale, compute this lane's squared error under
// the nearest-e2m1 quantize → dequantize roundtrip.
__device__ __forceinline__ float lane_quant_sse(float v, float scale) {
    if (scale <= 0.0f) return v * v;
    uint32_t nib = fp4_encode(v / scale);
    // Decode (inverse of fp4_encode used in the rest of the kernel).
    static constexpr float kTable[8] =
        {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = kTable[nib & 0x7u];
    float sign = ((nib >> 3) & 0x1u) ? -1.0f : 1.0f;
    float dq = sign * mag * scale;
    float err = v - dq;
    return err * err;
}

// Round raw-scale to E4M3 and back to f32, exactly mirroring what the
// production path does (cvt.rn.satfinite).
__device__ __forceinline__ float round_scale_e4m3(float s) {
    if (!(s > 0.0f)) return 0.0f;
    return float(__nv_fp8_e4m3(s));
}

extern "C"
__global__ void fused_rope_partial_nvfp4kv_bf16in_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    uint8_t*       __restrict__ key_cache_packed,   // [..., head_dim/2]
    uint8_t*       __restrict__ value_cache_packed, // [..., head_dim/2]
    __nv_fp8_e4m3* __restrict__ key_cache_scale,    // [..., head_dim/16]
    __nv_fp8_e4m3* __restrict__ value_cache_scale,  // [..., head_dim/16]
    const __half*  __restrict__ cos_table,
    const __half*  __restrict__ sin_table,
    const int*     __restrict__ positions,
    const int*     __restrict__ slot_mapping,
    const float*   __restrict__ q_scale_ptr,
    // === DYNAMIC NVFP4 Q SCALE ===
    // Per-(token, head) Q scale output. When non-null, the kernel
    // computes amax(|Q_post_rotation|) per (token, head_idx), derives
    // a fresh FP8 scale = max(amax/448, 1e-12), writes it here, and
    // quantizes Q with that dynamic scale. NVFP4 attention kernels
    // (decode + prefill) read the same `[num_tokens, num_heads]` f32
    // table to dequant Q. When null, falls back to the static scalar
    // `*q_scale_ptr` — backwards compatible with the pre-Hadamard
    // path. Required for `RVLLM_NVFP4_HADAMARD=1` because rotated Q
    // values can grow up to √D per channel and saturate the static
    // (typically 0.1) scalar.
    float*         __restrict__ q_scale_cache,
    // === END DYNAMIC NVFP4 Q SCALE ===
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int scale_policy,    // K scale policy (RVLLM_NVFP4_POLICY_*)
    int v_scale_policy,  // V scale policy (independent, for K/V split diagnostic)
    // === HADAMARD ROTATION ===
    // Per-layer ±1 sign vectors of length `head_dim` (i8 storage).
    // When BOTH pointers are non-null, the kernel applies signed
    // Walsh-Hadamard rotation R = H * diag(D) to Q post-RoPE pre-FP8-
    // quantize, and to K post-RoPE pre-NVFP4-quantize. When EITHER
    // hadamard_signs_q / hadamard_signs_k pointer is null, Q+K
    // rotation is bypassed.
    // V rotation is gated independently by `rotate_v` (when 1, V is
    // also rotated using `hadamard_signs_k`, the SAME R as K). The
    // companion kernel hadamard_unrotate_f16 then multiplies attn_out
    // by R^T before O-proj to recover the original P·V.
    const signed char* __restrict__ hadamard_signs_q,
    const signed char* __restrict__ hadamard_signs_k,
    int rotate_v,
    // === END HADAMARD ROTATION ===
    // === CYCLE 28 PRE-QUANT K F16 SIDECAR (codex pre-quant probe) ===
    // When non-null, kernel writes the exact f32 K value used for
    // NVFP4 packing (post-RoPE + post-optional-Hadamard) as f16 to
    // this buffer at offset (slot * num_kv_heads + head_idx) * head_dim
    // + tid. This bypasses the shadow-vs-primary cross-path comparison
    // ambiguity by giving the analyzer the EXACT input the quantizer
    // saw — apples-to-apples with my Python dequant of the packed nibbles
    // + scale.
    __half*        __restrict__ debug_k_prequant,
    // === END CYCLE 28 ===
    // === CYCLE 29 V PRE-QUANT SIDECAR ===
    __half*        __restrict__ debug_v_prequant,
    // === END CYCLE 29 ===
    // === CYCLE 31 STOCHASTIC ROUNDING (V only) ===
    int            stoch_round_v
    // === END CYCLE 31 ===
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    // === HADAMARD ROTATION ===
    // We KEEP all threads alive (no early return for tid>=head_dim)
    // when rotation is enabled, because FWHT requires every thread in
    // the block to participate in __syncthreads() and shared-memory
    // butterflies. Caller launches with blockDim.x == head_dim, so
    // tid < head_dim is always true at the launcher level.
    // === END HADAMARD ROTATION ===
    if (tid >= head_dim) return;

    // === HADAMARD ROTATION ===
    // Shared-memory scratch for FWHT. Sized for the largest Gemma 4
    // head_dim (=512). Allocated unconditionally (2 KiB) — well under
    // the per-block smem budget. Only used when rotation is enabled;
    // otherwise dead.
    __shared__ float s_hadamard[512];
    const bool hadamard_on =
        (hadamard_signs_q != nullptr) && (hadamard_signs_k != nullptr);
    // === END HADAMARD ROTATION ===

    const int half_head    = head_dim >> 1;
    const int half_rotary  = rotary_dim >> 1;
    const int group_id     = tid >> 4;            // which 16-block inside head_dim
    const int groups_per_head = head_dim >> 4;
    const int lane_in_group = tid & 15;

    const int pos = positions[token_idx];

    // Compute RoPE'd output value for my dim.
    auto rope_one = [&] (const __nv_bfloat16* in, int base) -> float {
        const bool is_lo = tid < half_head;
        const int pair   = is_lo ? tid + half_head : tid - half_head;
        const int freq   = is_lo ? tid : pair;

        float x_self = __bfloat162float(in[base + tid]);
        if (freq < half_rotary) {
            float x_pair = __bfloat162float(in[base + pair]);
            float cos_v  = __half2float(cos_table[pos * half_rotary + freq]);
            float sin_v  = __half2float(sin_table[pos * half_rotary + freq]);
            // Same matrix as the FP8 kernel:
            //   lo: x_lo*cos - x_hi*sin
            //   hi: x_lo*sin + x_hi*cos
            if (is_lo) return x_self * cos_v - x_pair * sin_v;
            else       return x_pair * sin_v + x_self * cos_v;
        }
        return x_self;  // pass-through (outside rotary range)
    };

    // ---- Q: stays FP8 per-tensor (MMA input side still fp8). ----
    if (head_idx < num_heads) {
        const int q_base = (token_idx * num_heads + head_idx) * head_dim;
        float v = rope_one(q_in, q_base);
        // === HADAMARD ROTATION ===
        // Apply R = H * diag(D) to Q post-RoPE, before FP8 quantize.
        // x_rot[tid] = sum_j H[tid,j] * D[j] * x[j], normalized by
        // 1/sqrt(head_dim). All threads in this block cooperate via
        // smem (block is per-(token, head_idx), so the smem buffer
        // exactly covers one Q vector of length head_dim).
        if (hadamard_on) {
            s_hadamard[tid] = v;
            __syncthreads();
            rvllm_hadamard::apply_signs_f32(
                s_hadamard, hadamard_signs_q, head_dim);
            rvllm_hadamard::fwht_inplace_f32(s_hadamard, head_dim);
            v = s_hadamard[tid];
            // NB: NO __syncthreads() here in the legacy path because
            // each thread reads its own slot. With dynamic Q scale we
            // reuse `s_hadamard` for the amax block-reduce below, so
            // the reduction sequence has its own sync.
        }
        // === END HADAMARD ROTATION ===

        // === DYNAMIC NVFP4 Q SCALE ===
        // Per-(token, head) dynamic Q scale: block-reduce amax over
        // `head_dim` threads, derive scale = max(amax/448, 1e-12),
        // thread 0 writes to `q_scale_cache[token_idx*num_heads + head_idx]`,
        // all threads quantize with 1/scale. When `q_scale_cache` is
        // null, fall back to the static scalar (legacy behaviour).
        float q_scale_inv;
        if (q_scale_cache != nullptr) {
            // Stage own value, warp+block reduce. Reuse s_hadamard as
            // scratch (size 512 floats, head_dim<=512, plenty of room).
            __syncthreads();  // ensure prior s_hadamard reads are done
            float my_abs = fabsf(v);
            // Warp-shuffle reduce within each 32-lane warp.
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                my_abs = fmaxf(my_abs,
                    __shfl_xor_sync(0xFFFFFFFFu, my_abs, off));
            }
            const int lane = tid & 31;
            const int warp = tid >> 5;
            // Stash one value per warp into smem.
            if (lane == 0) s_hadamard[warp] = my_abs;
            __syncthreads();
            const int num_warps = (head_dim + 31) >> 5;
            // First warp reduces across warps, broadcasts via slot 0.
            if (warp == 0) {
                float a = (lane < num_warps) ? s_hadamard[lane] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
                }
                if (lane == 0) {
                    float scale = fmaxf(a * (1.0f / 448.0f), 1e-12f);
                    s_hadamard[0] = scale;
                    q_scale_cache[token_idx * num_heads + head_idx] = scale;
                }
            }
            __syncthreads();
            float q_scale_dyn = s_hadamard[0];
            q_scale_inv = 1.0f / q_scale_dyn;
        } else {
            q_scale_inv = 1.0f / (*q_scale_ptr);
        }
        q_fp8_out[q_base + tid] = __nv_fp8_e4m3(v * q_scale_inv);
        // === END DYNAMIC NVFP4 Q SCALE ===
    }
    // === HADAMARD ROTATION ===
    // Sync between Q-rotation smem use and K-rotation smem use
    // below. Even when hadamard_on is false this is a no-op cost
    // (~1 cycle); when on, it ensures the smem buffer is safe to
    // reuse for K.
    __syncthreads();
    // === END HADAMARD ROTATION ===

    // ---- K, V: NVFP4-packed cache write. ----
    if (head_idx < num_kv_heads) {
        const int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int slot   = slot_mapping[token_idx];
        if (slot < 0) return;

        const int cache_off_bytes  = (slot * num_kv_heads + head_idx) * (head_dim >> 1);
        const int cache_off_scales = (slot * num_kv_heads + head_idx) * groups_per_head;

        // Helper to pack one (K or V) head.
        // === HADAMARD ROTATION ===
        // `apply_rotation` controls whether to apply Hadamard R to the
        // post-RoPE values before quantize. Set true ONLY for K when
        // hadamard_on; V is never rotated (would require rotating
        // O-proj weights, separate lift).
        // === END HADAMARD ROTATION ===
        auto quant_and_write = [&] (
            const __nv_bfloat16* in,
            uint8_t*       out_packed,
            __nv_fp8_e4m3* out_scales,
            bool           apply_rope,
            // === HADAMARD ROTATION ===
            bool           apply_rotation,
            // === END HADAMARD ROTATION ===
            int            policy,
            // === CYCLE 28 PRE-QUANT SIDECAR ===
            __half*        debug_prequant,
            // === END CYCLE 28 ===
            // === CYCLE 31 STOCHASTIC ROUNDING ===
            // Per-side gate. K stays deterministic (round-to-nearest)
            // because randomizing K would jitter softmax routing across
            // re-prefills of the same prompt. V uses stochastic when
            // gate is set — under attention's weighted sum the unbiased
            // noise tends to average out across long contexts.
            bool           stoch_round
            // === END CYCLE 31 ===
        ) {
            float v = apply_rope ? rope_one(in, k_base)
                                 : __bfloat162float(in[k_base + tid]);

            // === HADAMARD ROTATION ===
            // K rotation: same R as Q (orthogonal, so Q*K^T invariant).
            // The MSE scale policy below now picks scales over the
            // ROTATED K values — that's the intended Hadamard regime,
            // since rotation flattens outliers and MSE picks better
            // scales on the rotated distribution.
            if (apply_rotation) {
                __syncthreads();
                s_hadamard[tid] = v;
                __syncthreads();
                rvllm_hadamard::apply_signs_f32(
                    s_hadamard, hadamard_signs_k, head_dim);
                rvllm_hadamard::fwht_inplace_f32(s_hadamard, head_dim);
                v = s_hadamard[tid];
                __syncthreads();
            }
            // === END HADAMARD ROTATION ===

            // === CYCLE 28 PRE-QUANT SIDECAR ===
            // Write the exact f32 input the quantizer is about to consume
            // (post-RoPE + post-optional-Hadamard) as f16 to the debug
            // buffer. Same indexing as primary K cache: (slot * nkvh +
            // head) * head_dim + tid. Caller passes nullptr to disable.
            if (debug_prequant != nullptr) {
                debug_prequant[(slot * num_kv_heads + head_idx) * head_dim + tid]
                    = __float2half(v);
            }
            // === END CYCLE 28 ===

            // Per-16-block scale selection. Default path is the
            // range-preserving OCP baseline (peak/6). The MSE path
            // searches over 4 outlier-aware candidates and picks the
            // one that minimises sum-squared reconstruction error
            // over the 16-element block.
            float peak = block16_peak_abs(v);
            float scale_f32;
            if (policy == RVLLM_NVFP4_POLICY_MSE && peak > 0.0f) {
                float second = block16_second_largest_abs(v, peak);
                // 4 candidates (E4M3-rounded before scoring, to match
                // what the production path would actually store).
                float c0 = round_scale_e4m3(peak * (1.0f / 6.0f));
                float c1 = round_scale_e4m3(peak * (1.0f / 4.0f));
                float c2 = round_scale_e4m3(second * (1.0f / 6.0f));
                float c3 = round_scale_e4m3(second * (1.0f / 4.0f));
                // Per-lane squared error at each candidate, then
                // sum across the 16-lane group.
                float e0 = block16_sum(lane_quant_sse(v, c0));
                float e1 = block16_sum(lane_quant_sse(v, c1));
                float e2 = block16_sum(lane_quant_sse(v, c2));
                float e3 = block16_sum(lane_quant_sse(v, c3));
                // Pick min across {c0..c3}. Prefer c0 (= current
                // baseline) on ties to stay range-preserving when MSE
                // is indifferent.
                float best_e = e0;    float best_s = c0;
                if (e1 < best_e) { best_e = e1; best_s = c1; }
                if (e2 < best_e) { best_e = e2; best_s = c2; }
                if (e3 < best_e) { best_e = e3; best_s = c3; }
                scale_f32 = best_s;
            } else {
                // Baseline: peak / 6 (OCP NVFP4 default).
                scale_f32 = (peak > 0.0f) ? (peak * (1.0f / 6.0f)) : 0.0f;
            }
            __nv_fp8_e4m3 scale_e4m3 = __nv_fp8_e4m3(scale_f32);
            float scale = float(scale_e4m3);
            float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

            // First lane of each 16-group writes the scale.
            if (lane_in_group == 0) {
                out_scales[cache_off_scales + group_id] = scale_e4m3;
            }

            // Each thread encodes its own 4-bit nibble.
            // === CYCLE 31 STOCHASTIC ROUNDING ===
            // Seed = (slot, head, lane) packed — deterministic per
            // position so re-prefill of the same token reuses the same
            // random draw (otherwise prefix-cache would silently
            // diverge). Mixed via SplitMix32 inside fp4_encode_stochastic.
            uint32_t nibble;
            if (stoch_round) {
                uint32_t seed = static_cast<uint32_t>(slot) * 65521u
                              + static_cast<uint32_t>(head_idx) * 257u
                              + static_cast<uint32_t>(tid);
                nibble = fp4_encode_stochastic(v * inv_scale, seed);
            } else {
                nibble = fp4_encode(v * inv_scale);
            }
            // === END CYCLE 31 ===

            // Pair lanes (even, odd) combine into one byte: low = even,
            // high = odd. Use shuffle to pull the odd nibble to the
            // even lane, then write one byte per even thread.
            uint32_t odd_nibble = __shfl_down_sync(0xFFFFFFFFu, nibble, 1);
            if ((tid & 1) == 0) {
                uint8_t byte = static_cast<uint8_t>((odd_nibble << 4) | (nibble & 0xFu));
                out_packed[cache_off_bytes + (tid >> 1)] = byte;
            }
        };

        // === HADAMARD ROTATION ===
        // K rotated when hadamard_on (Q is rotated above by the same R,
        // so Q*K^T stays invariant). V is rotated when rotate_v != 0
        // (using same R as K). The post-attention `hadamard_unrotate_f16`
        // kernel must multiply attn_out by R^T before O-proj to recover
        // the original P·V; without that companion, O-proj sees rotated
        // input and the model output is wrong.
        const bool v_rotate_now = hadamard_on && (rotate_v != 0);
        quant_and_write(k_in, key_cache_packed,   key_cache_scale,
                        /*apply_rope=*/true,  /*apply_rotation=*/hadamard_on,
                        scale_policy, /*debug_prequant=*/debug_k_prequant,
                        /*stoch_round=*/false);
        quant_and_write(v_in, value_cache_packed, value_cache_scale,
                        /*apply_rope=*/false, /*apply_rotation=*/v_rotate_now,
                        v_scale_policy, /*debug_prequant=*/debug_v_prequant,
                        /*stoch_round=*/(stoch_round_v != 0));
        // === END HADAMARD ROTATION ===
    }
}
