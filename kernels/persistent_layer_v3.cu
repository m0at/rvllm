// Persistent transformer layer kernel v3 for M=1 decode.
// Executes an ENTIRE transformer layer in ONE kernel launch.
//
// Improvements over v2 (persistent_layer_v2.cu):
//   - Eliminated 32KB weight double buffer from shared memory
//   - Vectorized GEMV with float4 global loads (bandwidth-optimal for M=1)
//   - BC reduced from 64 to 32 for attention tiles
//   - ~17KB shared memory -> up to 8 blocks/SM -> 1024 blocks total
//   - Non-cooperative launch (1024 <= 132 * 8 = 1056)
//   - Target: 1284 GB/s bandwidth (86% of 1.5 TB/s H100)
//
// Phases (same as v2):
//   1: Residual add + RMSNorm + QKV GEMV (vectorized)
//   2: QKV bias + RoPE + KV cache write (parallel across heads)
//   3: Split-KV GQA attention (all blocks, combine step, BC=32)
//   4: O-proj GEMV (vectorized)
//   5: Residual add + RMSNorm + GateUp GEMV (vectorized)
//   6: Fused SiLU*mul + Down GEMV (vectorized, no shared buffers)
//
// Grid: (1024, 1, 1), Block: (256, 1, 1)
// Compile: nvcc -cubin -arch=sm_90 -O3 --use_fast_math

#include <cuda_fp16.h>
#include <cstdint>
#include <float.h>

// ---- Config ----
#define PLV3_THREADS 256
#define PLV3_WARPS   (PLV3_THREADS / 32)
#define PLV3_RPW     2
#define PLV3_BLOCK_N (PLV3_WARPS * PLV3_RPW)  // 16

// ---- DAG sync phases ----
#define PLV3_SYNC_QKV    0
#define PLV3_SYNC_ROPE   1
#define PLV3_SYNC_ATTN   2
#define PLV3_SYNC_OPROJ  3
#define PLV3_SYNC_LOAD5  4
#define PLV3_SYNC_GATEUP 5
#define PLV3_NUM_SYNCS   6

// ---- Attention config ----
#define PLV3_FA_BC    32
#define PLV3_FA_HPG   8
#define PLV3_FA_SCORE_STRIDE (PLV3_FA_BC + 1)  // 33 for bank avoidance

// ============================================================================
// Sync primitives
// ============================================================================

__device__ __forceinline__ void plv3_signal(int* flags, int phase) {
    if (threadIdx.x == 0) {
        asm volatile(
            "{ .reg .u32 tmp; atom.add.release.gpu.u32 tmp, [%0], 1; }"
            :: "l"((unsigned int*)&flags[phase]) : "memory"
        );
    }
}

__device__ __forceinline__ void plv3_wait(int* flags, int phase, int expected) {
    if (threadIdx.x == 0) {
        int val;
        do {
            asm volatile(
                "ld.global.acquire.sys.u32 %0, [%1];"
                : "=r"(val) : "l"((unsigned int*)&flags[phase]) : "memory"
            );
        } while (val < expected);
    }
    __syncthreads();
}

// ============================================================================
// cp.async primitives (used only for attention KV tile loads)
// ============================================================================

__device__ __forceinline__ void plv3_cp_async_16b(void* smem, const void* gmem) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem) : "memory"
    );
}

__device__ __forceinline__ void plv3_cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void plv3_cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// ============================================================================
// Math helpers
// ============================================================================

__device__ __forceinline__ float plv3_silu(float x) {
    return x * __frcp_rn(1.0f + expf(-x));
}

__device__ __forceinline__ float plv3_warp_reduce(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float plv3_warp_xor_reduce(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

template <bool Profile>
__device__ __forceinline__ void plv3_profile_stamp(int* sync_flags, int slot);

template <>
__device__ __forceinline__ void plv3_profile_stamp<false>(int*, int) {}

template <>
__device__ __forceinline__ void plv3_profile_stamp<true>(int* sync_flags, int slot) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long* clocks =
            reinterpret_cast<unsigned long long*>(sync_flags + PLV3_NUM_SYNCS);
        clocks[slot] = clock64();
    }
}

// ============================================================================
// Phase 1/5: Residual add + RMSNorm into shared memory (f32)
// All blocks compute independently; block 0 writes residual to global memory.
// ============================================================================

__device__ void plv3_add_rmsnorm(
    float* s_normed,
    float* s_scratch,
    __half* __restrict__ residual_out,
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,   // NULL for layer 0
    const __half* __restrict__ norm_w,
    float eps, int hidden_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int h2 = hidden_size / 2;

    float local_ss = 0.0f;

    if (prev_mlp != nullptr) {
        const half2* r2 = (const half2*)prev_residual;
        const half2* m2 = (const half2*)prev_mlp;
        for (int i = tid; i < h2; i += PLV3_THREADS) {
            half2 a = r2[i], b = m2[i];
            float v0 = __half2float(a.x) + __half2float(b.x);
            float v1 = __half2float(a.y) + __half2float(b.y);
            s_normed[i * 2]     = v0;
            s_normed[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
    } else {
        const half2* r2 = (const half2*)prev_residual;
        for (int i = tid; i < h2; i += PLV3_THREADS) {
            half2 a = r2[i];
            float v0 = __half2float(a.x), v1 = __half2float(a.y);
            s_normed[i * 2]     = v0;
            s_normed[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
    }

    local_ss = plv3_warp_reduce(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < PLV3_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = plv3_warp_reduce(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Block 0 writes pre-norm residual to global memory
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += PLV3_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    // Apply norm weights in-place in shared memory
    for (int i = tid; i < hidden_size; i += PLV3_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_w[i]) * rms_scale;
    __syncthreads();
}

// ============================================================================
// Bandwidth-optimal vectorized GEMV (replaces tensor-core GEMV)
// No shared memory for weights -- reads directly from global memory.
// Each warp processes RPW rows at a time, one row sequentially.
// Uses float4 (16-byte) loads for maximum memory bandwidth.
// ============================================================================

__device__ void plv3_gemv(
    __half* __restrict__ output,
    const float* __restrict__ s_input,  // normed vector in shared mem
    const __half* __restrict__ weight,  // [out_dim, in_dim] row-major
    int out_dim, int in_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int k4 = in_dim / 8;  // number of float4-sized chunks (8 f16 = 16 bytes)

    for (int block_base = blockIdx.x * PLV3_BLOCK_N;
         block_base < out_dim;
         block_base += gridDim.x * PLV3_BLOCK_N)
    {
        int warp_base = block_base + warp_id * PLV3_RPW;
        if (warp_base >= out_dim) continue;
        int valid_rows = min(PLV3_RPW, out_dim - warp_base);

        // Process one row at a time with full warp cooperation
        for (int r = 0; r < valid_rows; r++) {
            const __half* row = weight + (long long)(warp_base + r) * in_dim;
            float dot = 0.0f;

            for (int c = lane_id; c < k4; c += 32) {
                // Load 8 f16 values as float4 (16 bytes, coalesced)
                float4 v = *reinterpret_cast<const float4*>(&row[c * 8]);
                __half* h = reinterpret_cast<__half*>(&v);
                float inp0 = s_input[c * 8 + 0];
                float inp1 = s_input[c * 8 + 1];
                float inp2 = s_input[c * 8 + 2];
                float inp3 = s_input[c * 8 + 3];
                float inp4 = s_input[c * 8 + 4];
                float inp5 = s_input[c * 8 + 5];
                float inp6 = s_input[c * 8 + 6];
                float inp7 = s_input[c * 8 + 7];
                dot += __half2float(h[0]) * inp0 + __half2float(h[1]) * inp1
                     + __half2float(h[2]) * inp2 + __half2float(h[3]) * inp3
                     + __half2float(h[4]) * inp4 + __half2float(h[5]) * inp5
                     + __half2float(h[6]) * inp6 + __half2float(h[7]) * inp7;
            }

            // Handle remainder if in_dim is not divisible by 8
            int rem_start = k4 * 8;
            for (int k = rem_start + lane_id; k < in_dim; k += 32) {
                dot += __half2float(weight[(long long)(warp_base + r) * in_dim + k]) * s_input[k];
            }

            dot = plv3_warp_xor_reduce(dot);
            if (lane_id == 0) output[warp_base + r] = __float2half(dot);
        }
    }
}

// ============================================================================
// Phase 2: QKV bias + RoPE + KV cache write (parallel across heads)
// ============================================================================

__device__ void plv3_rope_cache(
    __half* __restrict__ qkv,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ qkv_bias,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int num_heads, int num_kv_heads, int head_dim,
    int q_dim, int kv_dim
) {
    const int tid = threadIdx.x;
    const int bid = (int)blockIdx.x;
    const int half_dim = head_dim / 2;
    const int pos = positions[0];
    const int slot = slot_mapping[0];
    const int total_heads = num_heads + 2 * num_kv_heads;

    if (bid < num_heads) {
        // Q head: bias + RoPE
        __half* q_head = qkv + bid * head_dim;
        if (qkv_bias) {
            const __half* bias = qkv_bias + bid * head_dim;
            for (int i = tid; i < head_dim; i += PLV3_THREADS)
                q_head[i] = __float2half(__half2float(q_head[i]) + __half2float(bias[i]));
        }
        for (int i = tid; i < half_dim; i += PLV3_THREADS) {
            float c = rope_cos[pos * half_dim + i];
            float s = rope_sin[pos * half_dim + i];
            float q0 = __half2float(q_head[2*i]);
            float q1 = __half2float(q_head[2*i+1]);
            q_head[2*i]   = __float2half(q0*c - q1*s);
            q_head[2*i+1] = __float2half(q0*s + q1*c);
        }
    } else if (bid < num_heads + num_kv_heads) {
        // K head: bias + RoPE + cache write
        int h = bid - num_heads;
        __half* k_head = qkv + q_dim + h * head_dim;
        if (qkv_bias) {
            const __half* bias = qkv_bias + q_dim + h * head_dim;
            for (int i = tid; i < head_dim; i += PLV3_THREADS)
                k_head[i] = __float2half(__half2float(k_head[i]) + __half2float(bias[i]));
        }
        for (int i = tid; i < half_dim; i += PLV3_THREADS) {
            float c = rope_cos[pos * half_dim + i];
            float s = rope_sin[pos * half_dim + i];
            float k0 = __half2float(k_head[2*i]);
            float k1 = __half2float(k_head[2*i+1]);
            float k0r = k0*c - k1*s, k1r = k0*s + k1*c;
            k_head[2*i]   = __float2half(k0r);
            k_head[2*i+1] = __float2half(k1r);
            if (slot >= 0) {
                int off = (slot * num_kv_heads + h) * head_dim;
                key_cache[off + 2*i]   = __float2half(k0r);
                key_cache[off + 2*i+1] = __float2half(k1r);
            }
        }
    } else if (bid < total_heads) {
        // V head: bias + cache write
        int h = bid - num_heads - num_kv_heads;
        __half* v_head = qkv + q_dim + kv_dim + h * head_dim;
        if (qkv_bias) {
            const __half* bias = qkv_bias + q_dim + kv_dim + h * head_dim;
            for (int i = tid; i < head_dim; i += PLV3_THREADS)
                v_head[i] = __float2half(__half2float(v_head[i]) + __half2float(bias[i]));
        }
        if (slot >= 0) {
            int off = (slot * num_kv_heads + h) * head_dim;
            for (int i = tid; i < head_dim; i += PLV3_THREADS)
                value_cache[off + i] = v_head[i];
        }
    }
}

// ============================================================================
// Phase 3: Split-KV attention (all blocks participate, BC=32)
// ============================================================================

__device__ __forceinline__ int plv3_num_splits(int ctx_len, int max_avail) {
    int d;
    if      (ctx_len <= 512)  d = 1;
    else if (ctx_len <= 2048) d = 2;
    else if (ctx_len <= 8192) d = 4;
    else                      d = 8;
    return min(d, max_avail);
}

__device__ __forceinline__ void plv3_load_kv_tile(
    __half* s_buf, const __half* cache,
    const int* block_tables, int kv_head_idx,
    int tile_start, int tile_len, int head_dim,
    int num_kv_heads, int block_size, int tid)
{
    const int chunks = (head_dim + 7) / 8;
    for (int tc = tid; tc < tile_len * chunks; tc += PLV3_THREADS) {
        int t = tc / chunks, ch = tc % chunks;
        int kv_pos = tile_start + t;
        int pi = kv_pos / block_size, po = kv_pos % block_size;
        int pb = block_tables[pi];
        const __half* src = &cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + ch * 8];
        __half* dst = &s_buf[t * head_dim + ch * 8];
        plv3_cp_async_16b(dst, src);
    }
}

__device__ __forceinline__ float plv3_block_max(float val, float* s_scratch) {
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, off);
        val = fmaxf(val, o);
    }
    if (lane_id == 0) s_scratch[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = s_scratch[0];
        for (int w = 1; w < PLV3_WARPS; w++) m = fmaxf(m, s_scratch[w]);
        s_scratch[0] = m;
    }
    __syncthreads();
    return s_scratch[0];
}

__device__ __forceinline__ float plv3_block_sum(float val, float* s_scratch) {
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    if (lane_id == 0) s_scratch[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < PLV3_WARPS; w++) s += s_scratch[w];
        s_scratch[0] = s;
    }
    __syncthreads();
    return s_scratch[0];
}

__device__ void plv3_splitkv_attention(
    __half* __restrict__ attn_out,
    const __half* __restrict__ qkv,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    int context_len,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, float attn_scale,
    float* __restrict__ split_max_buf,
    float* __restrict__ split_sum_buf,
    __half* __restrict__ split_acc_buf,
    int max_splits,
    float* smem, float* s_scratch
) {
    const int bid = (int)blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32, lane_id = tid % 32;
    const int hpg = num_heads / num_kv_heads;

    const int kv_head = bid % num_kv_heads;
    const int split_id = bid / num_kv_heads;
    const int max_avail = (int)gridDim.x / num_kv_heads;
    const int num_splits = plv3_num_splits(context_len, max_avail);

    if (split_id >= num_splits) return;

    const int total_tiles = (context_len + PLV3_FA_BC - 1) / PLV3_FA_BC;
    const int tps = (total_tiles + num_splits - 1) / num_splits;
    const int start_tile = split_id * tps;
    const int end_tile = min(start_tile + tps, total_tiles);

    if (start_tile >= total_tiles) {
        // Write sentinels
        for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
            int ws = (kv_head * max_splits + split_id) * hpg + g;
            if (tid == 0) { split_max_buf[ws] = -FLT_MAX; split_sum_buf[ws] = 0.0f; }
            for (int d = tid; d < head_dim; d += PLV3_THREADS)
                split_acc_buf[ws * head_dim + d] = __float2half(0.0f);
        }
        return;
    }

    // Shared memory: s_K[BC*hd], s_V[BC*hd], s_scores[HPG*STRIDE], s_warp[WARPS]
    __half* s_K = (__half*)smem;
    __half* s_V = s_K + PLV3_FA_BC * head_dim;
    float* s_scores = (float*)(s_V + PLV3_FA_BC * head_dim);
    float* s_warp = s_scores + PLV3_FA_HPG * PLV3_FA_SCORE_STRIDE;

    const int acc_dims = (head_dim + PLV3_THREADS - 1) / PLV3_THREADS;

    // Per-head registers
    float head_max[PLV3_FA_HPG], head_sum[PLV3_FA_HPG], head_acc[PLV3_FA_HPG][4];
    float q_reg[PLV3_FA_HPG][4];

    for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
        head_max[g] = -FLT_MAX;
        head_sum[g] = 0.0f;
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;

        // Load Q registers
        int qh = kv_head * hpg + g;
        int half2_iters = (head_dim + 63) / 64;
        for (int r = 0; r < half2_iters && r < 2; r++) {
            int d = lane_id * 2 + r * 64;
            if (d + 1 < head_dim) {
                q_reg[g][r*2]   = __half2float(qkv[qh * head_dim + d]) * attn_scale;
                q_reg[g][r*2+1] = __half2float(qkv[qh * head_dim + d + 1]) * attn_scale;
            } else {
                q_reg[g][r*2] = 0.0f;
                q_reg[g][r*2+1] = 0.0f;
            }
        }
    }

    for (int tile = start_tile; tile < end_tile; tile++) {
        int tile_start = tile * PLV3_FA_BC;
        int tile_len = min(PLV3_FA_BC, context_len - tile_start);

        // Load K
        plv3_load_kv_tile(s_K, key_cache, block_tables, kv_head,
                          tile_start, tile_len, head_dim, num_kv_heads, block_size, tid);
        plv3_cp_async_commit();
        plv3_cp_async_wait_0();
        __syncthreads();

        // QK^T per head
        for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
            float* gs = s_scores + g * PLV3_FA_SCORE_STRIDE;
            int half2_iters = (head_dim + 63) / 64;

            for (int t = warp_id; t < tile_len; t += PLV3_WARPS) {
                float dot = 0.0f;
                for (int r = 0; r < half2_iters && r < 2; r++) {
                    int d = lane_id * 2 + r * 64;
                    if (d + 1 < head_dim) {
                        half2 kv = *reinterpret_cast<const half2*>(&s_K[t * head_dim + d]);
                        dot += q_reg[g][r*2]   * __half2float(kv.x);
                        dot += q_reg[g][r*2+1] * __half2float(kv.y);
                    }
                }
                dot = plv3_warp_xor_reduce(dot);
                if (lane_id == 0) gs[t] = dot;
            }
            __syncthreads();

            // Online softmax
            float tile_max = plv3_block_max(
                (tid < tile_len) ? gs[tid] : -FLT_MAX, s_warp);

            float prev_max = head_max[g];
            float new_max = fmaxf(prev_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float corr = expf(prev_max - new_max);
                for (int r = 0; r < acc_dims && r < 4; r++) head_acc[g][r] *= corr;
                head_sum[g] *= corr;
            }
            head_max[g] = new_max;

            float my_exp = (tid < tile_len) ? expf(gs[tid] - new_max) : 0.0f;
            if (tid < tile_len) gs[tid] = my_exp;
            head_sum[g] += plv3_block_sum(my_exp, s_warp);
            __syncthreads();
        }

        // Load V
        plv3_load_kv_tile(s_V, value_cache, block_tables, kv_head,
                          tile_start, tile_len, head_dim, num_kv_heads, block_size, tid);
        plv3_cp_async_commit();
        plv3_cp_async_wait_0();
        __syncthreads();

        // P @ V
        for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
            float* gs = s_scores + g * PLV3_FA_SCORE_STRIDE;
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * PLV3_THREADS;
                if (d < head_dim) {
                    float vacc = 0.0f;
                    for (int t = 0; t < tile_len; t++)
                        vacc += gs[t] * __half2float(s_V[t * head_dim + d]);
                    head_acc[g][r] += vacc;
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
        int ws = (kv_head * max_splits + split_id) * hpg + g;

        if (num_splits == 1) {
            // Fast path: no combine needed, normalize and write directly
            float inv = (head_sum[g] > 0.0f) ? (1.0f / head_sum[g]) : 0.0f;
            int oh = (kv_head * hpg + g) * head_dim;
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * PLV3_THREADS;
                if (d < head_dim)
                    attn_out[oh + d] = __float2half(head_acc[g][r] * inv);
            }
        } else {
            // Multi-split: write partials for combine
            if (tid == 0) {
                split_max_buf[ws] = head_max[g];
                split_sum_buf[ws] = head_sum[g];
            }
            int ab = ws * head_dim;
            for (int r = 0; r < acc_dims && r < 4; r++) {
                int d = tid + r * PLV3_THREADS;
                if (d < head_dim)
                    split_acc_buf[ab + d] = __float2half(head_acc[g][r]);
            }
        }
    }
}

__device__ void plv3_splitkv_combine(
    __half* __restrict__ attn_out,
    const float* __restrict__ split_max_buf,
    const float* __restrict__ split_sum_buf,
    const __half* __restrict__ split_acc_buf,
    int num_heads, int num_kv_heads, int head_dim,
    int num_splits, int max_splits
) {
    const int bid = (int)blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = bid % num_kv_heads;
    const int split_id = bid / num_kv_heads;
    if (split_id != 0) return;

    const int hpg = num_heads / num_kv_heads;
    const int acc_dims = (head_dim + PLV3_THREADS - 1) / PLV3_THREADS;

    for (int g = 0; g < hpg && g < PLV3_FA_HPG; g++) {
        // Global max
        float gmax = -FLT_MAX;
        for (int s = 0; s < num_splits; s++) {
            int ws = (kv_head * max_splits + s) * hpg + g;
            gmax = fmaxf(gmax, split_max_buf[ws]);
        }
        if (gmax <= -FLT_MAX + 1.0f) {
            int oh = (kv_head * hpg + g) * head_dim;
            for (int d = tid; d < head_dim; d += PLV3_THREADS)
                attn_out[oh + d] = __float2half(0.0f);
            continue;
        }

        // Combined sum
        float csum = 0.0f;
        if (tid == 0) {
            for (int s = 0; s < num_splits; s++) {
                int ws = (kv_head * max_splits + s) * hpg + g;
                float sm = split_sum_buf[ws];
                if (sm > 0.0f) csum += expf(split_max_buf[ws] - gmax) * sm;
            }
        }
        __shared__ float s_csum;
        if (tid == 0) s_csum = csum;
        __syncthreads();
        csum = s_csum;
        __syncthreads();

        float inv = (csum > 0.0f) ? (1.0f / csum) : 0.0f;

        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * PLV3_THREADS;
            if (d < head_dim) {
                float acc_val = 0.0f;
                for (int s = 0; s < num_splits; s++) {
                    int ws = (kv_head * max_splits + s) * hpg + g;
                    float sm = split_sum_buf[ws];
                    if (sm > 0.0f) {
                        float corr = expf(split_max_buf[ws] - gmax);
                        acc_val += corr * __half2float(split_acc_buf[ws * head_dim + d]);
                    }
                }
                attn_out[(kv_head * hpg + g) * head_dim + d] = __float2half(acc_val * inv);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Phase 5: Residual + RMSNorm with race avoidance
// ============================================================================

__device__ void plv3_add_rmsnorm_phase5(
    float* s_normed, float* s_scratch,
    __half* __restrict__ residual_out,
    const __half* __restrict__ oproj,
    const __half* __restrict__ post_norm_w,
    float eps, int hidden_size,
    int* sync_flags
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32, lane_id = tid % 32;
    const int h2 = hidden_size / 2;

    const half2* r2 = (const half2*)residual_out;
    const half2* o2 = (const half2*)oproj;
    float local_ss = 0.0f;

    for (int i = tid; i < h2; i += PLV3_THREADS) {
        half2 a = r2[i], b = o2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i*2] = v0;
        s_normed[i*2+1] = v1;
        local_ss += v0*v0 + v1*v1;
    }

    local_ss = plv3_warp_reduce(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < PLV3_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = plv3_warp_reduce(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms = s_scratch[0];

    // DAG sync: all blocks loaded residual_out -> safe for block 0 to overwrite
    plv3_signal(sync_flags, PLV3_SYNC_LOAD5);
    plv3_wait(sync_flags, PLV3_SYNC_LOAD5, (int)gridDim.x);

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += PLV3_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += PLV3_THREADS)
        s_normed[i] = s_normed[i] * __half2float(post_norm_w[i]) * rms;
    __syncthreads();
}

// ============================================================================
// Phase 5.5: Compute the activated FFN vector directly from fused gate/up
// weights, then Phase 6 only has to run the down-projection.
// ============================================================================

__device__ void plv3_gateup_activate_gemv(
    __half* __restrict__ activated,
    const float* __restrict__ s_input,
    const __half* __restrict__ gateup_weight,
    int hidden_size,
    int intermediate_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int k4 = hidden_size / 8;

    for (int block_base = blockIdx.x * PLV3_BLOCK_N;
         block_base < intermediate_size;
         block_base += gridDim.x * PLV3_BLOCK_N)
    {
        int warp_base = block_base + warp_id * PLV3_RPW;
        if (warp_base >= intermediate_size) continue;
        int valid_rows = min(PLV3_RPW, intermediate_size - warp_base);

        for (int r = 0; r < valid_rows; r++) {
            int row_idx = warp_base + r;
            const __half* gate_row = gateup_weight + (long long)row_idx * hidden_size;
            const __half* up_row = gateup_weight + (long long)(row_idx + intermediate_size) * hidden_size;
            float gate_dot = 0.0f;
            float up_dot = 0.0f;

            for (int c = lane_id; c < k4; c += 32) {
                float4 gv = *reinterpret_cast<const float4*>(&gate_row[c * 8]);
                __half* gh = reinterpret_cast<__half*>(&gv);
                float4 uv = *reinterpret_cast<const float4*>(&up_row[c * 8]);
                __half* uh = reinterpret_cast<__half*>(&uv);
                float inp0 = s_input[c * 8 + 0];
                float inp1 = s_input[c * 8 + 1];
                float inp2 = s_input[c * 8 + 2];
                float inp3 = s_input[c * 8 + 3];
                float inp4 = s_input[c * 8 + 4];
                float inp5 = s_input[c * 8 + 5];
                float inp6 = s_input[c * 8 + 6];
                float inp7 = s_input[c * 8 + 7];

                gate_dot += __half2float(gh[0]) * inp0 + __half2float(gh[1]) * inp1
                         +  __half2float(gh[2]) * inp2 + __half2float(gh[3]) * inp3
                         +  __half2float(gh[4]) * inp4 + __half2float(gh[5]) * inp5
                         +  __half2float(gh[6]) * inp6 + __half2float(gh[7]) * inp7;
                up_dot += __half2float(uh[0]) * inp0 + __half2float(uh[1]) * inp1
                       +  __half2float(uh[2]) * inp2 + __half2float(uh[3]) * inp3
                       +  __half2float(uh[4]) * inp4 + __half2float(uh[5]) * inp5
                       +  __half2float(uh[6]) * inp6 + __half2float(uh[7]) * inp7;
            }

            int rem_start = k4 * 8;
            for (int k = rem_start + lane_id; k < hidden_size; k += 32) {
                float inp = s_input[k];
                gate_dot += __half2float(gate_row[k]) * inp;
                up_dot += __half2float(up_row[k]) * inp;
            }

            gate_dot = plv3_warp_xor_reduce(gate_dot);
            up_dot = plv3_warp_xor_reduce(up_dot);
            if (lane_id == 0)
                activated[row_idx] = __float2half(plv3_silu(gate_dot) * up_dot);
        }
    }
}

// ============================================================================
// Phase 6: Down-projection GEMV over the activated FFN vector.
// ============================================================================

__device__ void plv3_load_activated(
    __half* __restrict__ s_activated,
    const __half* __restrict__ activated,
    int intermediate_size
) {
    for (int i = threadIdx.x; i < intermediate_size; i += PLV3_THREADS)
        s_activated[i] = activated[i];
    __syncthreads();
}

__device__ void plv3_down_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ activated,
    const __half* __restrict__ down_weight,
    int hidden_size,
    int intermediate_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int k4 = intermediate_size / 8;  // number of float4-sized chunks

    for (int block_base = blockIdx.x * PLV3_BLOCK_N;
         block_base < hidden_size;
         block_base += gridDim.x * PLV3_BLOCK_N)
    {
        int warp_base = block_base + warp_id * PLV3_RPW;
        if (warp_base >= hidden_size) continue;
        int valid_rows = min(PLV3_RPW, hidden_size - warp_base);

        // Process one output row at a time
        for (int r = 0; r < valid_rows; r++) {
            const __half* w_row = down_weight + (long long)(warp_base + r) * intermediate_size;
            float dot = 0.0f;

            for (int c = lane_id; c < k4; c += 32) {
                // Load 8 weight values (16 bytes, coalesced)
                float4 wv = *reinterpret_cast<const float4*>(&w_row[c * 8]);
                __half* wh = reinterpret_cast<__half*>(&wv);

                // Load 8 activated values from shared memory
                float4 av = *reinterpret_cast<const float4*>(&activated[c * 8]);
                __half* ah = reinterpret_cast<__half*>(&av);

                // Activated vector dot product
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float w = __half2float(wh[j]);
                    dot += __half2float(ah[j]) * w;
                }
            }

            // Handle remainder if intermediate_size is not divisible by 8
            int rem_start = k4 * 8;
            for (int k = rem_start + lane_id; k < intermediate_size; k += 32) {
                float w = __half2float(down_weight[(long long)(warp_base + r) * intermediate_size + k]);
                dot += __half2float(activated[k]) * w;
            }

            dot = plv3_warp_xor_reduce(dot);
            if (lane_id == 0) output[warp_base + r] = __float2half(dot);
        }
    }
}

// ============================================================================
// MAIN PERSISTENT LAYER V3 KERNEL
// ============================================================================

template <bool Profile>
__device__ __forceinline__ void persistent_layer_v3_f16_body(
    // Outputs
    __half* __restrict__ mlp_out,
    __half* __restrict__ residual_out,
    // Inputs
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,       // NULL for layer 0
    // Attention I/O
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    // Weights
    const __half* __restrict__ norm_w,
    const __half* __restrict__ qkv_weight,     // [qkv_dim, hidden]
    const __half* __restrict__ qkv_bias,       // [qkv_dim] or NULL
    const __half* __restrict__ o_weight,        // [hidden, q_dim]
    const __half* __restrict__ post_norm_w,
    const __half* __restrict__ gateup_weight,  // [gate_up_dim, hidden]
    const __half* __restrict__ down_weight,    // [hidden, intermediate]
    // Scratch buffers
    __half* __restrict__ qkv_scratch,          // [qkv_dim]
    __half* __restrict__ attn_scratch,         // [q_dim]
    __half* __restrict__ oproj_scratch,        // [hidden]
    __half* __restrict__ gateup_scratch,       // activated vector in first [intermediate_size]
    // Split-KV scratch
    float* __restrict__ split_max_buf,
    float* __restrict__ split_sum_buf,
    __half* __restrict__ split_acc_buf,
    int max_splits,
    // Config
    float eps,
    float attn_scale,
    int hidden_size,
    int q_dim,
    int kv_dim,
    int qkv_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    int gate_up_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int* __restrict__ sync_flags             // [PLV3_NUM_SYNCS] zeroed before launch
) {
    // Dynamic shared memory layout:
    //   Phase 1/5: s_normed[hidden_size] (f32) + s_scratch[WARPS] (f32)
    //              = 3584*4 + 8*4 = 14368 bytes
    //   Phase 3:   s_K[BC*hd] (f16) + s_V[BC*hd] (f16) + s_scores[HPG*STRIDE] (f32) + s_warp[WARPS] (f32)
    //              = 32*128*2 + 32*128*2 + 8*33*4 + 8*4 = 8192+8192+1056+32 = 17472 bytes
    //   Phase 6:   activated[intermediate_size] (f16)
    //              = 18944 * 2 = 37888 bytes
    //   Max: ~37888 bytes
    extern __shared__ char smem_raw[];
    float* s_normed = (float*)smem_raw;
    float* s_scratch = s_normed + hidden_size;

    plv3_profile_stamp<Profile>(sync_flags, 0);

    // ====================================================================
    // PHASE 1: Residual add + RMSNorm + QKV GEMV
    // ====================================================================
    plv3_add_rmsnorm(s_normed, s_scratch, residual_out,
                     prev_residual, prev_mlp, norm_w, eps, hidden_size);

    plv3_gemv(qkv_scratch, s_normed, qkv_weight, qkv_dim, hidden_size);

    plv3_signal(sync_flags, PLV3_SYNC_QKV);
    plv3_wait(sync_flags, PLV3_SYNC_QKV, (int)gridDim.x);
    plv3_profile_stamp<Profile>(sync_flags, 1);

    // ====================================================================
    // PHASE 2: Bias + RoPE + KV cache write
    // ====================================================================
    plv3_rope_cache(qkv_scratch, key_cache, value_cache, qkv_bias,
                    positions, slot_mapping, rope_cos, rope_sin,
                    num_heads, num_kv_heads, head_dim, q_dim, kv_dim);

    plv3_signal(sync_flags, PLV3_SYNC_ROPE);
    plv3_wait(sync_flags, PLV3_SYNC_ROPE, (int)gridDim.x);
    plv3_profile_stamp<Profile>(sync_flags, 2);

    // ====================================================================
    // PHASE 3: Split-KV GQA attention
    // ====================================================================
    {
        int context_len = context_lens[0];
        int max_avail = (int)gridDim.x / num_kv_heads;
        int num_splits = plv3_num_splits(context_len, max_avail);

        // Reuse shared memory for attention layout
        float* attn_smem = (float*)smem_raw;
        float* attn_scratch_s = s_scratch;

        plv3_splitkv_attention(attn_scratch, qkv_scratch, key_cache, value_cache,
                               block_tables, context_len,
                               num_heads, num_kv_heads, head_dim,
                               block_size, attn_scale,
                               split_max_buf, split_sum_buf, split_acc_buf, max_splits,
                               attn_smem, attn_scratch_s);

        plv3_signal(sync_flags, PLV3_SYNC_ATTN);
        plv3_wait(sync_flags, PLV3_SYNC_ATTN, (int)gridDim.x);

        if (num_splits > 1) {
            plv3_splitkv_combine(attn_scratch, split_max_buf, split_sum_buf, split_acc_buf,
                                 num_heads, num_kv_heads, head_dim, num_splits, max_splits);
            __syncthreads();
        }
    }
    plv3_profile_stamp<Profile>(sync_flags, 3);

    // ====================================================================
    // PHASE 4: O-proj GEMV
    // ====================================================================
    {
        // Load attn output into shared memory as f32 for GEMV
        const int h2_q = q_dim / 2;
        const half2* attn2 = (const half2*)attn_scratch;
        for (int i = threadIdx.x; i < h2_q; i += PLV3_THREADS) {
            half2 a = attn2[i];
            s_normed[i*2]   = __half2float(a.x);
            s_normed[i*2+1] = __half2float(a.y);
        }
        __syncthreads();

        plv3_gemv(oproj_scratch, s_normed, o_weight, hidden_size, q_dim);
    }

    plv3_signal(sync_flags, PLV3_SYNC_OPROJ);
    plv3_wait(sync_flags, PLV3_SYNC_OPROJ, (int)gridDim.x);
    plv3_profile_stamp<Profile>(sync_flags, 4);

    // ====================================================================
    // PHASE 5: Residual add + RMSNorm + GateUp activation GEMV
    // ====================================================================
    plv3_add_rmsnorm_phase5(s_normed, s_scratch, residual_out, oproj_scratch,
                            post_norm_w, eps, hidden_size, sync_flags);

    plv3_gateup_activate_gemv(gateup_scratch, s_normed, gateup_weight,
                              hidden_size, intermediate_size);

    plv3_signal(sync_flags, PLV3_SYNC_GATEUP);
    plv3_wait(sync_flags, PLV3_SYNC_GATEUP, (int)gridDim.x);
    plv3_profile_stamp<Profile>(sync_flags, 5);

    // ====================================================================
    // PHASE 6: Down GEMV
    // ====================================================================
    {
        __half* s_activated = (__half*)smem_raw;
        plv3_load_activated(s_activated, gateup_scratch, intermediate_size);
        plv3_down_gemv(mlp_out, s_activated, down_weight, hidden_size, intermediate_size);
    }
    plv3_profile_stamp<Profile>(sync_flags, 6);
}

extern "C" __global__ void __launch_bounds__(256, 8)
persistent_layer_v3_f16(
    __half* __restrict__ mlp_out,
    __half* __restrict__ residual_out,
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const __half* __restrict__ norm_w,
    const __half* __restrict__ qkv_weight,
    const __half* __restrict__ qkv_bias,
    const __half* __restrict__ o_weight,
    const __half* __restrict__ post_norm_w,
    const __half* __restrict__ gateup_weight,
    const __half* __restrict__ down_weight,
    __half* __restrict__ qkv_scratch,
    __half* __restrict__ attn_scratch,
    __half* __restrict__ oproj_scratch,
    __half* __restrict__ gateup_scratch,
    float* __restrict__ split_max_buf,
    float* __restrict__ split_sum_buf,
    __half* __restrict__ split_acc_buf,
    int max_splits,
    float eps,
    float attn_scale,
    int hidden_size,
    int q_dim,
    int kv_dim,
    int qkv_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    int gate_up_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int* __restrict__ sync_flags
) {
    persistent_layer_v3_f16_body<false>(
        mlp_out, residual_out,
        prev_residual, prev_mlp,
        key_cache, value_cache, block_tables, context_lens, positions, slot_mapping,
        rope_cos, rope_sin,
        norm_w, qkv_weight, qkv_bias, o_weight, post_norm_w, gateup_weight, down_weight,
        qkv_scratch, attn_scratch, oproj_scratch, gateup_scratch,
        split_max_buf, split_sum_buf, split_acc_buf, max_splits,
        eps, attn_scale,
        hidden_size, q_dim, kv_dim, qkv_dim, num_heads, num_kv_heads, head_dim,
        intermediate_size, gate_up_dim, block_size, max_context_len, max_blocks_per_seq,
        sync_flags
    );
}

extern "C" __global__ void __launch_bounds__(256, 8)
persistent_layer_v3_f16_profile(
    __half* __restrict__ mlp_out,
    __half* __restrict__ residual_out,
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const __half* __restrict__ norm_w,
    const __half* __restrict__ qkv_weight,
    const __half* __restrict__ qkv_bias,
    const __half* __restrict__ o_weight,
    const __half* __restrict__ post_norm_w,
    const __half* __restrict__ gateup_weight,
    const __half* __restrict__ down_weight,
    __half* __restrict__ qkv_scratch,
    __half* __restrict__ attn_scratch,
    __half* __restrict__ oproj_scratch,
    __half* __restrict__ gateup_scratch,
    float* __restrict__ split_max_buf,
    float* __restrict__ split_sum_buf,
    __half* __restrict__ split_acc_buf,
    int max_splits,
    float eps,
    float attn_scale,
    int hidden_size,
    int q_dim,
    int kv_dim,
    int qkv_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    int gate_up_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int* __restrict__ sync_flags
) {
    persistent_layer_v3_f16_body<true>(
        mlp_out, residual_out,
        prev_residual, prev_mlp,
        key_cache, value_cache, block_tables, context_lens, positions, slot_mapping,
        rope_cos, rope_sin,
        norm_w, qkv_weight, qkv_bias, o_weight, post_norm_w, gateup_weight, down_weight,
        qkv_scratch, attn_scratch, oproj_scratch, gateup_scratch,
        split_max_buf, split_sum_buf, split_acc_buf, max_splits,
        eps, attn_scale,
        hidden_size, q_dim, kv_dim, qkv_dim, num_heads, num_kv_heads, head_dim,
        intermediate_size, gate_up_dim, block_size, max_context_len, max_blocks_per_seq,
        sync_flags
    );
}
