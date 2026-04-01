// Megakernel interpreter for Qwen 2.5 7B decode (M=1).
// Executes ALL 28 transformer layers + LM head in ONE kernel launch.
// Architecture: on-GPU interpreter with pre-computed instruction tape,
// atomic counter sync, paged shared memory. No cooperative groups.
//
// Compile: nvcc -cubin -arch=sm_90 -O3 --use_fast_math
// Grid: (256, 1, 1)  Block: (256, 1, 1)
// Shared: ~67 KB dynamic

#include <cuda_fp16.h>
#include <float.h>

// ============================================================================
// Constants
// ============================================================================
#define MK_THREADS     256
#define MK_WARPS       8
#define MK_RPB         8

// ============================================================================
// Instruction types
// ============================================================================
#define INSTR_NOP              0
#define INSTR_ADD_RMSNORM      1  // residual_add + RMSNorm -> smem page
#define INSTR_GEMV_STRIDED     2  // grid-strided GEMV: y = W @ x(smem)
#define INSTR_GEMV_CHUNKED     3  // chunked SiLU*mul + down-proj GEMV
#define INSTR_ROPE_CACHE       4  // bias + RoPE + KV cache write
#define INSTR_GQA_ATTENTION    5  // GQA flash attention decode
#define INSTR_ARGMAX           6  // argmax over output logits

// ============================================================================
// Instruction struct (64 bytes, one cache line)
// ============================================================================
struct __align__(64) MkInstr {
    // type + flags
    int      type;
    int      flags;           // bit 0: is_first_layer, bit 1: has_bias
                              // bit 2: clear_accum, bit 3: write_residual
    // sync
    int      wait_counter;    // -1 = no wait
    int      wait_value;
    int      signal_counter;  // -1 = no signal

    // dimensions
    int      dim_out;
    int      dim_in;
    int      dim_aux;         // chunk_id, num_heads, etc.

    // weight pointer index into weight_ptrs array
    int      weight_idx;
    int      norm_idx;
    int      bias_idx;        // -1 = no bias

    // global memory scratch offsets (byte offsets from scratch base)
    int      gmem_in;         // input offset
    int      gmem_out;        // output offset
    int      gmem_aux;        // auxiliary (residual, rope, etc.)
    int      gmem_aux2;       // second auxiliary

    float    eps;
};

// ============================================================================
// Atomic sync (same as persistent_layer_decode)
// ============================================================================
__device__ __forceinline__ void mk_signal(int* counters, int phase) {
    __threadfence();
    if (threadIdx.x == 0) atomicAdd(&counters[phase], 1);
}

__device__ __forceinline__ void mk_wait(int* counters, int phase, int expected) {
    if (threadIdx.x == 0) {
        while (atomicAdd(&counters[phase], 0) < expected) {}
    }
    __syncthreads();
}

// ============================================================================
// Device helpers
// ============================================================================
__device__ __forceinline__ float mk_warp_reduce(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float mk_warp_xor_reduce(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float mk_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// EXEC: ADD_RMSNORM
// Loads prev_residual + prev_mlp (or just prev_residual for layer 0)
// into shared memory, computes RMSNorm, applies norm weights.
// Block 0 writes pre-norm residual to global memory if flags & 8.
// ============================================================================
__device__ void exec_add_rmsnorm(
    float* smem, float* s_scratch,
    const MkInstr* instr,
    __half* scratch,
    const __half* const* weight_ptrs
) {
    const int tid = threadIdx.x;
    const int hidden = instr->dim_in;
    const int h2 = hidden / 2;
    const __half* norm_w = weight_ptrs[instr->norm_idx];
    const float eps = instr->eps;
    const bool is_first = (instr->flags & 1);

    const __half* prev_res = (__half*)(scratch + instr->gmem_in / 2);
    const __half* prev_mlp = is_first ? nullptr : (__half*)(scratch + instr->gmem_aux / 2);
    __half* residual_out = (__half*)(scratch + instr->gmem_out / 2);

    float local_ss = 0.0f;

    if (!is_first && prev_mlp != nullptr) {
        const half2* r2 = (const half2*)prev_res;
        const half2* m2 = (const half2*)prev_mlp;
        for (int i = tid; i < h2; i += MK_THREADS) {
            half2 a = r2[i], b = m2[i];
            float v0 = __half2float(a.x) + __half2float(b.x);
            float v1 = __half2float(a.y) + __half2float(b.y);
            smem[i * 2]     = v0;
            smem[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
    } else {
        const half2* r2 = (const half2*)prev_res;
        for (int i = tid; i < h2; i += MK_THREADS) {
            half2 a = r2[i];
            float v0 = __half2float(a.x), v1 = __half2float(a.y);
            smem[i * 2]     = v0;
            smem[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
    }

    // Warp reduction
    const int warp_id = tid / 32, lane_id = tid % 32;
    local_ss = mk_warp_reduce(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < MK_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = mk_warp_reduce(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden + eps);
    }
    __syncthreads();
    float rms_scale = s_scratch[0];

    // Block 0 writes pre-norm residual
    if (blockIdx.x == 0 && (instr->flags & 8)) {
        for (int i = tid; i < hidden; i += MK_THREADS)
            residual_out[i] = __float2half(smem[i]);
    }

    // Apply norm weights
    for (int i = tid; i < hidden; i += MK_THREADS)
        smem[i] = smem[i] * __half2float(norm_w[i]) * rms_scale;
    __syncthreads();
}

// ============================================================================
// EXEC: GEMV_STRIDED
// Grid-strided GEMV: output[row] = dot(weight[row,:], smem_input[:])
// All blocks participate.
// ============================================================================
__device__ void exec_gemv_strided(
    const float* smem_in, float* s_scratch,
    const MkInstr* instr,
    __half* scratch,
    const __half* const* weight_ptrs
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32, lane_id = tid % 32;
    const int out_dim = instr->dim_out;
    const int hidden  = instr->dim_in;
    const int h2 = hidden / 2;
    const __half* weight = weight_ptrs[instr->weight_idx];
    __half* output = (__half*)(scratch + instr->gmem_out / 2);

    for (int base = blockIdx.x * MK_RPB; base < out_dim; base += gridDim.x * MK_RPB) {
        int rows_here = min(MK_RPB, out_dim - base);
        for (int r = 0; r < rows_here; r++) {
            int row = base + r;
            const half2* w2 = (const half2*)(weight + (long long)row * hidden);
            float acc = 0.0f;
            for (int i = tid; i < h2; i += MK_THREADS) {
                half2 w = w2[i];
                acc += __half2float(w.x) * smem_in[i * 2];
                acc += __half2float(w.y) * smem_in[i * 2 + 1];
            }
            acc = mk_warp_reduce(acc);
            if (lane_id == 0) s_scratch[warp_id] = acc;
            __syncthreads();
            if (warp_id == 0) {
                float val = (lane_id < MK_WARPS) ? s_scratch[lane_id] : 0.0f;
                val = mk_warp_reduce(val);
                if (lane_id == 0) output[row] = __float2half(val);
            }
            if (r + 1 < rows_here) __syncthreads();
        }
    }
}

// ============================================================================
// EXEC: GEMV_CHUNKED (SiLU*mul + down-proj for one chunk)
// Warp-per-row, fused SiLU activation.
// ============================================================================
__device__ void exec_gemv_chunked(
    const MkInstr* instr,
    __half* scratch,
    const __half* const* weight_ptrs
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int hidden = instr->dim_out;       // output dim (hidden_size)
    const int chunk_size = instr->dim_in;    // cols this chunk
    const int chunk_id = instr->dim_aux;
    const int intermediate = chunk_size;
    const bool clear = (instr->flags & 4);

    const __half* gateup = (__half*)(scratch + instr->gmem_in / 2);
    const __half* gate = gateup + chunk_id * chunk_size;
    const __half* up   = gateup + (instr->gmem_aux / 2 > 0 ? instr->gmem_aux / 2 : 0);
    const __half* down_w = weight_ptrs[instr->weight_idx];
    __half* output = (__half*)(scratch + instr->gmem_out / 2);

    // Offset weight to this chunk's columns
    // down_weight is [hidden, full_intermediate], we want columns [chunk_id*chunk_size .. (chunk_id+1)*chunk_size]
    const int full_intermediate = chunk_size * 4; // MK_DOWN_CHUNKS

    for (int base = blockIdx.x * MK_RPB; base < hidden; base += gridDim.x * MK_RPB) {
        int row = base + warp_id;
        if (row >= hidden) continue;

        const half2* w2 = (const half2*)(down_w + (long long)row * full_intermediate + chunk_id * chunk_size);
        const half2* g2 = (const half2*)gate;
        const half2* u2 = (const half2*)up;
        const int k2 = chunk_size / 2;
        float acc = 0.0f;

        for (int i = lane_id; i < k2; i += 32) {
            half2 g = g2[i], u = u2[i], w = w2[i];
            float g0 = __half2float(g.x), g1 = __half2float(g.y);
            float u0 = __half2float(u.x), u1 = __half2float(u.y);
            float w0 = __half2float(w.x), w1 = __half2float(w.y);
            acc += mk_silu(g0) * u0 * w0;
            acc += mk_silu(g1) * u1 * w1;
        }

        acc = mk_warp_reduce(acc);
        if (lane_id == 0) {
            if (clear)
                output[row] = __float2half(acc);
            else
                output[row] = __float2half(__half2float(output[row]) + acc);
        }
    }
}

// ============================================================================
// EXEC: ROPE_CACHE
// Parallelized bias + RoPE + KV cache write across blocks.
// ============================================================================
__device__ void exec_rope_cache(
    const MkInstr* instr,
    __half* scratch,
    const __half* const* weight_ptrs,
    __half* key_cache, __half* value_cache,
    const int* block_tables, const int* positions,
    const int* slot_mapping,
    const float* rope_cos, const float* rope_sin
) {
    const int tid = threadIdx.x;
    const int bid = (int)blockIdx.x;
    const int head_dim = instr->dim_aux;
    const int half_dim = head_dim / 2;
    const int num_heads = instr->dim_out;      // Q heads
    const int num_kv_heads = instr->dim_in;    // KV heads
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int pos = positions[0];
    const int slot = slot_mapping[0];

    __half* qkv = (__half*)(scratch + instr->gmem_in / 2);
    const int total_heads = num_heads + 2 * num_kv_heads;

    // Optional bias
    if ((instr->flags & 2) && instr->bias_idx >= 0) {
        const __half* bias = weight_ptrs[instr->bias_idx];
        int qkv_dim = q_dim + 2 * kv_dim;
        for (int i = bid * MK_THREADS + tid; i < qkv_dim; i += gridDim.x * MK_THREADS)
            qkv[i] = __float2half(__half2float(qkv[i]) + __half2float(bias[i]));
        // Need all blocks to finish bias before RoPE
        // (handled by caller's sync design -- bias and RoPE are in same instruction)
    }

    if (bid < num_heads) {
        // Q head: RoPE only
        __half* q_head = qkv + bid * head_dim;
        for (int i = tid; i < half_dim; i += MK_THREADS) {
            float c = rope_cos[pos * half_dim + i];
            float s = rope_sin[pos * half_dim + i];
            float q0 = __half2float(q_head[2*i]);
            float q1 = __half2float(q_head[2*i+1]);
            q_head[2*i]   = __float2half(q0*c - q1*s);
            q_head[2*i+1] = __float2half(q0*s + q1*c);
        }
    } else if (bid < num_heads + num_kv_heads) {
        // K head: RoPE + cache write
        int h = bid - num_heads;
        __half* k_head = qkv + q_dim + h * head_dim;
        for (int i = tid; i < half_dim; i += MK_THREADS) {
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
        // V head: cache write (no RoPE)
        int h = bid - num_heads - num_kv_heads;
        __half* v_head = qkv + q_dim + kv_dim + h * head_dim;
        if (slot >= 0) {
            int off = (slot * num_kv_heads + h) * head_dim;
            for (int i = tid; i < head_dim; i += MK_THREADS)
                value_cache[off + i] = v_head[i];
        }
    }
}

// ============================================================================
// EXEC: GQA_ATTENTION
// Same algorithm as persistent_layer_decode Phase 3.
// ============================================================================
#define MK_FA_BC 64
#define MK_FA_MAX_HPG 8

__device__ void exec_gqa_attention(
    float* smem, float* s_scratch,
    const MkInstr* instr,
    __half* scratch,
    const __half* key_cache, const __half* value_cache,
    const int* block_tables, const int* context_lens,
    int block_size
) {
    const int bid = (int)blockIdx.x;
    const int tid = threadIdx.x;
    const int num_kv_heads = instr->dim_in;
    if (bid >= num_kv_heads) return;

    const int kv_head_idx = bid;
    const int warp_id = tid / 32, lane_id = tid % 32;
    const int head_dim = instr->dim_aux;
    const int num_heads = instr->dim_out;
    const int heads_per_group = num_heads / num_kv_heads;
    const int context_len = context_lens[0];
    const int num_tiles = (context_len + MK_FA_BC - 1) / MK_FA_BC;
    const int dims_per_thread = (head_dim + MK_THREADS - 1) / MK_THREADS;
    const float attn_scale = instr->eps; // reuse eps field for attn_scale

    __half* qkv = (__half*)(scratch + instr->gmem_in / 2);
    __half* attn_out = (__half*)(scratch + instr->gmem_out / 2);

    float* s_kv     = smem;
    float* s_scores = smem + MK_FA_BC * head_dim;
    float* s_warp   = s_scores + MK_FA_MAX_HPG * MK_FA_BC;

    float head_row_max[MK_FA_MAX_HPG];
    float head_row_sum[MK_FA_MAX_HPG];
    float head_acc[MK_FA_MAX_HPG][4];
    float q_reg[4];

    for (int g = 0; g < heads_per_group && g < MK_FA_MAX_HPG; g++) {
        head_row_max[g] = -FLT_MAX;
        head_row_sum[g] = 0.0f;
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * MK_FA_BC;
        int tile_len = min(MK_FA_BC, context_len - tile_start);

        // Load K tile
        int total_h2 = (tile_len * head_dim) / 2;
        for (int idx = tid; idx < total_h2; idx += MK_THREADS) {
            int elem = idx * 2;
            int t = elem / head_dim, d = elem % head_dim;
            int kv_pos = tile_start + t;
            int pi = kv_pos / block_size, po = kv_pos % block_size;
            int pb = block_tables[pi];
            int base = ((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d;
            __half2 h2v = *reinterpret_cast<const __half2*>(&key_cache[base]);
            s_kv[t * head_dim + d]     = __half2float(h2v.x);
            s_kv[t * head_dim + d + 1] = __half2float(h2v.y);
        }
        __syncthreads();

        // QK^T + online softmax per head
        for (int g = 0; g < heads_per_group && g < MK_FA_MAX_HPG; g++) {
            int head_idx = kv_head_idx * heads_per_group + g;
            float* g_scores = s_scores + g * MK_FA_BC;
            int q_base = head_idx * head_dim;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * MK_THREADS;
                q_reg[r] = (d < head_dim) ? (__half2float(qkv[q_base + d]) * attn_scale) : 0.0f;
            }

            for (int t = 0; t < tile_len; t++) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * MK_THREADS;
                    if (d < head_dim) dot += q_reg[r] * s_kv[t * head_dim + d];
                }
                dot = mk_warp_xor_reduce(dot);
                if (lane_id == 0) s_warp[warp_id] = dot;
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int w = 0; w < MK_WARPS; w++) total += s_warp[w];
                    g_scores[t] = total;
                }
                __syncthreads();
            }

            // Online softmax
            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, g_scores[t]);
                s_warp[0] = tile_max;
            }
            __syncthreads();
            tile_max = s_warp[0];
            __syncthreads();

            float prev_max = head_row_max[g];
            float new_max = fmaxf(prev_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) head_acc[g][r] *= correction;
                head_row_sum[g] *= correction;
            }
            head_row_max[g] = new_max;

            if (tid == 0) {
                float tsum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    float v = expf(g_scores[t] - new_max);
                    g_scores[t] = v;
                    tsum += v;
                }
                s_warp[0] = tsum;
            }
            __syncthreads();
            head_row_sum[g] += s_warp[0];
            __syncthreads();
        }

        // Load V tile (reuse s_kv)
        for (int idx = tid; idx < total_h2; idx += MK_THREADS) {
            int elem = idx * 2;
            int t = elem / head_dim, d = elem % head_dim;
            int kv_pos = tile_start + t;
            int pi = kv_pos / block_size, po = kv_pos % block_size;
            int pb = block_tables[pi];
            int base = ((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d;
            __half2 h2v = *reinterpret_cast<const __half2*>(&value_cache[base]);
            s_kv[t * head_dim + d]     = __half2float(h2v.x);
            s_kv[t * head_dim + d + 1] = __half2float(h2v.y);
        }
        __syncthreads();

        // P @ V
        for (int g = 0; g < heads_per_group && g < MK_FA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * MK_FA_BC;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * MK_THREADS;
                if (d < head_dim) {
                    float v_acc = 0.0f;
                    for (int t = 0; t < tile_len; t++)
                        v_acc += g_scores[t] * s_kv[t * head_dim + d];
                    head_acc[g][r] += v_acc;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    for (int g = 0; g < heads_per_group && g < MK_FA_MAX_HPG; g++) {
        int head_idx = kv_head_idx * heads_per_group + g;
        float inv = (head_row_sum[g] > 0.0f) ? (1.0f / head_row_sum[g]) : 0.0f;
        int out_base = head_idx * head_dim;
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * MK_THREADS;
            if (d < head_dim)
                attn_out[out_base + d] = __float2half(head_acc[g][r] * inv);
        }
    }
}

// ============================================================================
// EXEC: ARGMAX over output logits
// ============================================================================
__device__ void exec_argmax(
    float* s_scratch,
    const MkInstr* instr,
    __half* scratch,
    int* output_token
) {
    const int tid = threadIdx.x;
    const int vocab = instr->dim_out;
    const __half* logits = (__half*)(scratch + instr->gmem_in / 2);

    // Only block 0 does argmax
    if (blockIdx.x != 0) return;

    float best_val = -FLT_MAX;
    int   best_idx = 0;

    for (int i = tid; i < vocab; i += MK_THREADS) {
        float v = __half2float(logits[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    // Warp reduction for max
    for (int off = 16; off > 0; off >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, off);
        int   other_idx = __shfl_xor_sync(0xffffffff, best_idx, off);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }

    int warp_id = tid / 32, lane_id = tid % 32;
    __shared__ float s_vals[8];
    __shared__ int   s_idxs[8];
    if (lane_id == 0) { s_vals[warp_id] = best_val; s_idxs[warp_id] = best_idx; }
    __syncthreads();

    if (tid == 0) {
        float bv = s_vals[0]; int bi = s_idxs[0];
        for (int w = 1; w < MK_WARPS; w++) {
            if (s_vals[w] > bv) { bv = s_vals[w]; bi = s_idxs[w]; }
        }
        output_token[0] = bi;
    }
}

// ============================================================================
// MAIN MEGAKERNEL
// ============================================================================
extern "C" __global__ void __launch_bounds__(MK_THREADS, 2)
megakernel_decode_f16(
    const MkInstr*       __restrict__ instructions,
    int                               num_instructions,
    const __half* const* __restrict__ weight_ptrs,
    __half*              __restrict__ scratch,
    int*                              sync_counters,
    __half*              __restrict__ key_cache,
    __half*              __restrict__ value_cache,
    const int*           __restrict__ block_tables,
    const int*           __restrict__ context_lens,
    const int*           __restrict__ positions,
    const int*           __restrict__ slot_mapping,
    const float*         __restrict__ rope_cos,
    const float*         __restrict__ rope_sin,
    int*                 __restrict__ output_token,
    int                               block_size,
    int                               max_context_len
) {
    extern __shared__ float smem[];
    float* s_scratch = smem + 3584; // after hidden_size floats

    // Instruction interpreter loop
    for (int pc = 0; pc < num_instructions; pc++) {
        // Broadcast instruction to all threads via shared memory
        __shared__ MkInstr s_instr;
        if (threadIdx.x == 0) s_instr = instructions[pc];
        __syncthreads();
        const MkInstr* instr = &s_instr;

        // Wait
        if (instr->wait_counter >= 0) {
            mk_wait(sync_counters, instr->wait_counter, instr->wait_value);
        }

        // Dispatch
        switch (instr->type) {
        case INSTR_ADD_RMSNORM:
            exec_add_rmsnorm(smem, s_scratch, instr, scratch, weight_ptrs);
            break;
        case INSTR_GEMV_STRIDED:
            exec_gemv_strided(smem, s_scratch, instr, scratch, weight_ptrs);
            break;
        case INSTR_GEMV_CHUNKED:
            exec_gemv_chunked(instr, scratch, weight_ptrs);
            break;
        case INSTR_ROPE_CACHE:
            exec_rope_cache(instr, scratch, weight_ptrs,
                key_cache, value_cache, block_tables,
                positions, slot_mapping, rope_cos, rope_sin);
            break;
        case INSTR_GQA_ATTENTION:
            exec_gqa_attention(smem, s_scratch, instr, scratch,
                key_cache, value_cache, block_tables, context_lens,
                block_size);
            break;
        case INSTR_ARGMAX:
            exec_argmax(s_scratch, instr, scratch, output_token);
            break;
        default:
            break;
        }

        // Signal
        if (instr->signal_counter >= 0) {
            mk_signal(sync_counters, instr->signal_counter);
        }
    }
}
