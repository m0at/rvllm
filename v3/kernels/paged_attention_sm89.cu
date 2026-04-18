// Paged attention kernels for SM89 (Ada Lovelace).
//
// Drop-in replacement for the FA3 SM90 .so. Exports the same C ABI so
// the Rust Fa3Kernels loader can dlopen this .so and resolve symbols.
//
// Build:
//   nvcc -shared -o libfa_sm89_kernels.so paged_attention_sm89.cu \
//        -arch=sm_89 -O3 --use_fast_math -Xcompiler -fPIC
//
// Kernels: paged decode (f16 and fp8) + paged prefill (fp8).
// Single-split design: one thread block per (batch, q_head). Iterates
// sequentially over KV pages. Online softmax avoids two-pass.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>

// FP8 E4M3 -> float conversion (manual, no cuda_fp8.h dependency).
// E4M3: 1 sign, 4 exp (bias=7), 3 mantissa. Range [-448, 448].
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    uint32_t s = (x >> 7) & 1;
    uint32_t e = (x >> 3) & 0xF;
    uint32_t m = x & 0x7;
    if (e == 0) {
        if (m == 0) return 0.0f;
        float val = (float)m * 1.953125e-3f; // m * 2^-9
        return s ? -val : val;
    }
    if (e == 15 && m == 7) {
        return __int_as_float(0x7FC00000); // NaN
    }
    uint32_t f32 = (s << 31) | ((e + 120u) << 23) | (m << 20);
    return __int_as_float(f32);
}

// -------------------------------------------------------------------
// Paged decode: one Q token per sequence (batch decode).
//
// Grid:  (batch_size * num_heads, 1, 1)
// Block: (HEAD_DIM, 1, 1)   e.g. 256 threads for head_dim=256
//
// Each thread owns dimension `tid` of Q and accumulates dimension
// `tid` of the output via online softmax over all KV tokens.
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_decode_f16_kernel(
    const __half* __restrict__ q,          // [batch, num_heads, head_dim]
    const __half* __restrict__ k_cache,    // [num_blocks_total, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ v_cache,    // [num_blocks_total, block_size, num_kv_heads, head_dim]
    __half* __restrict__ output,           // [batch, num_heads, head_dim]
    const int* __restrict__ block_tables,  // [batch, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [batch]
    float scale,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq
) {
    const int bid = blockIdx.x;
    const int batch_idx = bid / num_heads;
    const int head_idx  = bid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) {
        output[bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    float q_val = __half2float(q[bid * HEAD_DIM + tid]);

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_pages = (ctx_len + block_size - 1) / block_size;

    for (int p = 0; p < num_pages; p++) {
        int phys = block_tables[batch_idx * max_blocks_per_seq + p];
        int toks = min(block_size, ctx_len - p * block_size);

        for (int t = 0; t < toks; t++) {
            int kv_idx = ((phys * block_size + t) * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_val = __half2float(k_cache[kv_idx]);

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = __half2float(v_cache[kv_idx]);
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    output[bid * HEAD_DIM + tid] = __float2half(acc);
}

// FP8 variant: Q/K/V are FP8 E4M3 with per-tensor descales. Output f16.
template<int HEAD_DIM>
__global__ void paged_decode_fp8_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    __half* __restrict__ output,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float* __restrict__ q_descale_ptr,
    const float* __restrict__ k_descale_ptr,
    const float* __restrict__ v_descale_ptr,
    float scale,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq
) {
    const int bid = blockIdx.x;
    const int batch_idx = bid / num_heads;
    const int head_idx  = bid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) {
        output[bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    const float q_ds = *q_descale_ptr;
    const float k_ds = *k_descale_ptr;
    const float v_ds = *v_descale_ptr;

    float q_val = fp8e4m3_to_float(q_fp8[bid * HEAD_DIM + tid]) * q_ds;

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_pages = (ctx_len + block_size - 1) / block_size;

    for (int p = 0; p < num_pages; p++) {
        int phys = block_tables[batch_idx * max_blocks_per_seq + p];
        int toks = min(block_size, ctx_len - p * block_size);

        for (int t = 0; t < toks; t++) {
            int kv_idx = ((phys * block_size + t) * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_val = fp8e4m3_to_float(k_cache_fp8[kv_idx]) * k_ds;

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = fp8e4m3_to_float(v_cache_fp8[kv_idx]) * v_ds;
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    output[bid * HEAD_DIM + tid] = __float2half(acc);
}

// -------------------------------------------------------------------
// Paged prefill FP8: multiple Q tokens per sequence with causal mask.
//
// Grid:  (total_q * num_heads, 1, 1)
// Block: (HEAD_DIM, 1, 1)
//
// Each block handles one (q_token_in_batch, head) pair. The q token's
// sequence index and position within the sequence are derived from
// cu_seqlens_q. Causal masking: q at position p attends to KV [0..p].
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_prefill_fp8_kernel(
    const uint8_t* __restrict__ q_fp8,       // [total_q, num_heads, head_dim]
    const uint8_t* __restrict__ k_cache_fp8, // paged
    const uint8_t* __restrict__ v_cache_fp8, // paged
    __half* __restrict__ output,             // [total_q, num_heads, head_dim]
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ cu_seqlens_q,    // [batch+1]
    const float* __restrict__ q_descale_ptr,
    const float* __restrict__ k_descale_ptr,
    const float* __restrict__ v_descale_ptr,
    float scale,
    int total_q,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq
) {
    const int gid = blockIdx.x;
    const int q_token_idx = gid / num_heads;
    const int head_idx    = gid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    if (q_token_idx >= total_q) return;

    // Find which sequence this Q token belongs to (linear scan; batch is small).
    int seq_idx = 0;
    for (int s = 0; s < batch_size; s++) {
        if (q_token_idx < cu_seqlens_q[s + 1]) { seq_idx = s; break; }
    }
    int q_pos_in_seq = q_token_idx - cu_seqlens_q[seq_idx];

    // Causal: attend to KV positions [0, q_pos_in_seq].
    int causal_len = q_pos_in_seq + 1;
    int ctx_len = context_lens[seq_idx];
    int attend_len = min(causal_len, ctx_len);

    const float q_ds = *q_descale_ptr;
    const float k_ds = *k_descale_ptr;
    const float v_ds = *v_descale_ptr;

    int q_offset = (q_token_idx * num_heads + head_idx) * HEAD_DIM + tid;
    float q_val = fp8e4m3_to_float(q_fp8[q_offset]) * q_ds;

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_pages = (attend_len + block_size - 1) / block_size;

    for (int p = 0; p < num_pages; p++) {
        int phys = block_tables[seq_idx * max_blocks_per_seq + p];
        int toks = min(block_size, attend_len - p * block_size);

        for (int t = 0; t < toks; t++) {
            int kv_idx = ((phys * block_size + t) * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_val = fp8e4m3_to_float(k_cache_fp8[kv_idx]) * k_ds;

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = fp8e4m3_to_float(v_cache_fp8[kv_idx]) * v_ds;
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    int out_idx = (q_token_idx * num_heads + head_idx) * HEAD_DIM + tid;
    output[out_idx] = __float2half(acc);
}

// -------------------------------------------------------------------
// C ABI wrappers. Symbol names: fa_sm89_*
// -------------------------------------------------------------------

extern "C" {

int fa_sm89_workspace_size(int batch_size, int num_heads, int max_num_splits) {
    (void)batch_size; (void)num_heads; (void)max_num_splits;
    return 0; // single-split, no workspace needed
}

int fa_sm89_paged_decode(
    void* q, void* k_cache, void* v_cache, void* output,
    void* block_tables, void* context_lens, void* workspace,
    float scale,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    (void)workspace; (void)num_blocks_total; (void)window_size_left;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = batch_size * num_heads;
    if (grid == 0) return 0;

    #define LAUNCH_F16(HD) \
        paged_decode_f16_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const __half*)q, (const __half*)k_cache, (const __half*)v_cache, \
            (__half*)output, (const int*)block_tables, (const int*)context_lens, \
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq)

    if      (head_dim == 128) { LAUNCH_F16(128); }
    else if (head_dim == 256) { LAUNCH_F16(256); }
    else if (head_dim == 512) { LAUNCH_F16(512); }
    else { return -1; }
    #undef LAUNCH_F16

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int fa_sm89_paged_decode_fp8(
    void* q_fp8, void* k_cache_fp8, void* v_cache_fp8, void* output,
    void* block_tables, void* context_lens, void* workspace,
    float* q_descale, float* k_descale, float* v_descale,
    float scale,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    (void)workspace; (void)num_blocks_total; (void)window_size_left;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = batch_size * num_heads;
    if (grid == 0) return 0;

    #define LAUNCH_FP8(HD) \
        paged_decode_fp8_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const uint8_t*)q_fp8, (const uint8_t*)k_cache_fp8, \
            (const uint8_t*)v_cache_fp8, (__half*)output, \
            (const int*)block_tables, (const int*)context_lens, \
            (const float*)q_descale, (const float*)k_descale, \
            (const float*)v_descale, \
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq)

    if      (head_dim == 128) { LAUNCH_FP8(128); }
    else if (head_dim == 256) { LAUNCH_FP8(256); }
    else if (head_dim == 512) { LAUNCH_FP8(512); }
    else { return -1; }
    #undef LAUNCH_FP8

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int fa_sm89_paged_prefill_fp8(
    void* q_fp8, void* k_cache_fp8, void* v_cache_fp8, void* output,
    void* block_tables, void* context_lens, void* cu_seqlens_q,
    void* workspace,
    float* q_descale, float* k_descale, float* v_descale,
    float scale,
    int total_q, int max_seqlen_q,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    (void)workspace; (void)max_seqlen_q; (void)num_blocks_total; (void)window_size_left;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = total_q * num_heads;
    if (grid == 0) return 0;

    #define LAUNCH_PREFILL(HD) \
        paged_prefill_fp8_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const uint8_t*)q_fp8, (const uint8_t*)k_cache_fp8, \
            (const uint8_t*)v_cache_fp8, (__half*)output, \
            (const int*)block_tables, (const int*)context_lens, \
            (const int*)cu_seqlens_q, \
            (const float*)q_descale, (const float*)k_descale, \
            (const float*)v_descale, \
            scale, total_q, batch_size, num_heads, num_kv_heads, \
            block_size, max_blocks_per_seq)

    if      (head_dim == 128) { LAUNCH_PREFILL(128); }
    else if (head_dim == 256) { LAUNCH_PREFILL(256); }
    else if (head_dim == 512) { LAUNCH_PREFILL(512); }
    else { return -1; }
    #undef LAUNCH_PREFILL

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

} // extern "C"
