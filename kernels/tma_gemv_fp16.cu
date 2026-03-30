// Async-prefetch GEMV kernel for sm_90+ (H100).
// Uses cp.async double-buffering to overlap weight loads with compute.
// f16 I/O, f32 accumulation. 8 rows per block, warp-per-row.
//
// The input vector is cached in shared memory once (shared across all 8 rows).
// Weight tiles are double-buffered via cp.async: while tile N computes,
// tile N+1 prefetches asynchronously through the memory subsystem.
//
// Launch config:
//   Grid:  ((out_dim + 7) / 8, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: in_dim * sizeof(half) + 2 * RPB * TILE_K * sizeof(half) + RPB * sizeof(float)

#include <cuda_fp16.h>

#define TMA_THREADS 256
#define TMA_RPB 8
#define TMA_WARP_SIZE 32
#define TMA_TILE_K 256

// cp.async: copy 16 bytes (8 half values) from global to shared
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(smem_addr), "l"(global_ptr)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 1;" ::: "memory");
    }
}

__device__ __forceinline__ float warp_reduce_sum_tma(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(TMA_THREADS, 2)
tma_gemv_fp16_kernel(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [in_dim]
    const __half* __restrict__ weight, // [out_dim, in_dim]
    int out_dim,
    int in_dim
) {
    const int base_row = blockIdx.x * TMA_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / TMA_WARP_SIZE;
    const int lane_id = tid % TMA_WARP_SIZE;
    const int row = base_row + warp_id;

    // Shared memory layout:
    //   s_input:  [in_dim] half -- input vector cached for all rows
    //   s_weight: [2][RPB][TILE_K] half -- double-buffered weight tiles
    //   s_scratch: [RPB] float -- for any scratch (unused here, kept for alignment)
    extern __shared__ char smem_raw[];
    __half* s_input = (__half*)smem_raw;
    __half* s_weight = (__half*)(smem_raw + in_dim * sizeof(__half));
    // s_weight layout: buffer b, row r, element k -> s_weight[b * RPB * TMA_TILE_K + r * TMA_TILE_K + k]

    // Phase 1: Cooperatively load the entire input vector into shared memory.
    // All 256 threads participate, loading half2 for bandwidth.
    const int in2 = in_dim / 2;
    const half2* input2 = (const half2*)input;
    half2* s_input2 = (half2*)s_input;
    for (int i = tid; i < in2; i += TMA_THREADS) {
        s_input2[i] = input2[i];
    }
    if ((in_dim & 1) && tid == 0) {
        s_input[in_dim - 1] = input[in_dim - 1];
    }
    __syncthreads();

    if (row >= out_dim) return;

    const int num_tiles = (in_dim + TMA_TILE_K - 1) / TMA_TILE_K;
    const __half* w_row = weight + (long long)row * in_dim;

    // Offset into this warp's weight buffer slot
    // Each warp (row within block) has its own TILE_K region within the double buffer
    const int warp_buf_offset = warp_id * TMA_TILE_K;

    // -- Prefetch tile 0 into buffer 0 --
    {
        const int tile_start = 0;
        const int tile_end = (TMA_TILE_K < in_dim) ? TMA_TILE_K : in_dim;
        const int tile_len = tile_end - tile_start;
        // Each lane in the warp copies a chunk via cp.async (16 bytes = 8 halves at a time)
        const __half* src_base = w_row + tile_start;
        __half* dst_base = s_weight + 0 * TMA_RPB * TMA_TILE_K + warp_buf_offset;
        // Use cp.async in 16-byte chunks; each lane handles strided chunks
        const int elems_per_cp = 8; // 16 bytes / 2 bytes per half
        const int num_cps = tile_len / elems_per_cp;
        for (int c = lane_id; c < num_cps; c += TMA_WARP_SIZE) {
            cp_async_16(dst_base + c * elems_per_cp, src_base + c * elems_per_cp);
        }
        // Handle remainder with direct load (rare, only if tile_len not multiple of 8)
        const int cp_covered = num_cps * elems_per_cp;
        for (int i = cp_covered + lane_id; i < tile_len; i += TMA_WARP_SIZE) {
            dst_base[i] = src_base[i];
        }
    }
    cp_async_commit();

    float acc = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        const int buf_cur = t & 1;
        const int buf_nxt = 1 - buf_cur;
        const int tile_start = t * TMA_TILE_K;
        const int tile_end = ((t + 1) * TMA_TILE_K < in_dim) ? (t + 1) * TMA_TILE_K : in_dim;
        const int tile_len = tile_end - tile_start;

        // Prefetch next tile into buf_nxt (if there is a next tile)
        if (t + 1 < num_tiles) {
            const int nxt_start = (t + 1) * TMA_TILE_K;
            const int nxt_end = ((t + 2) * TMA_TILE_K < in_dim) ? (t + 2) * TMA_TILE_K : in_dim;
            const int nxt_len = nxt_end - nxt_start;
            const __half* src_base = w_row + nxt_start;
            __half* dst_base = s_weight + buf_nxt * TMA_RPB * TMA_TILE_K + warp_buf_offset;
            const int elems_per_cp = 8;
            const int num_cps = nxt_len / elems_per_cp;
            for (int c = lane_id; c < num_cps; c += TMA_WARP_SIZE) {
                cp_async_16(dst_base + c * elems_per_cp, src_base + c * elems_per_cp);
            }
            const int cp_covered = num_cps * elems_per_cp;
            for (int i = cp_covered + lane_id; i < nxt_len; i += TMA_WARP_SIZE) {
                dst_base[i] = src_base[i];
            }
            cp_async_commit();
        }

        // Wait for current tile's data
        cp_async_wait_group(t + 1 < num_tiles ? 1 : 0);
        __syncwarp();

        // Compute dot product on current tile using half2
        const __half* w_tile = s_weight + buf_cur * TMA_RPB * TMA_TILE_K + warp_buf_offset;
        const __half* in_tile = s_input + tile_start;

        const int tile_h2 = tile_len / 2;
        const half2* w_tile2 = (const half2*)w_tile;
        const half2* in_tile2 = (const half2*)in_tile;

        for (int i = lane_id; i < tile_h2; i += TMA_WARP_SIZE) {
            half2 w = w_tile2[i];
            half2 x = in_tile2[i];
            acc += __half2float(w.x) * __half2float(x.x) + __half2float(w.y) * __half2float(x.y);
        }
        // Handle odd element
        if ((tile_len & 1) && lane_id == 0) {
            acc += __half2float(w_tile[tile_len - 1]) * __half2float(in_tile[tile_len - 1]);
        }
    }

    // Warp reduction
    acc = warp_reduce_sum_tma(acc);

    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}

// Variant with bias addition
extern "C"
__global__ void __launch_bounds__(TMA_THREADS, 2)
tma_gemv_fp16_bias_kernel(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [in_dim]
    const __half* __restrict__ weight, // [out_dim, in_dim]
    const __half* __restrict__ bias,   // [out_dim]
    int out_dim,
    int in_dim
) {
    const int base_row = blockIdx.x * TMA_RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / TMA_WARP_SIZE;
    const int lane_id = tid % TMA_WARP_SIZE;
    const int row = base_row + warp_id;

    extern __shared__ char smem_raw[];
    __half* s_input = (__half*)smem_raw;
    __half* s_weight = (__half*)(smem_raw + in_dim * sizeof(__half));

    const int in2 = in_dim / 2;
    const half2* input2 = (const half2*)input;
    half2* s_input2 = (half2*)s_input;
    for (int i = tid; i < in2; i += TMA_THREADS) {
        s_input2[i] = input2[i];
    }
    if ((in_dim & 1) && tid == 0) {
        s_input[in_dim - 1] = input[in_dim - 1];
    }
    __syncthreads();

    if (row >= out_dim) return;

    const int num_tiles = (in_dim + TMA_TILE_K - 1) / TMA_TILE_K;
    const __half* w_row = weight + (long long)row * in_dim;
    const int warp_buf_offset = warp_id * TMA_TILE_K;

    // Prefetch tile 0
    {
        const int tile_end = (TMA_TILE_K < in_dim) ? TMA_TILE_K : in_dim;
        const __half* src_base = w_row;
        __half* dst_base = s_weight + warp_buf_offset;
        const int num_cps = tile_end / 8;
        for (int c = lane_id; c < num_cps; c += TMA_WARP_SIZE) {
            cp_async_16(dst_base + c * 8, src_base + c * 8);
        }
        for (int i = num_cps * 8 + lane_id; i < tile_end; i += TMA_WARP_SIZE) {
            dst_base[i] = src_base[i];
        }
    }
    cp_async_commit();

    float acc = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        const int buf_cur = t & 1;
        const int buf_nxt = 1 - buf_cur;
        const int tile_start = t * TMA_TILE_K;
        const int tile_end = ((t + 1) * TMA_TILE_K < in_dim) ? (t + 1) * TMA_TILE_K : in_dim;
        const int tile_len = tile_end - tile_start;

        if (t + 1 < num_tiles) {
            const int nxt_start = (t + 1) * TMA_TILE_K;
            const int nxt_end = ((t + 2) * TMA_TILE_K < in_dim) ? (t + 2) * TMA_TILE_K : in_dim;
            const int nxt_len = nxt_end - nxt_start;
            const __half* src_base = w_row + nxt_start;
            __half* dst_base = s_weight + buf_nxt * TMA_RPB * TMA_TILE_K + warp_buf_offset;
            const int num_cps = nxt_len / 8;
            for (int c = lane_id; c < num_cps; c += TMA_WARP_SIZE) {
                cp_async_16(dst_base + c * 8, src_base + c * 8);
            }
            for (int i = num_cps * 8 + lane_id; i < nxt_len; i += TMA_WARP_SIZE) {
                dst_base[i] = src_base[i];
            }
            cp_async_commit();
        }

        cp_async_wait_group(t + 1 < num_tiles ? 1 : 0);
        __syncwarp();

        const __half* w_tile = s_weight + buf_cur * TMA_RPB * TMA_TILE_K + warp_buf_offset;
        const __half* in_tile = s_input + tile_start;
        const int tile_h2 = tile_len / 2;
        const half2* w_tile2 = (const half2*)w_tile;
        const half2* in_tile2 = (const half2*)in_tile;

        for (int i = lane_id; i < tile_h2; i += TMA_WARP_SIZE) {
            half2 w = w_tile2[i];
            half2 x = in_tile2[i];
            acc += __half2float(w.x) * __half2float(x.x) + __half2float(w.y) * __half2float(x.y);
        }
        if ((tile_len & 1) && lane_id == 0) {
            acc += __half2float(w_tile[tile_len - 1]) * __half2float(in_tile[tile_len - 1]);
        }
    }

    acc = warp_reduce_sum_tma(acc);

    if (lane_id == 0) {
        output[row] = __float2half(acc + __half2float(bias[row]));
    }
}
