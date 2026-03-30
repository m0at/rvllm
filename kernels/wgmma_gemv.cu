// Tensor-core GEMV kernel for M=1 decode using wmma (sm_80+).
//
// y[n] = weight[n, k] @ x[k]  (f16 I/O, f32 accumulation via tensor cores)
//
// Strategy: reshape GEMV as [16, K] x [K, 16] -> [16, 16] matmul.
// Input vector x[K] is broadcast to all 16 rows of A fragment.
// Weight rows are packed into 16-column B tiles.
// Only row 0 of the output accumulator contains the real dot products.
// Each warp computes 16 output elements via m16n16k16 tensor core ops.
//
// Compared to scalar GEMV (gemv_f16.cu) which does 2 FMA/thread/cycle,
// tensor cores do 16x16x16 = 4096 FMA per warp per instruction.
//
// Launch config:
//   Grid:  ((out_dim + BLOCK_N - 1) / BLOCK_N, 1, 1)  where BLOCK_N = 128
//   Block: (256, 1, 1) -- 8 warps, each handles 16 output rows
//   Shared mem: pad16(in_dim)*2 + 256*2 + 8*256*2 + 8*256*4 bytes
//     = pad16(in_dim)*2 + 512 + 4096 + 8192
//     e.g. in_dim=1536 -> 3072 + 512 + 4096 + 8192 = 15872 bytes
//
// Requires: sm_80+ (Ampere tensor cores), in_dim multiple of 16

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WGMMA_THREADS 256
#define WGMMA_WARPS (WGMMA_THREADS / 32)
// Each warp processes 16 output rows via one m16n16k16 tile
#define TILE_N 16
// Total output rows per block = WGMMA_WARPS * TILE_N = 128
#define BLOCK_N (WGMMA_WARPS * TILE_N)

// y[n] = weight[n, k] @ x[k]
// weight is [out_dim, in_dim] row-major
// For wmma: A = input broadcast [16, K], B = weight chunk [K, 16] (col-major view)
extern "C"
__global__ void __launch_bounds__(WGMMA_THREADS)
wgmma_gemv_f16_kernel(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [in_dim]
    const __half* __restrict__ weight, // [out_dim, in_dim] row-major
    int out_dim,
    int in_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Base output row for this warp
    const int block_base = blockIdx.x * BLOCK_N;
    const int warp_base = block_base + warp_id * TILE_N;

    // Load input vector into shared memory (all threads cooperate)
    extern __shared__ __half s_input[];

    for (int i = tid; i < in_dim; i += WGMMA_THREADS) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Early exit if this warp's output rows are out of bounds
    if (warp_base >= out_dim) return;

    // How many valid output rows this warp handles
    const int valid_rows = min(TILE_N, out_dim - warp_base);

    // Accumulator fragment: [16, 16] in f32
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Tile shared memory for loading A and B fragments
    // A fragment [16, 16]: input vector broadcast to 16 rows
    // B fragment [16, 16]: 16 weight rows, 16 columns of K
    //
    // We use a small per-warp staging area in shared memory.
    // Layout: after s_input[in_dim], we have staging space.
    // Each warp needs 16*16*2 = 512 bytes for A tile + 512 bytes for B tile.
    // But we can share the A tile across warps since it's the same input.
    //
    // Actually, wmma::load_matrix_sync can load from global/shared memory
    // directly. We need the data arranged correctly:
    //   A [16, 16] row-major: each row is x[k:k+16] (identical for all 16 rows)
    //   B [16, 16] col-major: column j is weight[warp_base+j, k:k+16]
    //
    // We'll stage A and B tiles in shared memory past the input vector.
    // Offset: s_input + in_dim (padded to 16 alignment)
    const int in_dim_pad = (in_dim + 15) & ~15;
    // Per-warp tile staging: A is shared (warp 0 writes), B is per-warp
    __half* s_a_tile = s_input + in_dim_pad;  // [16, 16] = 256 elements
    __half* s_b_tiles = s_a_tile + 256;       // [WGMMA_WARPS][16][16]
    __half* s_b_tile = s_b_tiles + warp_id * 256;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

    // Iterate over K dimension in tiles of 16
    for (int k = 0; k < in_dim; k += 16) {
        const int k_tile = min(16, in_dim - k);

        // Build A tile: broadcast input[k:k+16] to all 16 rows
        // Each of 32 threads in the warp fills part of the 16x16 tile
        // Total elements = 256, so each of 32 lanes fills 8 elements
        if (warp_id == 0) {
            // Only warp 0 builds A tile (shared across all warps)
            for (int i = lane_id; i < 256; i += 32) {
                int col = i % 16;
                s_a_tile[i] = (col < k_tile) ? s_input[k + col] : __float2half(0.0f);
            }
        }

        // Build B tile: weight[warp_base+j, k:k+16] as col-major [16, 16]
        // Col-major means B[row][col] = s_b_tile[col * 16 + row]
        // row = K index (0..15), col = output row within warp tile (0..15)
        // B[ki][nj] = weight[(warp_base + nj) * in_dim + (k + ki)]
        for (int i = lane_id; i < 256; i += 32) {
            int ki = i % 16;  // K index within tile
            int nj = i / 16;  // output row within warp tile
            int global_row = warp_base + nj;
            int global_k = k + ki;
            if (global_row < out_dim && global_k < in_dim) {
                s_b_tile[i] = weight[(long long)global_row * in_dim + global_k];
            } else {
                s_b_tile[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Load fragments and do tensor core MMA
        wmma::load_matrix_sync(a_frag, s_a_tile, 16);
        wmma::load_matrix_sync(b_frag, s_b_tile, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // Store accumulator to shared memory, then extract row 0
    // acc is [16, 16] row-major. Row 0 contains the 16 dot products.
    float* s_acc = (float*)(s_b_tiles + WGMMA_WARPS * 256);
    float* s_warp_acc = s_acc + warp_id * 256;

    wmma::store_matrix_sync(s_warp_acc, acc, 16, wmma::mem_row_major);
    // No sync needed -- only this warp reads its own data

    // Row 0 of acc has the results: s_warp_acc[0..15]
    // Lane 0 writes up to 16 outputs
    if (lane_id < valid_rows) {
        output[warp_base + lane_id] = __float2half(s_warp_acc[lane_id]);
    }
}

// Variant with 4x K-unrolling for larger hidden dims.
// Processes 4 k-tiles per iteration to better amortize shared mem staging.
extern "C"
__global__ void __launch_bounds__(WGMMA_THREADS)
wgmma_gemv_f16_unrolled_kernel(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [in_dim]
    const __half* __restrict__ weight, // [out_dim, in_dim] row-major
    int out_dim,
    int in_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int block_base = blockIdx.x * BLOCK_N;
    const int warp_base = block_base + warp_id * TILE_N;

    extern __shared__ __half s_input[];

    // Cooperative load of input vector
    for (int i = tid; i < in_dim; i += WGMMA_THREADS) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (warp_base >= out_dim) return;
    const int valid_rows = min(TILE_N, out_dim - warp_base);

    // For this variant, skip shared memory staging for B.
    // Load B directly from global memory into fragment.
    // A fragment is still broadcast from shared input.
    //
    // We stage A in registers by using fill_fragment manually.
    // wmma requires loading from memory, so we still need staging.

    const int in_dim_pad = (in_dim + 15) & ~15;
    __half* s_a_tile = s_input + in_dim_pad;
    // B tiles loaded directly from global memory -- need weight to be
    // arranged as col-major [16, 16] blocks. Since weight is row-major
    // [out_dim, in_dim], we need to transpose on the fly.
    // Unfortunately wmma::load_matrix_sync requires contiguous memory,
    // so we still stage B in shared memory.
    __half* s_b_tiles = s_a_tile + 256;
    __half* s_b_tile = s_b_tiles + warp_id * 256;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

    const int k_tiles = (in_dim + 15) / 16;

    for (int kt = 0; kt < k_tiles; kt++) {
        const int k = kt * 16;
        const int k_valid = min(16, in_dim - k);

        // Build A tile
        if (warp_id == 0) {
            for (int i = lane_id; i < 256; i += 32) {
                int col = i % 16;
                s_a_tile[i] = (col < k_valid) ? s_input[k + col] : __float2half(0.0f);
            }
        }

        // Build B tile -- vectorized with half2 when possible
        for (int i = lane_id; i < 256; i += 32) {
            int ki = i % 16;
            int nj = i / 16;
            int global_row = warp_base + nj;
            int global_k = k + ki;
            if (global_row < out_dim && ki < k_valid) {
                s_b_tile[i] = weight[(long long)global_row * in_dim + global_k];
            } else {
                s_b_tile[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, s_a_tile, 16);
        wmma::load_matrix_sync(b_frag, s_b_tile, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // Extract row 0 of accumulator
    float* s_acc = (float*)(s_b_tiles + WGMMA_WARPS * 256);
    float* s_warp_acc = s_acc + warp_id * 256;

    wmma::store_matrix_sync(s_warp_acc, acc, 16, wmma::mem_row_major);

    if (lane_id < valid_rows) {
        output[warp_base + lane_id] = __float2half(s_warp_acc[lane_id]);
    }
}

// Split-K variant: multiple blocks cooperate on one output chunk.
// Each block processes a slice of K, then atomicAdd partial results.
// Useful when in_dim >> out_dim (e.g., down_proj: in=8960, out=1536).
extern "C"
__global__ void __launch_bounds__(WGMMA_THREADS)
wgmma_gemv_f16_splitk_kernel(
    float* __restrict__ output_f32,    // [out_dim] f32 accumulator (zeroed before launch)
    const __half* __restrict__ input,  // [in_dim]
    const __half* __restrict__ weight, // [out_dim, in_dim] row-major
    int out_dim,
    int in_dim,
    int k_start,                       // start of K slice for this block
    int k_end                          // end of K slice for this block
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int block_base = blockIdx.x * BLOCK_N;
    const int warp_base = block_base + warp_id * TILE_N;

    const int k_len = k_end - k_start;

    extern __shared__ __half s_input[];

    // Load input slice into shared memory
    for (int i = tid; i < k_len; i += WGMMA_THREADS) {
        s_input[i] = input[k_start + i];
    }
    __syncthreads();

    if (warp_base >= out_dim) return;
    const int valid_rows = min(TILE_N, out_dim - warp_base);

    const int k_len_pad = (k_len + 15) & ~15;
    __half* s_a_tile = s_input + k_len_pad;
    __half* s_b_tiles = s_a_tile + 256;
    __half* s_b_tile = s_b_tiles + warp_id * 256;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

    for (int k = 0; k < k_len; k += 16) {
        const int k_valid = min(16, k_len - k);

        if (warp_id == 0) {
            for (int i = lane_id; i < 256; i += 32) {
                int col = i % 16;
                s_a_tile[i] = (col < k_valid) ? s_input[k + col] : __float2half(0.0f);
            }
        }

        for (int i = lane_id; i < 256; i += 32) {
            int ki = i % 16;
            int nj = i / 16;
            int global_row = warp_base + nj;
            int global_k = k_start + k + ki;
            if (global_row < out_dim && ki < k_valid) {
                s_b_tile[i] = weight[(long long)global_row * in_dim + global_k];
            } else {
                s_b_tile[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, s_a_tile, 16);
        wmma::load_matrix_sync(b_frag, s_b_tile, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // Extract row 0 and atomicAdd to output
    float* s_acc = (float*)(s_b_tiles + WGMMA_WARPS * 256);
    float* s_warp_acc = s_acc + warp_id * 256;

    wmma::store_matrix_sync(s_warp_acc, acc, 16, wmma::mem_row_major);

    if (lane_id < valid_rows) {
        atomicAdd(&output_f32[warp_base + lane_id], s_warp_acc[lane_id]);
    }
}

// Convert split-K f32 accumulator to f16 output
extern "C"
__global__ void splitk_f32_to_f16_kernel(
    __half* __restrict__ output,
    const float* __restrict__ input_f32,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input_f32[idx]);
    }
}
