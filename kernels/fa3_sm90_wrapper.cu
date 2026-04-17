// FA3 SM90 wrapper: thin extern "C" layer around Dao-AILab FlashAttention-3 hopper kernels.
// Compiled on H100 with CUTLASS headers; no ATen/PyTorch dependency.
//
// Provides paged-KV decode attention for SM90 using WGMMA/TMA.
// KV cache layout: [num_blocks, block_size, num_kv_heads, head_dim] (matches rvLLM).

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "heuristics.h"
#include "tile_size.h"

// Forward declarations of the instantiated templates we link against.
// Paged, non-split, PackGQA=true (matches vLLM decode path), fp16
template<> void run_mha_fwd_<90, cutlass::half_t, 128, 128, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, split, PackGQA=true (for low-batch high-seqlen), fp16
template<> void run_mha_fwd_<90, cutlass::half_t, 128, 128, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, non-split, PackGQA=true, e4m3 (FP8 KV)
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, split, PackGQA=true, e4m3 (FP8 KV)
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Combine kernel for split-KV (reduction is fp16 O regardless of Q/K/V dtype)
template<> void run_mha_fwd_combine_<cutlass::half_t, float, 128>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

// prepare_varlen_num_blocks from flash_prepare_scheduler.cu
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream,
                               bool packgqa, int blockM, int blockN, bool enable_pdl);

// Block sizes for hdim=128, fp16, PagedKVNonTMA=true, non-causal
static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;

// Round up helper
static inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

extern "C" int fa3_sm90_workspace_size(
    int batch_size,
    int num_heads,
    int max_num_splits  // pass 128 for safety
) {
    int b_rounded = round_multiple(batch_size, 4);
    // Scheduler metadata: up to 4 vectors of b_rounded ints + 1 semaphore
    int metadata_ints = b_rounded * 4 + 1;
    int metadata_bytes = metadata_ints * sizeof(int);
    // Align to 256 bytes
    metadata_bytes = round_multiple(metadata_bytes, 256);

    // softmax_lse: [batch, num_heads] floats (seqlen_q=1)
    int lse_bytes = batch_size * num_heads * sizeof(float);
    lse_bytes = round_multiple(lse_bytes, 256);

    // oaccum for split: [batch, num_splits, num_heads, head_dim] floats
    int oaccum_bytes = batch_size * max_num_splits * num_heads * 128 * sizeof(float);
    oaccum_bytes = round_multiple(oaccum_bytes, 256);

    // lseaccum for split: [batch, num_splits, num_heads] floats
    int lseaccum_bytes = batch_size * max_num_splits * num_heads * sizeof(float);
    lseaccum_bytes = round_multiple(lseaccum_bytes, 256);

    return metadata_bytes + lse_bytes + oaccum_bytes + lseaccum_bytes;
}

// Internal dispatcher: both fp16 KV and fp8 (e4m3) KV paths share param setup.
// Decode path: cu_seqlens_q == nullptr, max_seqlen_q == 1, total_q == batch_size.
// Prefill path: cu_seqlens_q is the per-seq Q offset prefix sum of length
// batch+1; max_seqlen_q is the longest seq's Q length; total_q is the
// sum; is_causal_prefill == true applies a causal mask so query t only
// sees K positions 0..=t within its own seq.
static int fa3_sm90_paged_decode_impl(
    void* q_ptr,
    void* k_cache_ptr,
    void* v_cache_ptr,
    void* o_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    int*  cu_seqlens_q_ptr,   // nullptr for decode
    int   max_seqlen_q,       // 1 for decode
    int   total_q,            // batch_size for decode, sum of Q lens for prefill
    void* workspace_ptr,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    bool  is_fp8,
    bool  is_causal_prefill,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    cudaStream_t stream
) {
    if (head_dim != 128) {
        fprintf(stderr, "fa3_sm90_paged_decode: only head_dim=128 supported, got %d\n", head_dim);
        return -1;
    }

    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int arch = props.major * 10 + props.minor;
    int num_sm = props.multiProcessorCount;

    // Compute workspace layout
    int b_rounded = round_multiple(batch_size, 4);
    int metadata_ints = b_rounded * 4 + 1;
    int metadata_bytes = round_multiple(metadata_ints * (int)sizeof(int), 256);
    int lse_bytes = round_multiple(batch_size * num_heads * (int)sizeof(float), 256);

    char* ws = (char*)workspace_ptr;
    int* metadata_ptr = (int*)ws;
    float* lse_ptr = (float*)(ws + metadata_bytes);
    float* oaccum_ptr = (float*)(ws + metadata_bytes + lse_bytes);
    int oaccum_bytes = round_multiple(batch_size * 128 * num_heads * 128 * (int)sizeof(float), 256);
    float* lseaccum_ptr = (float*)(ws + metadata_bytes + lse_bytes + oaccum_bytes);

    // Zero the metadata region (semaphore must start at 0)
    cudaMemsetAsync(metadata_ptr, 0, metadata_bytes, stream);

    // Populate Flash_fwd_params
    Flash_fwd_params params = {};

    params.is_bf16 = false;
    params.is_fp32 = false;
    params.is_e4m3 = is_fp8;
    // Per-tensor FP8 descale: scalar broadcast (strides all zero).
    params.q_descale_ptr = is_fp8 ? q_descale_ptr : nullptr;
    params.k_descale_ptr = is_fp8 ? k_descale_ptr : nullptr;
    params.v_descale_ptr = is_fp8 ? v_descale_ptr : nullptr;
    params.q_descale_batch_stride = 0;
    params.q_descale_head_stride  = 0;
    params.k_descale_batch_stride = 0;
    params.k_descale_head_stride  = 0;
    params.v_descale_batch_stride = 0;
    params.v_descale_head_stride  = 0;

    // Q: decode is [batch, num_heads, head_dim] (1 row per seq).
    // Prefill is [total_q, num_heads, head_dim] indexed via cu_seqlens_q.
    params.q_ptr = q_ptr;
    params.q_batch_stride = num_heads * head_dim;
    params.q_row_stride = num_heads * head_dim;
    params.q_head_stride = head_dim;

    // K: [num_blocks_total, block_size, num_kv_heads, head_dim]
    params.k_ptr = k_cache_ptr;
    params.k_batch_stride = block_size * num_kv_heads * head_dim;  // stride between pages
    params.k_row_stride = num_kv_heads * head_dim;  // stride between tokens in page
    params.k_head_stride = head_dim;

    // V: same layout as K
    params.v_ptr = v_cache_ptr;
    params.v_batch_stride = block_size * num_kv_heads * head_dim;
    params.v_row_stride = num_kv_heads * head_dim;
    params.v_head_stride = head_dim;
    params.v_dim_stride = 1;  // contiguous in head_dim

    // O: [batch, num_heads, head_dim] treated as [batch, 1, num_heads, head_dim]
    params.o_ptr = o_ptr;
    params.o_batch_stride = num_heads * head_dim;
    params.o_row_stride = num_heads * head_dim;
    params.o_head_stride = head_dim;

    // Dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.d = head_dim;
    params.d_rounded = 128;
    params.dv = head_dim;
    params.dv_rounded = 128;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_blocks_per_seq * block_size;
    params.seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    params.seqlen_k_rounded = round_multiple(params.seqlen_k, kBlockN);
    params.total_q = total_q;  // decode: batch; prefill: sum of per-seq Q lens

    // Paged KV
    params.page_table = block_tables_ptr;
    params.page_table_batch_stride = max_blocks_per_seq;
    params.page_size = block_size;
    params.num_pages = num_blocks_total;
    params.pagedkv_tma = false;

    // Varlen via seqused_k (actual context lengths per sequence)
    params.seqused_k = context_lens_ptr;
    params.cu_seqlens_q = cu_seqlens_q_ptr;   // nullptr for decode
    params.cu_seqlens_k = nullptr;
    params.cu_seqlens_knew = nullptr;
    params.seqused_q = nullptr;
    params.leftpad_k = nullptr;

    // No KV append
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.seqlen_knew = 0;
    params.total_knew = 0;

    // No QV
    params.qv_ptr = nullptr;

    // No rotary (applied before attention in rvLLM)
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.seqlens_rotary = nullptr;
    params.rotary_dim = 0;
    params.is_rotary_interleaved = false;

    // No KV batch indexing
    params.kv_batch_idx = nullptr;
    params.b_k = batch_size;

    // Softmax
    params.scale_softmax = scale;
    params.softcap = 0.0f;

    // No dropout
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    // Decode: seqlen_q == 1, non-causal (query sees all prior KV).
    // Prefill: causal self-attention over Q-then-KV, query t sees K 0..=t.
    params.is_causal = is_causal_prefill;
    params.is_local = false;
    params.window_size_left = params.seqlen_k - 1;
    params.window_size_right = is_causal_prefill ? 0 : 0;  // right=0 in both
    params.attention_chunk = 0;

    // Architecture
    params.arch = arch;
    params.num_sm = num_sm;

    // LSE output
    params.softmax_lse_ptr = lse_ptr;

    // PackGQA = true (matches template)
    params.pack_gqa = true;

    // Determine num_splits
    int qhead_per_khead = num_heads / num_kv_heads;
    // Decode: 1 query token per seq. Prefill: max_seqlen_q query tokens.
    int num_m_blocks = (max_seqlen_q * qhead_per_khead + kBlockM - 1) / kBlockM;
    int num_n_blocks = (params.seqlen_k + kBlockN - 1) / kBlockN;
    int total_mblocks = batch_size * num_kv_heads * num_m_blocks;
    int kv_bytes_per_elem = is_fp8 ? 1 : 2;
    int size_one_kv_head = params.seqlen_k * head_dim * 2 * kv_bytes_per_elem;  // K+V
    int ns = num_splits_heuristic(total_mblocks, num_sm, num_n_blocks,
                                  num_m_blocks, size_one_kv_head,
                                  false /*is_causal_or_local*/, 128);
    params.num_splits = ns;
    bool use_split = ns > 1;

    // Scheduler metadata setup — matches upstream hopper/flash_api.cpp:
    //   varlen_sort_batches = !is_local
    //   head_swizzle        = is_causal || is_local
    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle        = params.is_causal || params.is_local;
    int num_vectors = 2;  // num_splits_dynamic + num_m_blocks (always for prepare_varlen)
    if (params.varlen_sort_batches) num_vectors += 1;  // varlen_batch_idx
    if (params.head_swizzle) num_vectors += 1;          // num_nheads_in_l2

    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    int semaphore_offset = b_rounded * num_vectors;

    params.num_splits_dynamic_ptr = metadata_ptr;
    params.num_m_blocks_ptr = metadata_ptr + b_rounded;
    params.varlen_batch_idx_ptr = params.varlen_sort_batches ? metadata_ptr + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = metadata_ptr + semaphore_offset;
    params.tile_count_semaphore_offset = semaphore_offset;

    params.skip_scheduler_metadata_computation = false;
    params.prepare_varlen_pdl = false;

    // Split-KV output buffers
    if (use_split) {
        // oaccum: [batch, num_splits, num_heads, head_dim] float
        params.oaccum_ptr = oaccum_ptr;
        params.oaccum_split_stride = num_heads * head_dim;
        params.oaccum_batch_stride = ns * num_heads * head_dim;
        params.oaccum_row_stride = num_heads * head_dim;
        params.oaccum_head_stride = head_dim;

        // lseaccum: [num_splits, batch, num_heads] float
        params.softmax_lseaccum_ptr = lseaccum_ptr;
        params.lseaccum_split_stride = batch_size * num_heads;
        params.lseaccum_batch_stride = num_heads;
        params.lseaccum_head_stride = 1;
    } else {
        params.oaccum_ptr = nullptr;
        params.softmax_lseaccum_ptr = nullptr;
    }

    params.rng_state = nullptr;

    // Launch the kernel. FP8 and FP16 KV use different template instantiations;
    // the combine kernel is fp16-output only, reused by both.
    if (is_fp8) {
        if (use_split) {
            run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true>(params, stream);
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, false);
        } else {
            run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true>(params, stream);
        }
    } else {
        if (use_split) {
            run_mha_fwd_<90, cutlass::half_t, 128, 128, true, true, false, true>(params, stream);
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, false);
        } else {
            run_mha_fwd_<90, cutlass::half_t, 128, 128, false, true, false, true>(params, stream);
        }
    }

    return 0;
}

extern "C" {

// FP16 KV path (unchanged ABI): calls the shared impl with is_fp8=false.
int fa3_sm90_paged_decode(
    void* q_ptr,
    void* k_cache_ptr,
    void* v_cache_ptr,
    void* o_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    void* workspace_ptr,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    cudaStream_t stream
) {
    return fa3_sm90_paged_decode_impl(
        q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
        block_tables_ptr, context_lens_ptr,
        /*cu_seqlens_q=*/nullptr, /*max_seqlen_q=*/1, /*total_q=*/batch_size,
        workspace_ptr,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/false, /*is_causal_prefill=*/false,
        /*q_descale=*/nullptr, /*k_descale=*/nullptr, /*v_descale=*/nullptr,
        stream);
}

// FP8 E4M3 KV path. Q / K cache / V cache are FP8 (1 byte/elem).
// q_descale / k_descale / v_descale point at single-scalar f32 scales on device.
// O is fp16 (FA3 E4M3 kernels dequant inside softmax and write fp16 output).
int fa3_sm90_paged_decode_fp8(
    void* q_fp8_ptr,
    void* k_cache_fp8_ptr,
    void* v_cache_fp8_ptr,
    void* o_f16_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    void* workspace_ptr,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    cudaStream_t stream
) {
    return fa3_sm90_paged_decode_impl(
        q_fp8_ptr, k_cache_fp8_ptr, v_cache_fp8_ptr, o_f16_ptr,
        block_tables_ptr, context_lens_ptr,
        /*cu_seqlens_q=*/nullptr, /*max_seqlen_q=*/1, /*total_q=*/batch_size,
        workspace_ptr,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/true, /*is_causal_prefill=*/false,
        q_descale_ptr, k_descale_ptr, v_descale_ptr,
        stream);
}

// FP8 E4M3 paged PREFILL: Q / K cache / V cache are FP8. cu_seqlens_q
// gives per-seq offsets in a varlen [total_q, num_heads, head_dim] Q
// tensor; max_seqlen_q is the longest per-seq Q length. Causal
// self-attention (query t only sees K 0..=t within its own seq).
int fa3_sm90_paged_prefill_fp8(
    void* q_fp8_ptr,
    void* k_cache_fp8_ptr,
    void* v_cache_fp8_ptr,
    void* o_f16_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    int*  cu_seqlens_q_ptr,
    void* workspace_ptr,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    float scale,
    int   total_q,
    int   max_seqlen_q,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    cudaStream_t stream
) {
    return fa3_sm90_paged_decode_impl(
        q_fp8_ptr, k_cache_fp8_ptr, v_cache_fp8_ptr, o_f16_ptr,
        block_tables_ptr, context_lens_ptr,
        cu_seqlens_q_ptr, max_seqlen_q, total_q,
        workspace_ptr,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/true, /*is_causal_prefill=*/true,
        q_descale_ptr, k_descale_ptr, v_descale_ptr,
        stream);
}

}  // extern "C"
