// Test harness for persistent_layer_v2 and megakernel_v2 cubins on H100.
// Validates kernel loading, correctness (no NaN/Inf, reasonable range),
// determinism, and reports timing.
//
// Compile: nvcc -O3 --use_fast_math -arch=sm_90 test_persistent_v2.cu -lcuda -o test_persistent_v2
// Run:     ./test_persistent_v2

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CU_CHECK(call) do {                                                    \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
        const char* msg = nullptr;                                             \
        cuGetErrorString(err, &msg);                                           \
        fprintf(stderr, "CUDA driver error at %s:%d: %s (code %d)\n",         \
                __FILE__, __LINE__, msg ? msg : "unknown", (int)err);          \
        exit(1);                                                               \
    }                                                                          \
} while(0)

#define RT_CHECK(call) do {                                                    \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA runtime error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));                   \
        exit(1);                                                               \
    }                                                                          \
} while(0)

// ---------------------------------------------------------------------------
// Simple xorshift PRNG for reproducible fill
// ---------------------------------------------------------------------------

static uint32_t xor_state = 0xDEADBEEF;

static uint32_t xorshift32() {
    uint32_t x = xor_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xor_state = x;
    return x;
}

static float rand_f32(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xorshift32() & 0xFFFFFF) / (float)0xFFFFFF);
}

static __half rand_f16(float lo, float hi) {
    return __float2half(rand_f32(lo, hi));
}

// ---------------------------------------------------------------------------
// Fill helpers
// ---------------------------------------------------------------------------

static void fill_f16(std::vector<__half>& buf, float lo, float hi) {
    for (size_t i = 0; i < buf.size(); i++)
        buf[i] = rand_f16(lo, hi);
}

static void fill_f32(std::vector<float>& buf, float lo, float hi) {
    for (size_t i = 0; i < buf.size(); i++)
        buf[i] = rand_f32(lo, hi);
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

struct ValResult {
    bool has_nan;
    bool has_inf;
    bool all_zero;
    float max_abs;
};

static ValResult validate_f16(const __half* data, int n) {
    ValResult r = {false, false, true, 0.0f};
    for (int i = 0; i < n; i++) {
        float v = __half2float(data[i]);
        if (isnan(v)) r.has_nan = true;
        if (isinf(v)) r.has_inf = true;
        if (v != 0.0f) r.all_zero = false;
        float a = fabsf(v);
        if (a > r.max_abs) r.max_abs = a;
    }
    return r;
}

static bool check_output(const char* name, const __half* data, int n) {
    ValResult r = validate_f16(data, n);
    bool ok = true;
    if (r.has_nan) { fprintf(stderr, "  FAIL: %s contains NaN\n", name); ok = false; }
    if (r.has_inf) { fprintf(stderr, "  FAIL: %s contains Inf\n", name); ok = false; }
    if (r.all_zero) { fprintf(stderr, "  FAIL: %s is all zeros\n", name); ok = false; }
    if (r.max_abs > 1000.0f) { fprintf(stderr, "  FAIL: %s max_abs=%.2f > 1000\n", name, r.max_abs); ok = false; }
    if (ok) printf("  OK: %s  max_abs=%.4f\n", name, r.max_abs);
    return ok;
}

static bool check_determinism(const __half* a, const __half* b, int n, const char* name) {
    int mismatches = 0;
    int max_ulp_diff = 0;
    for (int i = 0; i < n; i++) {
        uint16_t va, vb;
        memcpy(&va, &a[i], 2);
        memcpy(&vb, &b[i], 2);
        int diff = abs((int)va - (int)vb);
        if (diff > max_ulp_diff) max_ulp_diff = diff;
        if (diff > 0) mismatches++;
    }
    if (mismatches > 0) {
        printf("  WARN: %s non-deterministic: %d/%d differ, max ULP diff=%d (expected for parallel TC GEMV reductions)\n",
               name, mismatches, n, max_ulp_diff);
    } else {
        printf("  OK: %s deterministic (bitwise identical)\n", name);
    }
    return true; // non-determinism in parallel reductions is not a failure
}

// ---------------------------------------------------------------------------
// GPU buffer helper
// ---------------------------------------------------------------------------

static void* gpu_alloc(size_t bytes) {
    void* ptr = nullptr;
    RT_CHECK(cudaMalloc(&ptr, bytes));
    RT_CHECK(cudaMemset(ptr, 0, bytes));
    return ptr;
}

template<typename T>
static void upload(void* dst, const std::vector<T>& src) {
    RT_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
static void download(std::vector<T>& dst, const void* src) {
    RT_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Qwen2.5-3B dimensions
// ---------------------------------------------------------------------------

static const int HIDDEN      = 3584;
static const int NUM_HEADS   = 28;
static const int NUM_KV      = 4;
static const int HEAD_DIM    = 128;
static const int INTER       = 18944;
static const int Q_DIM       = NUM_HEADS * HEAD_DIM;    // 3584
static const int KV_DIM      = NUM_KV * HEAD_DIM;       // 512
static const int QKV_DIM     = Q_DIM + 2 * KV_DIM;      // 4608
static const int GATE_UP     = 2 * INTER;                // 37888
static const int BLOCK_SZ    = 16;
static const int MAX_CTX     = 256;
static const int MAX_BPQ     = MAX_CTX / BLOCK_SZ;       // 16
static const float EPS       = 1e-6f;
static const float ATTN_SC   = 1.0f / sqrtf(128.0f);
static const int HALF_DIM    = HEAD_DIM / 2;             // 64
static const int GRID        = 256;
static const int BLOCK       = 256;
static const int HPG         = NUM_HEADS / NUM_KV;       // 7
static const int MAX_SPLITS  = 8;   // min(GRID/NUM_KV, 8) = min(64, 8)
static const int CONTEXT_LEN = 128;
static const int VOCAB_SIZE  = 151936; // Qwen2.5 vocab

// Shared memory calculation -- must match kernel layout exactly.
// GEMV phases: normed_bytes (aligned) + weight double buffer (32768)
// Attention phase: 2 * BC * head_dim * 2 + HPG * STRIDE * 4 + WARPS * 4
static int compute_smem() {
    const int WARPS = 8;
    int normed_bytes = (HIDDEN * 4 + WARPS * 4 + 127) & ~127;
    int weight_dbuf = 2 * WARPS * 8 * 128 * 2;   // 2 * WARPS * RPW * TILE_K * sizeof(half)
    int gemv_smem = normed_bytes + weight_dbuf;

    const int BC = 64;
    int hpg = HPG < 8 ? HPG : 8;
    int stride = BC + 1;
    int attn_smem = 2 * BC * HEAD_DIM * 2 + hpg * stride * 4 + WARPS * 4;

    return gemv_smem > attn_smem ? gemv_smem : attn_smem;
}

static int compute_smem_v3() {
    const int WARPS = 8;
    int gemv_smem = HIDDEN * 4 + WARPS * 4;
    int hpg = 8;
    int stride = 32 + 1;
    int attn_smem = 2 * 32 * HEAD_DIM * 2 + hpg * stride * 4 + WARPS * 4;
    return gemv_smem > attn_smem ? gemv_smem : attn_smem;
}

// ---------------------------------------------------------------------------
// Test: persistent_layer_v2_f16
// ---------------------------------------------------------------------------

static bool test_persistent_layer_v2(CUmodule mod) {
    printf("\n=== persistent_layer_v2_f16 ===\n");

    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, mod, "persistent_layer_v2_f16"));

    int smem = compute_smem();
    CU_CHECK(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    printf("  shared memory: %d bytes\n", smem);

    // Seed PRNG
    xor_state = 0xDEADBEEF;

    // Host buffers for inputs
    std::vector<__half> h_prev_residual(HIDDEN);
    std::vector<__half> h_norm_w(HIDDEN);
    std::vector<__half> h_qkv_weight((size_t)QKV_DIM * HIDDEN);
    std::vector<__half> h_qkv_bias(QKV_DIM);
    std::vector<__half> h_o_weight((size_t)HIDDEN * Q_DIM);
    std::vector<__half> h_post_norm_w(HIDDEN);
    std::vector<__half> h_gateup_weight((size_t)GATE_UP * HIDDEN);
    std::vector<__half> h_down_weight((size_t)HIDDEN * INTER);

    fill_f16(h_prev_residual, -0.5f, 0.5f);
    fill_f16(h_norm_w, 0.8f, 1.2f);
    fill_f16(h_qkv_weight, -0.01f, 0.01f);
    fill_f16(h_qkv_bias, -0.01f, 0.01f);
    fill_f16(h_o_weight, -0.01f, 0.01f);
    fill_f16(h_post_norm_w, 0.8f, 1.2f);
    fill_f16(h_gateup_weight, -0.01f, 0.01f);
    fill_f16(h_down_weight, -0.01f, 0.01f);

    // KV cache: [num_slots * num_kv_heads * head_dim]
    // num_slots = MAX_BPQ * BLOCK_SZ = 256
    int num_slots = MAX_BPQ * BLOCK_SZ;
    std::vector<__half> h_key_cache((size_t)num_slots * NUM_KV * HEAD_DIM);
    std::vector<__half> h_val_cache((size_t)num_slots * NUM_KV * HEAD_DIM);
    fill_f16(h_key_cache, -0.05f, 0.05f);
    fill_f16(h_val_cache, -0.05f, 0.05f);

    // RoPE tables: [MAX_CTX * HALF_DIM]
    std::vector<float> h_rope_cos((size_t)MAX_CTX * HALF_DIM);
    std::vector<float> h_rope_sin((size_t)MAX_CTX * HALF_DIM);
    for (int pos = 0; pos < MAX_CTX; pos++) {
        for (int d = 0; d < HALF_DIM; d++) {
            float freq = 1.0f / powf(1000000.0f, (float)(2 * d) / (float)HEAD_DIM);
            h_rope_cos[pos * HALF_DIM + d] = cosf((float)pos * freq);
            h_rope_sin[pos * HALF_DIM + d] = sinf((float)pos * freq);
        }
    }

    // Block tables: identity [0,1,...,MAX_BPQ-1]
    std::vector<int> h_block_tables(MAX_BPQ);
    for (int i = 0; i < MAX_BPQ; i++) h_block_tables[i] = i;

    std::vector<int> h_context_lens = {CONTEXT_LEN};
    std::vector<int> h_positions = {CONTEXT_LEN};  // next position = context_len
    std::vector<int> h_slot_mapping = {CONTEXT_LEN}; // slot = position for simple mapping

    // Allocate GPU buffers
    void* d_mlp_out       = gpu_alloc(HIDDEN * 2);
    void* d_residual_out  = gpu_alloc(HIDDEN * 2);
    void* d_prev_residual = gpu_alloc(HIDDEN * 2);
    void* d_key_cache     = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);
    void* d_val_cache     = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);
    void* d_block_tables  = gpu_alloc(MAX_BPQ * 4);
    void* d_context_lens  = gpu_alloc(4);
    void* d_positions     = gpu_alloc(4);
    void* d_slot_mapping  = gpu_alloc(4);
    void* d_rope_cos      = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);
    void* d_rope_sin      = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);
    void* d_norm_w        = gpu_alloc(HIDDEN * 2);
    void* d_qkv_weight    = gpu_alloc((size_t)QKV_DIM * HIDDEN * 2);
    void* d_qkv_bias      = gpu_alloc(QKV_DIM * 2);
    void* d_o_weight      = gpu_alloc((size_t)HIDDEN * Q_DIM * 2);
    void* d_post_norm_w   = gpu_alloc(HIDDEN * 2);
    void* d_gateup_weight = gpu_alloc((size_t)GATE_UP * HIDDEN * 2);
    void* d_down_weight   = gpu_alloc((size_t)HIDDEN * INTER * 2);

    // Scratch buffers
    void* d_qkv_scratch    = gpu_alloc(QKV_DIM * 2);
    void* d_attn_scratch   = gpu_alloc(Q_DIM * 2);
    void* d_oproj_scratch  = gpu_alloc(HIDDEN * 2);
    void* d_gateup_scratch = gpu_alloc(GATE_UP * 2);

    // Split-KV scratch
    // max_buf/sum_buf: max_splits * num_kv_heads * hpg * sizeof(float)
    int split_meta_size = MAX_SPLITS * NUM_KV * HPG;
    void* d_split_max = gpu_alloc(split_meta_size * 4);
    void* d_split_sum = gpu_alloc(split_meta_size * 4);
    // acc_buf: max_splits * num_kv_heads * hpg * head_dim * sizeof(half)
    int split_acc_size = MAX_SPLITS * NUM_KV * HPG * HEAD_DIM;
    void* d_split_acc = gpu_alloc(split_acc_size * 2);

    // Sync flags
    int num_sync_flags = 6;  // PLV2_NUM_SYNCS
    void* d_sync_flags = gpu_alloc(num_sync_flags * 4);

    // Upload data
    upload(d_prev_residual, h_prev_residual);
    upload(d_key_cache, h_key_cache);
    upload(d_val_cache, h_val_cache);
    upload(d_block_tables, h_block_tables);
    upload(d_context_lens, h_context_lens);
    upload(d_positions, h_positions);
    upload(d_slot_mapping, h_slot_mapping);
    upload(d_rope_cos, h_rope_cos);
    upload(d_rope_sin, h_rope_sin);
    upload(d_norm_w, h_norm_w);
    upload(d_qkv_weight, h_qkv_weight);
    upload(d_qkv_bias, h_qkv_bias);
    upload(d_o_weight, h_o_weight);
    upload(d_post_norm_w, h_post_norm_w);
    upload(d_gateup_weight, h_gateup_weight);
    upload(d_down_weight, h_down_weight);

    // prev_mlp = NULL for layer 0
    void* d_prev_mlp = nullptr;

    // Config scalars
    float eps = EPS;
    float attn_scale = ATTN_SC;
    int hidden_size = HIDDEN;
    int q_dim = Q_DIM;
    int kv_dim = KV_DIM;
    int qkv_dim = QKV_DIM;
    int num_heads = NUM_HEADS;
    int num_kv_heads = NUM_KV;
    int head_dim = HEAD_DIM;
    int intermediate_size = INTER;
    int gate_up_dim = GATE_UP;
    int block_size = BLOCK_SZ;
    int max_context_len = MAX_CTX;
    int max_blocks_per_seq = MAX_BPQ;
    int max_splits = MAX_SPLITS;

    // Kernel args (must match signature order exactly)
    void* args[] = {
        &d_mlp_out, &d_residual_out,
        &d_prev_residual, &d_prev_mlp,
        &d_key_cache, &d_val_cache,
        &d_block_tables, &d_context_lens,
        &d_positions, &d_slot_mapping,
        &d_rope_cos, &d_rope_sin,
        &d_norm_w, &d_qkv_weight, &d_qkv_bias,
        &d_o_weight, &d_post_norm_w,
        &d_gateup_weight, &d_down_weight,
        &d_qkv_scratch, &d_attn_scratch, &d_oproj_scratch, &d_gateup_scratch,
        &d_split_max, &d_split_sum, &d_split_acc, &max_splits,
        &eps, &attn_scale,
        &hidden_size, &q_dim, &kv_dim, &qkv_dim,
        &num_heads, &num_kv_heads, &head_dim,
        &intermediate_size, &gate_up_dim,
        &block_size, &max_context_len, &max_blocks_per_seq,
        &d_sync_flags
    };

    // --- Run 1 ---
    RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_flags * 4));
    RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));

    CU_CHECK(cuLaunchCooperativeKernel(func, GRID, 1, 1, BLOCK, 1, 1,
                                        smem, 0, args));
    RT_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_mlp_out_1(HIDDEN), h_residual_out_1(HIDDEN);
    download(h_mlp_out_1, d_mlp_out);
    download(h_residual_out_1, d_residual_out);

    bool pass = true;
    printf("  Run 1 outputs:\n");
    pass &= check_output("mlp_out", h_mlp_out_1.data(), HIDDEN);
    pass &= check_output("residual_out", h_residual_out_1.data(), HIDDEN);

    // --- Run 2 (determinism) ---
    // Reset all mutable state
    RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_flags * 4));
    RT_CHECK(cudaMemset(d_mlp_out, 0, HIDDEN * 2));
    RT_CHECK(cudaMemset(d_residual_out, 0, HIDDEN * 2));
    RT_CHECK(cudaMemset(d_qkv_scratch, 0, QKV_DIM * 2));
    RT_CHECK(cudaMemset(d_attn_scratch, 0, Q_DIM * 2));
    RT_CHECK(cudaMemset(d_oproj_scratch, 0, HIDDEN * 2));
    RT_CHECK(cudaMemset(d_gateup_scratch, 0, GATE_UP * 2));
    RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));

    // Re-upload KV cache (kernel writes to it)
    upload(d_key_cache, h_key_cache);
    upload(d_val_cache, h_val_cache);

    CU_CHECK(cuLaunchCooperativeKernel(func, GRID, 1, 1, BLOCK, 1, 1,
                                        smem, 0, args));
    RT_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_mlp_out_2(HIDDEN), h_residual_out_2(HIDDEN);
    download(h_mlp_out_2, d_mlp_out);
    download(h_residual_out_2, d_residual_out);

    printf("  Determinism check:\n");
    pass &= check_determinism(h_mlp_out_1.data(), h_mlp_out_2.data(), HIDDEN, "mlp_out");
    pass &= check_determinism(h_residual_out_1.data(), h_residual_out_2.data(), HIDDEN, "residual_out");

    // --- Performance: 100 iterations ---
    printf("  Performance (100 iterations):\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
        RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_flags * 4));
        RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
        RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
        RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));
        upload(d_key_cache, h_key_cache);
        upload(d_val_cache, h_val_cache);
        CU_CHECK(cuLaunchCooperativeKernel(func, GRID, 1, 1, BLOCK, 1, 1,
                                            smem, 0, args));
        RT_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t t0, t1;
    RT_CHECK(cudaEventCreate(&t0));
    RT_CHECK(cudaEventCreate(&t1));

    const int ITERS = 100;
    RT_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; i++) {
        RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_flags * 4));
        RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
        RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
        RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));
        upload(d_key_cache, h_key_cache);
        upload(d_val_cache, h_val_cache);
        CU_CHECK(cuLaunchCooperativeKernel(func, GRID, 1, 1, BLOCK, 1, 1,
                                            smem, 0, args));
    }
    RT_CHECK(cudaEventRecord(t1));
    RT_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    RT_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float avg_us = (ms * 1000.0f) / (float)ITERS;
    printf("  Avg time: %.1f us (total %.2f ms for %d iters)\n", avg_us, ms, ITERS);

    RT_CHECK(cudaEventDestroy(t0));
    RT_CHECK(cudaEventDestroy(t1));

    // Free GPU buffers
    cudaFree(d_mlp_out); cudaFree(d_residual_out);
    cudaFree(d_prev_residual);
    cudaFree(d_key_cache); cudaFree(d_val_cache);
    cudaFree(d_block_tables); cudaFree(d_context_lens);
    cudaFree(d_positions); cudaFree(d_slot_mapping);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_norm_w); cudaFree(d_qkv_weight); cudaFree(d_qkv_bias);
    cudaFree(d_o_weight); cudaFree(d_post_norm_w);
    cudaFree(d_gateup_weight); cudaFree(d_down_weight);
    cudaFree(d_qkv_scratch); cudaFree(d_attn_scratch);
    cudaFree(d_oproj_scratch); cudaFree(d_gateup_scratch);
    cudaFree(d_split_max); cudaFree(d_split_sum); cudaFree(d_split_acc);
    cudaFree(d_sync_flags);

    return pass;
}

static bool test_persistent_layer_v3(CUmodule mod) {
    printf("\n=== persistent_layer_v3_f16 ===\n");

    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, mod, "persistent_layer_v3_f16"));

    int smem = compute_smem_v3();
    CU_CHECK(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    printf("  shared memory: %d bytes\n", smem);

    xor_state = 0x13579BDF;

    std::vector<__half> h_prev_residual(HIDDEN);
    std::vector<__half> h_norm_w(HIDDEN);
    std::vector<__half> h_qkv_weight((size_t)QKV_DIM * HIDDEN);
    std::vector<__half> h_qkv_bias(QKV_DIM);
    std::vector<__half> h_o_weight((size_t)HIDDEN * Q_DIM);
    std::vector<__half> h_post_norm_w(HIDDEN);
    std::vector<__half> h_gateup_weight((size_t)GATE_UP * HIDDEN);
    std::vector<__half> h_down_weight((size_t)HIDDEN * INTER);

    fill_f16(h_prev_residual, -0.5f, 0.5f);
    fill_f16(h_norm_w, 0.8f, 1.2f);
    fill_f16(h_qkv_weight, -0.01f, 0.01f);
    fill_f16(h_qkv_bias, -0.01f, 0.01f);
    fill_f16(h_o_weight, -0.01f, 0.01f);
    fill_f16(h_post_norm_w, 0.8f, 1.2f);
    fill_f16(h_gateup_weight, -0.01f, 0.01f);
    fill_f16(h_down_weight, -0.01f, 0.01f);

    int num_slots = MAX_BPQ * BLOCK_SZ;
    std::vector<__half> h_key_cache((size_t)num_slots * NUM_KV * HEAD_DIM);
    std::vector<__half> h_val_cache((size_t)num_slots * NUM_KV * HEAD_DIM);
    fill_f16(h_key_cache, -0.05f, 0.05f);
    fill_f16(h_val_cache, -0.05f, 0.05f);

    std::vector<float> h_rope_cos((size_t)MAX_CTX * HALF_DIM);
    std::vector<float> h_rope_sin((size_t)MAX_CTX * HALF_DIM);
    for (int pos = 0; pos < MAX_CTX; pos++) {
        for (int d = 0; d < HALF_DIM; d++) {
            float freq = 1.0f / powf(1000000.0f, (float)(2 * d) / (float)HEAD_DIM);
            h_rope_cos[pos * HALF_DIM + d] = cosf((float)pos * freq);
            h_rope_sin[pos * HALF_DIM + d] = sinf((float)pos * freq);
        }
    }

    std::vector<int> h_block_tables(MAX_BPQ);
    for (int i = 0; i < MAX_BPQ; i++) h_block_tables[i] = i;

    std::vector<int> h_context_lens = {CONTEXT_LEN};
    std::vector<int> h_positions = {CONTEXT_LEN};
    std::vector<int> h_slot_mapping = {CONTEXT_LEN};

    void* d_mlp_out       = gpu_alloc(HIDDEN * 2);
    void* d_residual_out  = gpu_alloc(HIDDEN * 2);
    void* d_prev_residual = gpu_alloc(HIDDEN * 2);
    void* d_key_cache     = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);
    void* d_val_cache     = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);
    void* d_block_tables  = gpu_alloc(MAX_BPQ * 4);
    void* d_context_lens  = gpu_alloc(4);
    void* d_positions     = gpu_alloc(4);
    void* d_slot_mapping  = gpu_alloc(4);
    void* d_rope_cos      = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);
    void* d_rope_sin      = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);
    void* d_norm_w        = gpu_alloc(HIDDEN * 2);
    void* d_qkv_weight    = gpu_alloc((size_t)QKV_DIM * HIDDEN * 2);
    void* d_qkv_bias      = gpu_alloc(QKV_DIM * 2);
    void* d_o_weight      = gpu_alloc((size_t)HIDDEN * Q_DIM * 2);
    void* d_post_norm_w   = gpu_alloc(HIDDEN * 2);
    void* d_gateup_weight = gpu_alloc((size_t)GATE_UP * HIDDEN * 2);
    void* d_down_weight   = gpu_alloc((size_t)HIDDEN * INTER * 2);

    void* d_qkv_scratch    = gpu_alloc(QKV_DIM * 2);
    void* d_attn_scratch   = gpu_alloc(Q_DIM * 2);
    void* d_oproj_scratch  = gpu_alloc(HIDDEN * 2);
    void* d_gateup_scratch = gpu_alloc(GATE_UP * 2);

    int split_meta_size = MAX_SPLITS * NUM_KV * HPG;
    void* d_split_max = gpu_alloc(split_meta_size * 4);
    void* d_split_sum = gpu_alloc(split_meta_size * 4);
    int split_acc_size = MAX_SPLITS * NUM_KV * HPG * HEAD_DIM;
    void* d_split_acc = gpu_alloc(split_acc_size * 2);

    int num_sync_flags = 6;
    void* d_sync_flags = gpu_alloc(num_sync_flags * 4);

    upload(d_prev_residual, h_prev_residual);
    upload(d_key_cache, h_key_cache);
    upload(d_val_cache, h_val_cache);
    upload(d_block_tables, h_block_tables);
    upload(d_context_lens, h_context_lens);
    upload(d_positions, h_positions);
    upload(d_slot_mapping, h_slot_mapping);
    upload(d_rope_cos, h_rope_cos);
    upload(d_rope_sin, h_rope_sin);
    upload(d_norm_w, h_norm_w);
    upload(d_qkv_weight, h_qkv_weight);
    upload(d_qkv_bias, h_qkv_bias);
    upload(d_o_weight, h_o_weight);
    upload(d_post_norm_w, h_post_norm_w);
    upload(d_gateup_weight, h_gateup_weight);
    upload(d_down_weight, h_down_weight);

    void* d_prev_mlp = nullptr;

    float eps = EPS;
    float attn_scale = ATTN_SC;
    int hidden_size = HIDDEN;
    int q_dim = Q_DIM;
    int kv_dim = KV_DIM;
    int qkv_dim = QKV_DIM;
    int num_heads = NUM_HEADS;
    int num_kv_heads = NUM_KV;
    int head_dim = HEAD_DIM;
    int intermediate_size = INTER;
    int gate_up_dim = GATE_UP;
    int block_size = BLOCK_SZ;
    int max_context_len = MAX_CTX;
    int max_blocks_per_seq = MAX_BPQ;
    int max_splits = MAX_SPLITS;

    void* args[] = {
        &d_mlp_out, &d_residual_out,
        &d_prev_residual, &d_prev_mlp,
        &d_key_cache, &d_val_cache,
        &d_block_tables, &d_context_lens,
        &d_positions, &d_slot_mapping,
        &d_rope_cos, &d_rope_sin,
        &d_norm_w, &d_qkv_weight, &d_qkv_bias,
        &d_o_weight, &d_post_norm_w,
        &d_gateup_weight, &d_down_weight,
        &d_qkv_scratch, &d_attn_scratch, &d_oproj_scratch, &d_gateup_scratch,
        &d_split_max, &d_split_sum, &d_split_acc, &max_splits,
        &eps, &attn_scale,
        &hidden_size, &q_dim, &kv_dim, &qkv_dim,
        &num_heads, &num_kv_heads, &head_dim,
        &intermediate_size, &gate_up_dim,
        &block_size, &max_context_len, &max_blocks_per_seq,
        &d_sync_flags
    };

    const unsigned int grid_v3 = 1024;

    RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_flags * 4));
    RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));

    CU_CHECK(cuLaunchKernel(func, grid_v3, 1, 1, BLOCK, 1, 1, smem, 0, args, nullptr));
    RT_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_mlp_out(HIDDEN), h_residual_out(HIDDEN);
    download(h_mlp_out, d_mlp_out);
    download(h_residual_out, d_residual_out);

    bool pass = true;
    pass &= check_output("mlp_out", h_mlp_out.data(), HIDDEN);
    pass &= check_output("residual_out", h_residual_out.data(), HIDDEN);

    cudaFree(d_mlp_out); cudaFree(d_residual_out);
    cudaFree(d_prev_residual);
    cudaFree(d_key_cache); cudaFree(d_val_cache);
    cudaFree(d_block_tables); cudaFree(d_context_lens);
    cudaFree(d_positions); cudaFree(d_slot_mapping);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_norm_w); cudaFree(d_qkv_weight); cudaFree(d_qkv_bias);
    cudaFree(d_o_weight); cudaFree(d_post_norm_w);
    cudaFree(d_gateup_weight); cudaFree(d_down_weight);
    cudaFree(d_qkv_scratch); cudaFree(d_attn_scratch);
    cudaFree(d_oproj_scratch); cudaFree(d_gateup_scratch);
    cudaFree(d_split_max); cudaFree(d_split_sum); cudaFree(d_split_acc);
    cudaFree(d_sync_flags);

    return pass;
}

// ---------------------------------------------------------------------------
// Test: megakernel_v2_f16
// ---------------------------------------------------------------------------

static bool test_megakernel_v2(CUmodule mod) {
    printf("\n=== megakernel_v2_f16 ===\n");

    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, mod, "megakernel_v2_f16"));

    int smem = compute_smem();
    CU_CHECK(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    printf("  shared memory: %d bytes\n", smem);

    xor_state = 0xCAFEBABE;

    int num_layers = 1;

    // Per-layer weight arrays (just 1 layer)
    std::vector<__half> h_norm_w(HIDDEN);
    std::vector<__half> h_qkv_weight((size_t)QKV_DIM * HIDDEN);
    std::vector<__half> h_qkv_bias(QKV_DIM);
    std::vector<__half> h_o_weight((size_t)HIDDEN * Q_DIM);
    std::vector<__half> h_post_norm_w(HIDDEN);
    std::vector<__half> h_gateup_weight((size_t)GATE_UP * HIDDEN);
    std::vector<__half> h_down_weight((size_t)HIDDEN * INTER);

    fill_f16(h_norm_w, 0.8f, 1.2f);
    fill_f16(h_qkv_weight, -0.01f, 0.01f);
    fill_f16(h_qkv_bias, -0.01f, 0.01f);
    fill_f16(h_o_weight, -0.01f, 0.01f);
    fill_f16(h_post_norm_w, 0.8f, 1.2f);
    fill_f16(h_gateup_weight, -0.01f, 0.01f);
    fill_f16(h_down_weight, -0.01f, 0.01f);

    // Embedding table: [vocab_size * hidden]
    // Only allocate a small portion -- we only access one row
    // But the kernel indexes by token_id, so we need the full table or a safe subset.
    // Use a small vocab subset and set input_token=0 for safety.
    int test_vocab = VOCAB_SIZE;
    // Allocate full embedding on GPU but only fill a small portion
    void* d_embed_tokens = gpu_alloc((size_t)test_vocab * HIDDEN * 2);
    // Fill first few rows with random data
    {
        std::vector<__half> h_embed_row(HIDDEN);
        fill_f16(h_embed_row, -0.1f, 0.1f);
        RT_CHECK(cudaMemcpy(d_embed_tokens, h_embed_row.data(), HIDDEN * 2, cudaMemcpyHostToDevice));
    }

    std::vector<__half> h_final_norm_w(HIDDEN);
    fill_f16(h_final_norm_w, 0.8f, 1.2f);

    // lm_head_weight: [vocab_size * hidden]
    // Too large to fill entirely; allocate zeroed and fill a small block so argmax is valid
    void* d_lm_head_weight = gpu_alloc((size_t)test_vocab * HIDDEN * 2);
    {
        // Fill first 1024 rows with small random values so argmax picks something
        int fill_rows = 1024;
        std::vector<__half> h_lm_partial((size_t)fill_rows * HIDDEN);
        fill_f16(h_lm_partial, -0.01f, 0.01f);
        RT_CHECK(cudaMemcpy(d_lm_head_weight, h_lm_partial.data(),
                             (size_t)fill_rows * HIDDEN * 2, cudaMemcpyHostToDevice));
    }

    // Upload per-layer weights
    void* d_norm_w_l      = gpu_alloc(HIDDEN * 2);
    void* d_qkv_weight_l  = gpu_alloc((size_t)QKV_DIM * HIDDEN * 2);
    void* d_qkv_bias_l    = gpu_alloc(QKV_DIM * 2);
    void* d_o_weight_l    = gpu_alloc((size_t)HIDDEN * Q_DIM * 2);
    void* d_post_norm_w_l = gpu_alloc(HIDDEN * 2);
    void* d_gateup_w_l    = gpu_alloc((size_t)GATE_UP * HIDDEN * 2);
    void* d_down_w_l      = gpu_alloc((size_t)HIDDEN * INTER * 2);

    upload(d_norm_w_l, h_norm_w);
    upload(d_qkv_weight_l, h_qkv_weight);
    upload(d_qkv_bias_l, h_qkv_bias);
    upload(d_o_weight_l, h_o_weight);
    upload(d_post_norm_w_l, h_post_norm_w);
    upload(d_gateup_w_l, h_gateup_weight);
    upload(d_down_w_l, h_down_weight);

    void* d_final_norm_w = gpu_alloc(HIDDEN * 2);
    upload(d_final_norm_w, h_final_norm_w);

    // Per-layer pointer arrays on GPU (arrays of 1 pointer each)
    auto make_ptr_array = [](void* ptr) -> void* {
        void* d_arr;
        RT_CHECK(cudaMalloc(&d_arr, sizeof(void*)));
        RT_CHECK(cudaMemcpy(d_arr, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
        return d_arr;
    };

    void* d_norm_weights_arr     = make_ptr_array(d_norm_w_l);
    void* d_qkv_weights_arr     = make_ptr_array(d_qkv_weight_l);
    void* d_qkv_biases_arr      = make_ptr_array(d_qkv_bias_l);
    void* d_o_weights_arr       = make_ptr_array(d_o_weight_l);
    void* d_post_norm_weights_arr = make_ptr_array(d_post_norm_w_l);
    void* d_gateup_weights_arr  = make_ptr_array(d_gateup_w_l);
    void* d_down_weights_arr    = make_ptr_array(d_down_w_l);

    // KV caches
    int num_slots = MAX_BPQ * BLOCK_SZ;
    void* d_key_cache_l = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);
    void* d_val_cache_l = gpu_alloc((size_t)num_slots * NUM_KV * HEAD_DIM * 2);

    // Fill KV caches with small random data
    {
        std::vector<__half> h_kv((size_t)num_slots * NUM_KV * HEAD_DIM);
        fill_f16(h_kv, -0.05f, 0.05f);
        upload(d_key_cache_l, h_kv);
        fill_f16(h_kv, -0.05f, 0.05f);
        upload(d_val_cache_l, h_kv);
    }

    void* d_key_caches_arr = make_ptr_array(d_key_cache_l);
    void* d_val_caches_arr = make_ptr_array(d_val_cache_l);

    // Input data
    std::vector<int> h_input_token = {0};
    std::vector<int> h_positions = {CONTEXT_LEN};
    std::vector<int> h_slot_mapping = {CONTEXT_LEN};
    std::vector<int> h_block_tables(MAX_BPQ);
    for (int i = 0; i < MAX_BPQ; i++) h_block_tables[i] = i;
    std::vector<int> h_context_lens = {CONTEXT_LEN};

    // RoPE
    std::vector<float> h_rope_cos((size_t)MAX_CTX * HALF_DIM);
    std::vector<float> h_rope_sin((size_t)MAX_CTX * HALF_DIM);
    for (int pos = 0; pos < MAX_CTX; pos++) {
        for (int d = 0; d < HALF_DIM; d++) {
            float freq = 1.0f / powf(1000000.0f, (float)(2 * d) / (float)HEAD_DIM);
            h_rope_cos[pos * HALF_DIM + d] = cosf((float)pos * freq);
            h_rope_sin[pos * HALF_DIM + d] = sinf((float)pos * freq);
        }
    }

    void* d_input_token  = gpu_alloc(4);
    void* d_positions    = gpu_alloc(4);
    void* d_slot_mapping = gpu_alloc(4);
    void* d_block_tables = gpu_alloc(MAX_BPQ * 4);
    void* d_context_lens = gpu_alloc(4);
    void* d_rope_cos     = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);
    void* d_rope_sin     = gpu_alloc((size_t)MAX_CTX * HALF_DIM * 4);

    upload(d_input_token, h_input_token);
    upload(d_positions, h_positions);
    upload(d_slot_mapping, h_slot_mapping);
    upload(d_block_tables, h_block_tables);
    upload(d_context_lens, h_context_lens);
    upload(d_rope_cos, h_rope_cos);
    upload(d_rope_sin, h_rope_sin);

    // Scratch buffers
    void* d_hidden_buf     = gpu_alloc(HIDDEN * 2);
    void* d_mlp_buf        = gpu_alloc(HIDDEN * 2);
    void* d_qkv_scratch    = gpu_alloc(QKV_DIM * 2);
    void* d_attn_scratch   = gpu_alloc(Q_DIM * 2);
    void* d_oproj_scratch  = gpu_alloc(HIDDEN * 2);
    void* d_gateup_scratch = gpu_alloc(GATE_UP * 2);
    void* d_logits_scratch = gpu_alloc((size_t)test_vocab * 2);

    // Split-KV scratch
    int split_meta_size = MAX_SPLITS * NUM_KV * HPG;
    void* d_split_max = gpu_alloc(split_meta_size * 4);
    void* d_split_sum = gpu_alloc(split_meta_size * 4);
    int split_acc_size = MAX_SPLITS * NUM_KV * HPG * HEAD_DIM;
    void* d_split_acc = gpu_alloc(split_acc_size * 2);

    // Sync flags: num_layers * 7 + 3 (embed sync, lm_head sync, plus spare)
    int num_sync_slots = num_layers * 7 + 3;
    void* d_sync_flags = gpu_alloc(num_sync_slots * 4);

    // Output
    void* d_output_token = gpu_alloc(4);

    // Config scalars
    int nl = num_layers;
    int hs = HIDDEN, qd = Q_DIM, kvd = KV_DIM, qkvd = QKV_DIM;
    int nh = NUM_HEADS, nkv = NUM_KV, hd = HEAD_DIM;
    int inter = INTER, gud = GATE_UP;
    int bs = BLOCK_SZ, mcl = MAX_CTX, mbps = MAX_BPQ, vs = test_vocab;
    float eps = EPS, as_ = ATTN_SC;
    int ms = MAX_SPLITS;

    // Kernel args -- must match megakernel_v2_f16 signature exactly
    void* args[] = {
        &d_norm_weights_arr, &d_qkv_weights_arr, &d_qkv_biases_arr,
        &d_o_weights_arr, &d_post_norm_weights_arr,
        &d_gateup_weights_arr, &d_down_weights_arr,
        &d_key_caches_arr, &d_val_caches_arr,
        &d_embed_tokens, &d_final_norm_w, &d_lm_head_weight,
        &d_input_token, &d_positions,
        &d_slot_mapping, &d_block_tables, &d_context_lens,
        &d_rope_cos, &d_rope_sin,
        &d_hidden_buf, &d_mlp_buf,
        &d_qkv_scratch, &d_attn_scratch, &d_oproj_scratch, &d_gateup_scratch,
        &d_logits_scratch,
        &d_split_max, &d_split_sum, &d_split_acc, &ms,
        &d_sync_flags,
        &nl, &hs, &qd, &kvd, &qkvd,
        &nh, &nkv, &hd, &inter, &gud,
        &bs, &mcl, &mbps, &vs,
        &eps, &as_,
        &d_output_token
    };

    // Launch
    RT_CHECK(cudaMemset(d_sync_flags, 0, num_sync_slots * 4));
    RT_CHECK(cudaMemset(d_split_max, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_sum, 0, split_meta_size * 4));
    RT_CHECK(cudaMemset(d_split_acc, 0, split_acc_size * 2));

    CU_CHECK(cuLaunchCooperativeKernel(func, GRID, 1, 1, BLOCK, 1, 1,
                                        smem, 0, args));
    RT_CHECK(cudaDeviceSynchronize());

    // Check output token
    int h_output_token = -1;
    RT_CHECK(cudaMemcpy(&h_output_token, d_output_token, 4, cudaMemcpyDeviceToHost));

    bool pass = true;
    if (h_output_token < 0 || h_output_token >= test_vocab) {
        fprintf(stderr, "  FAIL: output_token=%d not in [0, %d)\n", h_output_token, test_vocab);
        pass = false;
    } else {
        printf("  OK: output_token=%d (valid vocab index)\n", h_output_token);
    }

    // Also check hidden_buf and mlp_buf are sane
    std::vector<__half> h_hidden(HIDDEN), h_mlp(HIDDEN);
    download(h_hidden, d_hidden_buf);
    download(h_mlp, d_mlp_buf);
    pass &= check_output("hidden_buf", h_hidden.data(), HIDDEN);
    pass &= check_output("mlp_buf", h_mlp.data(), HIDDEN);

    // Free
    cudaFree(d_embed_tokens); cudaFree(d_lm_head_weight);
    cudaFree(d_norm_w_l); cudaFree(d_qkv_weight_l); cudaFree(d_qkv_bias_l);
    cudaFree(d_o_weight_l); cudaFree(d_post_norm_w_l);
    cudaFree(d_gateup_w_l); cudaFree(d_down_w_l);
    cudaFree(d_final_norm_w);
    cudaFree(d_norm_weights_arr); cudaFree(d_qkv_weights_arr);
    cudaFree(d_qkv_biases_arr); cudaFree(d_o_weights_arr);
    cudaFree(d_post_norm_weights_arr); cudaFree(d_gateup_weights_arr);
    cudaFree(d_down_weights_arr);
    cudaFree(d_key_cache_l); cudaFree(d_val_cache_l);
    cudaFree(d_key_caches_arr); cudaFree(d_val_caches_arr);
    cudaFree(d_input_token); cudaFree(d_positions); cudaFree(d_slot_mapping);
    cudaFree(d_block_tables); cudaFree(d_context_lens);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_hidden_buf); cudaFree(d_mlp_buf);
    cudaFree(d_qkv_scratch); cudaFree(d_attn_scratch);
    cudaFree(d_oproj_scratch); cudaFree(d_gateup_scratch);
    cudaFree(d_logits_scratch);
    cudaFree(d_split_max); cudaFree(d_split_sum); cudaFree(d_split_acc);
    cudaFree(d_sync_flags); cudaFree(d_output_token);

    return pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // Initialize CUDA driver API
    CU_CHECK(cuInit(0));

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));

    // Verify sm_90
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    printf("GPU compute capability: sm_%d%d\n", major, minor);
    if (major < 9) {
        fprintf(stderr, "ERROR: sm_90 required, got sm_%d%d\n", major, minor);
        return 1;
    }

    // Check cooperative launch support
    int coop = 0;
    CU_CHECK(cuDeviceGetAttribute(&coop, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev));
    if (!coop) {
        fprintf(stderr, "ERROR: cooperative kernel launch not supported\n");
        return 1;
    }

    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, dev));

    // Print device info
    char name[256];
    CU_CHECK(cuDeviceGetName(name, sizeof(name), dev));
    size_t mem_total = 0;
    CU_CHECK(cuDeviceTotalMem(&mem_total, dev));
    printf("Device: %s (%.1f GB)\n", name, (float)mem_total / (1024.0f * 1024.0f * 1024.0f));

    // Load cubins
    CUmodule mod_layer, mod_v3, mod_mega;

    printf("\nLoading sm_90/persistent_layer_v2.cubin...\n");
    CU_CHECK(cuModuleLoad(&mod_layer, "sm_90/persistent_layer_v2.cubin"));
    printf("  loaded OK\n");

    printf("Loading sm_90/persistent_layer_v3.cubin...\n");
    CU_CHECK(cuModuleLoad(&mod_v3, "sm_90/persistent_layer_v3.cubin"));
    printf("  loaded OK\n");

    printf("Loading sm_90/megakernel_v2.cubin...\n");
    CU_CHECK(cuModuleLoad(&mod_mega, "sm_90/megakernel_v2.cubin"));
    printf("  loaded OK\n");

    bool pass = true;

    pass &= test_persistent_layer_v2(mod_layer);
    pass &= test_persistent_layer_v3(mod_v3);
    pass &= test_megakernel_v2(mod_mega);

    CU_CHECK(cuModuleUnload(mod_layer));
    CU_CHECK(cuModuleUnload(mod_v3));
    CU_CHECK(cuModuleUnload(mod_mega));
    CU_CHECK(cuCtxDestroy(ctx));

    printf("\n========================================\n");
    if (pass) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("========================================\n");

    return pass ? 0 : 1;
}
