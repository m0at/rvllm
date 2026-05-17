// Smoke and contract tests for libw4a8_gemm.so.
//
// These cases are intentionally tiny but targeted:
// - all-ones reduction catches basic encode/GEMM breakage
// - one-hot projections catch B row/column layout mistakes
// - calibrated signed sentinels catch INT4 nibble/sign handling
// - calibrated unique scales catch scale stride/order mistakes
// - rowscale polarity catches activation scale direction

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" int rvllm_w4a8_encode_weight_fp16(
    const void* w_fp16,
    int n,
    int k,
    int group_size,
    void* w_int4_out,
    void* scales_packed_out,
    void* scales_f32_workspace,
    int shuffle,
    cudaStream_t stream);

extern "C" int rvllm_w4a8_encode_weight_fp16_with_scales(
    const void* w_fp16,
    const float* calibrated_scales_f32,
    int n,
    int k,
    int group_size,
    void* w_int4_out,
    void* scales_packed_out,
    void* scales_f32_workspace,
    int shuffle,
    cudaStream_t stream);

extern "C" int rvllm_w4a8_gemm_run(
    const void* a_fp8,
    const void* b_int4_reordered,
    const void* b_scales_packed,
    const void* c_f16,
    void* d_f16,
    int m,
    int n,
    int k,
    int group_size,
    float alpha,
    float beta,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" int rvllm_w4a8_gemm_run_rowscale(
    const void* a_fp8,
    const float* a_scales,
    const void* b_int4_reordered,
    const void* b_scales_packed,
    void* d_f16,
    int m,
    int n,
    int k,
    int group_size,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" size_t rvllm_w4a8_gemm_workspace_size(int m, int n, int k);
extern "C" size_t rvllm_w4a8_int4_reordered_bytes(int n, int k);

static constexpr int GROUP = 128;
static constexpr unsigned char FP8_ZERO = 0x00;
static constexpr unsigned char FP8_ONE = 0x38;

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void cuda_free(void* p) {
    if (p != nullptr) cudaFree(p);
}

static int run_case(
    const char* name,
    int M,
    int N,
    int K,
    const std::vector<unsigned char>& h_a,
    const std::vector<__half>& h_w,
    const std::vector<float>* h_calibrated_scales,
    const std::vector<float>* h_row_scales,
    const std::vector<float>& ref,
    float tolerance,
    float* max_abs_out,
    size_t* workspace_out) {
    if ((int)h_a.size() != M * K || (int)h_w.size() != N * K || (int)ref.size() != M * N) {
        std::fprintf(stderr, "%s invalid host buffer sizes\n", name);
        return 10;
    }
    if (h_calibrated_scales && (int)h_calibrated_scales->size() != N * (K / GROUP)) {
        std::fprintf(stderr, "%s invalid calibrated scale size\n", name);
        return 11;
    }
    if (h_row_scales && (int)h_row_scales->size() != M) {
        std::fprintf(stderr, "%s invalid row scale size\n", name);
        return 12;
    }

    void *d_a = nullptr, *d_w = nullptr, *d_int4 = nullptr, *d_scales = nullptr;
    void *d_scale_ws = nullptr, *d_c = nullptr, *d_out = nullptr, *d_ws = nullptr;
    void *d_calibrated_scales = nullptr, *d_row_scales = nullptr;

    size_t int4_bytes = rvllm_w4a8_int4_reordered_bytes(N, K);
    size_t scale_bytes = (size_t)N * (K / GROUP) * 8;
    size_t scale_ws_bytes = (size_t)N * (K / GROUP) * sizeof(float);
    size_t ws_bytes = rvllm_w4a8_gemm_workspace_size(M, N, K);
    if (int4_bytes == 0) {
        std::fprintf(stderr, "%s invalid int4 byte size\n", name);
        return 13;
    }

    check(cudaMalloc(&d_a, h_a.size()), "cudaMalloc A");
    check(cudaMalloc(&d_w, h_w.size() * sizeof(__half)), "cudaMalloc W");
    check(cudaMalloc(&d_int4, int4_bytes), "cudaMalloc int4");
    check(cudaMalloc(&d_scales, scale_bytes), "cudaMalloc scales");
    check(cudaMalloc(&d_scale_ws, scale_ws_bytes), "cudaMalloc scale_ws");
    check(cudaMalloc(&d_c, (size_t)M * N * sizeof(__half)), "cudaMalloc C");
    check(cudaMalloc(&d_out, (size_t)M * N * sizeof(__half)), "cudaMalloc out");
    if (ws_bytes != 0) check(cudaMalloc(&d_ws, ws_bytes), "cudaMalloc ws");
    if (h_calibrated_scales) {
        check(cudaMalloc(&d_calibrated_scales, h_calibrated_scales->size() * sizeof(float)), "cudaMalloc calibrated scales");
    }
    if (h_row_scales) {
        check(cudaMalloc(&d_row_scales, h_row_scales->size() * sizeof(float)), "cudaMalloc row scales");
    }

    check(cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy W");
    check(cudaMemset(d_c, 0, (size_t)M * N * sizeof(__half)), "zero C");
    if (h_calibrated_scales) {
        check(cudaMemcpy(
                  d_calibrated_scales,
                  h_calibrated_scales->data(),
                  h_calibrated_scales->size() * sizeof(float),
                  cudaMemcpyHostToDevice),
              "copy calibrated scales");
    }
    if (h_row_scales) {
        check(cudaMemcpy(
                  d_row_scales,
                  h_row_scales->data(),
                  h_row_scales->size() * sizeof(float),
                  cudaMemcpyHostToDevice),
              "copy row scales");
    }

    int rc = 0;
    if (h_calibrated_scales) {
        rc = rvllm_w4a8_encode_weight_fp16_with_scales(
            d_w, (const float*)d_calibrated_scales, N, K, GROUP, d_int4, d_scales, d_scale_ws, 1, 0);
    } else {
        rc = rvllm_w4a8_encode_weight_fp16(
            d_w, N, K, GROUP, d_int4, d_scales, d_scale_ws, 1, 0);
    }
    check(cudaDeviceSynchronize(), "sync encode");
    if (rc != 0) {
        std::fprintf(stderr, "%s encode rc=%d\n", name, rc);
        return 20;
    }

    if (h_row_scales) {
        rc = rvllm_w4a8_gemm_run_rowscale(
            d_a, (const float*)d_row_scales, d_int4, d_scales, d_out, M, N, K, GROUP, d_ws, ws_bytes, 0);
    } else {
        rc = rvllm_w4a8_gemm_run(
            d_a, d_int4, d_scales, d_c, d_out, M, N, K, GROUP, 1.0f, 0.0f, d_ws, ws_bytes, 0);
    }
    check(cudaDeviceSynchronize(), "sync gemm");
    if (rc != 0) {
        std::fprintf(stderr, "%s gemm rc=%d\n", name, rc);
        return 21;
    }

    std::vector<__half> h_out((size_t)M * N);
    check(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy out");

    float max_abs = 0.0f;
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float got = __half2float(h_out[(size_t)m * N + n]);
            float err = std::fabs(got - ref[(size_t)m * N + n]);
            max_abs = std::fmax(max_abs, err);
            if ((!std::isfinite(got) || err > tolerance) && bad < 8) {
                std::fprintf(
                    stderr,
                    "%s m=%d n=%d got=%f ref=%f err=%f tolerance=%f\n",
                    name,
                    m,
                    n,
                    got,
                    ref[(size_t)m * N + n],
                    err,
                    tolerance);
                ++bad;
            }
        }
    }

    cuda_free(d_a);
    cuda_free(d_w);
    cuda_free(d_int4);
    cuda_free(d_scales);
    cuda_free(d_scale_ws);
    cuda_free(d_c);
    cuda_free(d_out);
    cuda_free(d_ws);
    cuda_free(d_calibrated_scales);
    cuda_free(d_row_scales);

    *max_abs_out = max_abs;
    *workspace_out = ws_bytes;
    std::printf("w4a8_smoke case=%s max_abs=%f workspace=%zu\n", name, max_abs, ws_bytes);
    return max_abs <= tolerance ? 0 : 30;
}

static int signed_value(int x) {
    static constexpr int vals[16] = {-8, -7, -5, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, -2, -4, -6};
    return vals[x & 15];
}

static int all_ones_reduction(float* max_abs, size_t* workspace) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    std::vector<unsigned char> a(M * K, FP8_ONE);
    std::vector<__half> w(N * K);
    std::vector<float> ref(M * N, 0.0f);
    std::vector<float> col_ref(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int v = ((n * 17 + k * 5) % 15) - 7;
            if (k == 0) v = 7;
            w[n * K + k] = __float2half((float)v);
            col_ref[n] += (float)v;
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) ref[m * N + n] = col_ref[n];
    }
    return run_case("all_ones_reduction", M, N, K, a, w, nullptr, nullptr, ref, 1.0f, max_abs, workspace);
}

static int one_hot_layout(float* max_abs, size_t* workspace) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    std::vector<unsigned char> a(M * K, FP8_ZERO);
    std::vector<__half> w(N * K);
    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) a[m * K + m] = FP8_ONE;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int v = ((n * 3 + k * 5) % 15) - 7;
            if (k == 0) v = 7;
            w[n * K + k] = __float2half((float)v);
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) ref[m * N + n] = __half2float(w[n * K + m]);
    }
    return run_case("one_hot_layout", M, N, K, a, w, nullptr, nullptr, ref, 1.0f, max_abs, workspace);
}

static int calibrated_signed_sentinel(float* max_abs, size_t* workspace) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    std::vector<unsigned char> a(M * K, FP8_ONE);
    std::vector<__half> w(N * K);
    std::vector<float> scales(N * (K / GROUP), 1.0f);
    std::vector<float> ref(M * N, 0.0f);
    std::vector<float> col_ref(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int v = signed_value(n + k);
            w[n * K + k] = __float2half((float)v);
            col_ref[n] += (float)v;
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) ref[m * N + n] = col_ref[n];
    }
    return run_case("calibrated_signed_sentinel", M, N, K, a, w, &scales, nullptr, ref, 1.0f, max_abs, workspace);
}

static int calibrated_scale_stride(float* max_abs, size_t* workspace) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 256;
    constexpr int SCALE_K = K / GROUP;
    std::vector<unsigned char> a(M * K, FP8_ZERO);
    std::vector<__half> w(N * K);
    std::vector<float> scales(N * SCALE_K);
    std::vector<float> ref(M * N, 0.0f);

    for (int m = 0; m < M; ++m) {
        int g = m & 1;
        a[m * K + g * GROUP] = FP8_ONE;
    }
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < SCALE_K; ++g) {
            float s = 0.5f;
            if (n >= 64 && g == 0) s = 2.0f;
            if (n < 64 && g == 1) s = 1.0f;
            if (n >= 64 && g == 1) s = 4.0f;
            scales[n * SCALE_K + g] = s;
            for (int k = g * GROUP; k < (g + 1) * GROUP; ++k) {
                w[n * K + k] = __float2half(s);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        int g = m & 1;
        for (int n = 0; n < N; ++n) ref[m * N + n] = scales[n * SCALE_K + g];
    }
    return run_case("calibrated_scale_stride", M, N, K, a, w, &scales, nullptr, ref, 0.25f, max_abs, workspace);
}

static int rowscale_polarity(float* max_abs, size_t* workspace) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    std::vector<unsigned char> a(M * K, FP8_ZERO);
    std::vector<__half> w(N * K, __float2half(0.0f));
    std::vector<float> row_scales(M);
    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        a[m * K] = FP8_ONE;
        row_scales[m] = m < 64 ? 0.5f : 2.0f;
    }
    for (int n = 0; n < N; ++n) w[n * K] = __float2half(1.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) ref[m * N + n] = row_scales[m];
    }
    return run_case("rowscale_polarity", M, N, K, a, w, nullptr, &row_scales, ref, 0.125f, max_abs, workspace);
}

int main() {
    struct Test {
        const char* name;
        int (*fn)(float*, size_t*);
    };
    Test tests[] = {
        {"all_ones_reduction", all_ones_reduction},
        {"one_hot_layout", one_hot_layout},
        {"calibrated_signed_sentinel", calibrated_signed_sentinel},
        {"calibrated_scale_stride", calibrated_scale_stride},
        {"rowscale_polarity", rowscale_polarity},
    };

    float global_max = 0.0f;
    size_t last_workspace = 0;
    for (const auto& test : tests) {
        float max_abs = 0.0f;
        size_t workspace = 0;
        int rc = test.fn(&max_abs, &workspace);
        global_max = std::fmax(global_max, max_abs);
        last_workspace = workspace;
        if (rc != 0) {
            std::fprintf(stderr, "w4a8_smoke failed case=%s rc=%d\n", test.name, rc);
            return rc;
        }
    }
    std::printf("w4a8_smoke max_abs=%f workspace=%zu cases=%zu\n", global_max, last_workspace, sizeof(tests) / sizeof(tests[0]));
    return 0;
}
