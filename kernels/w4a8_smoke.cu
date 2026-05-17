// Smoke test for libw4a8_gemm.so.
//
// Builds a tiny exact case:
//   A_fp8 = all 1.0, alpha = 1.0
//   W_f16 = signed integers in [-7,7] with max(|W|)=7 per group
// The W4 encoder should be lossless, so D[m,n] must equal sum_k W[n,k].

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

extern "C" size_t rvllm_w4a8_gemm_workspace_size(int m, int n, int k);
extern "C" size_t rvllm_w4a8_int4_reordered_bytes(int n, int k);

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int GROUP = 128;
    static_assert(K % GROUP == 0);

    std::vector<unsigned char> h_a(M * K, 0x38); // E4M3 1.0
    std::vector<__half> h_w(N * K);
    std::vector<float> ref(N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int v = ((n * 17 + k * 5) % 15) - 7;
            if (k == 0) v = 7;
            h_w[n * K + k] = __float2half(static_cast<float>(v));
            ref[n] += static_cast<float>(v);
        }
    }

    void *d_a = nullptr, *d_w = nullptr, *d_int4 = nullptr, *d_scales = nullptr;
    void *d_scale_ws = nullptr, *d_c = nullptr, *d_out = nullptr, *d_ws = nullptr;
    size_t int4_bytes = rvllm_w4a8_int4_reordered_bytes(N, K);
    check(cudaMalloc(&d_a, h_a.size()), "cudaMalloc A");
    check(cudaMalloc(&d_w, h_w.size() * sizeof(__half)), "cudaMalloc W");
    check(cudaMalloc(&d_int4, int4_bytes), "cudaMalloc int4");
    check(cudaMalloc(&d_scales, N * (K / GROUP) * 8), "cudaMalloc scales");
    check(cudaMalloc(&d_scale_ws, N * (K / GROUP) * sizeof(float)), "cudaMalloc scale_ws");
    check(cudaMalloc(&d_c, M * N * sizeof(__half)), "cudaMalloc C");
    check(cudaMalloc(&d_out, M * N * sizeof(__half)), "cudaMalloc out");
    size_t ws_bytes = rvllm_w4a8_gemm_workspace_size(M, N, K);
    check(cudaMalloc(&d_ws, ws_bytes), "cudaMalloc ws");

    check(cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy W");
    check(cudaMemset(d_c, 0, M * N * sizeof(__half)), "zero C");

    int rc = rvllm_w4a8_encode_weight_fp16(
        d_w, N, K, GROUP, d_int4, d_scales, d_scale_ws, 0, 0);
    check(cudaDeviceSynchronize(), "sync encode");
    if (rc != 0) {
        std::fprintf(stderr, "encode rc=%d\n", rc);
        return 2;
    }
    rc = rvllm_w4a8_gemm_run(
        d_a, d_int4, d_scales, d_c, d_out, M, N, K, GROUP,
        1.0f, 0.0f, d_ws, ws_bytes, 0);
    check(cudaDeviceSynchronize(), "sync gemm");
    if (rc != 0) {
        std::fprintf(stderr, "gemm rc=%d\n", rc);
        return 3;
    }

    std::vector<__half> h_out(M * N);
    check(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy out");

    float max_abs = 0.0f;
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float got = __half2float(h_out[m * N + n]);
            float err = std::fabs(got - ref[n]);
            max_abs = std::fmax(max_abs, err);
            if (err > 1.0f && bad < 8) {
                std::fprintf(stderr, "m=%d n=%d got=%f ref=%f err=%f\n", m, n, got, ref[n], err);
                ++bad;
            }
        }
    }
    std::printf("w4a8_smoke max_abs=%f workspace=%zu\n", max_abs, ws_bytes);
    return max_abs <= 1.0f ? 0 : 4;
}
