// CUTLASS stub .so for Ampere (SM86) deployments.
//
// Exports all variant symbols that CutlassLib::load() resolves at init.
// Functions are never called (cuBLASLt handles all FP8 GEMMs on Ampere),
// but must exist for dlsym to succeed during bring-up.
//
// Build:
//   gcc -shared -fPIC -o libcutlass_stub_sm86.so cutlass_stub_sm86.c
//
// Variants 0..15:  cutlass_fp8_gemm_v{N} + workspace_size
// Variants 100..109: cutlass_fp8_gemm_residual_v{N} + workspace_size

#include <stddef.h>

// Non-residual GEMM stub: returns 0 (success), never called.
#define GEMM_STUB(N) \
    int cutlass_fp8_gemm_v##N( \
        void* out, const void* a, const void* b, \
        const void* a_scales, const void* b_scale, \
        int m, int n, int k, void* ws, size_t ws_sz, void* stream) { \
        (void)out;(void)a;(void)b;(void)a_scales;(void)b_scale; \
        (void)m;(void)n;(void)k;(void)ws;(void)ws_sz;(void)stream; \
        return 0; \
    } \
    size_t cutlass_fp8_gemm_v##N##_workspace_size(int m, int n, int k) { \
        (void)m;(void)n;(void)k; return 0; \
    }

// Residual GEMM stub.
#define GEMM_RES_STUB(N) \
    int cutlass_fp8_gemm_residual_v##N( \
        void* out, const void* a, const void* b, \
        const void* a_scales, const void* b_scale, \
        const void* residual, \
        int m, int n, int k, void* ws, size_t ws_sz, void* stream) { \
        (void)out;(void)a;(void)b;(void)a_scales;(void)b_scale; \
        (void)residual;(void)m;(void)n;(void)k;(void)ws;(void)ws_sz;(void)stream; \
        return 0; \
    } \
    size_t cutlass_fp8_gemm_residual_v##N##_workspace_size(int m, int n, int k) { \
        (void)m;(void)n;(void)k; return 0; \
    }

GEMM_STUB(0)  GEMM_STUB(1)  GEMM_STUB(2)  GEMM_STUB(3)
GEMM_STUB(4)  GEMM_STUB(5)  GEMM_STUB(6)  GEMM_STUB(7)
GEMM_STUB(8)  GEMM_STUB(9)  GEMM_STUB(10) GEMM_STUB(11)
GEMM_STUB(12) GEMM_STUB(13) GEMM_STUB(14) GEMM_STUB(15)

GEMM_RES_STUB(0) GEMM_RES_STUB(1) GEMM_RES_STUB(2) GEMM_RES_STUB(3)
GEMM_RES_STUB(4) GEMM_RES_STUB(5) GEMM_RES_STUB(6) GEMM_RES_STUB(7)
GEMM_RES_STUB(8) GEMM_RES_STUB(9)
