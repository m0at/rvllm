// Cycle 38 probe: does cuGreenCtxCreate work on GB10 / sm_121?
//
// Cheapest possible test of the CUDA Driver Green Context API on this
// hardware. If the API returns CUDA_ERROR_NOT_SUPPORTED here, the rest
// of the example-95 work is moot. If it succeeds, we know we can bind
// streams to SM partitions and the engineering is feasible.
//
// Build: nvcc -O2 -std=c++17 -arch=sm_121a kernels/probe_green_ctx.cu \
//        -lcuda -o /tmp/probe_green_ctx
// Run:   /tmp/probe_green_ctx

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#define CK(call)                                                              \
  do {                                                                        \
    CUresult _r = (call);                                                     \
    if (_r != CUDA_SUCCESS) {                                                 \
      const char* msg = nullptr;                                              \
      cuGetErrorString(_r, &msg);                                             \
      std::fprintf(stderr, "%s:%d: %s = %d (%s)\n", __FILE__, __LINE__,       \
                   #call, (int)_r, msg ? msg : "?");                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

int main() {
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));

  char name[128];
  CK(cuDeviceGetName(name, sizeof(name), dev));
  int cc_major = 0, cc_minor = 0;
  CK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  int sm_count = 0;
  CK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  std::printf("device: %s sm_%d%d total_sms=%d\n", name, cc_major, cc_minor, sm_count);

  // Get the SM resource handle at the device level.
  CUdevResource sm_resource;
  CUresult r = cuDeviceGetDevResource(dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    std::printf("cuDeviceGetDevResource: %d (%s) — Green Context UNSUPPORTED on this device\n",
                (int)r, msg ? msg : "?");
    return 2;
  }
  std::printf("dev SM resource: type=%u sms.smCount=%u sms.smCoscheduledAlignment=%u\n",
              (unsigned)sm_resource.type, sm_resource.sm.smCount,
              sm_resource.sm.smCoscheduledAlignment);

  // Try to split: half primary, rest secondary. GB10 has 48 SMs total.
  unsigned target_primary = sm_resource.sm.smCount / 2;
  // Round to alignment.
  unsigned align = sm_resource.sm.smCoscheduledAlignment;
  if (align > 0) {
    target_primary = (target_primary / align) * align;
    if (target_primary == 0) target_primary = align;
  }
  unsigned remaining = sm_resource.sm.smCount - target_primary;
  std::printf("target split: primary=%u remaining=%u (alignment=%u)\n",
              target_primary, remaining, align);

  CUdevResource primary_split, remaining_split;
  unsigned int n_groups = 1;
  // cuDevSmResourceSplitByCount: splits an SM resource into N groups.
  // The simpler path: split into exactly two named partitions of sizes
  // [target_primary, remaining]. Use cuDevResourceGenerateDesc + Create.
  CUresult split_r = cuDevSmResourceSplitByCount(
      &primary_split, &n_groups, &sm_resource, &remaining_split,
      /*useFlags*/ 0, target_primary);
  if (split_r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(split_r, &msg);
    std::printf("cuDevSmResourceSplitByCount: %d (%s) — split FAILED\n",
                (int)split_r, msg ? msg : "?");
    return 2;
  }
  std::printf("split ok: primary.smCount=%u remaining.smCount=%u n_groups=%u\n",
              primary_split.sm.smCount, remaining_split.sm.smCount, n_groups);

  // Generate descriptors and create green contexts.
  CUdevResourceDesc primary_desc, remaining_desc;
  CK(cuDevResourceGenerateDesc(&primary_desc, &primary_split, 1));
  CK(cuDevResourceGenerateDesc(&remaining_desc, &remaining_split, 1));

  CUgreenCtx primary_gctx, remaining_gctx;
  CUresult gr = cuGreenCtxCreate(&primary_gctx, primary_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
  if (gr != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(gr, &msg);
    std::printf("cuGreenCtxCreate(primary): %d (%s) — UNSUPPORTED on this device\n",
                (int)gr, msg ? msg : "?");
    return 3;
  }
  CK(cuGreenCtxCreate(&remaining_gctx, remaining_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  std::printf("green contexts created OK\n");

  // Create streams bound to each partition.
  CUstream s1, s2;
  CK(cuGreenCtxStreamCreate(&s1, primary_gctx, CU_STREAM_NON_BLOCKING, 0));
  CK(cuGreenCtxStreamCreate(&s2, remaining_gctx, CU_STREAM_NON_BLOCKING, 0));
  std::printf("partition streams created OK\n");

  // Tear down.
  CK(cuStreamDestroy(s1));
  CK(cuStreamDestroy(s2));
  CK(cuGreenCtxDestroy(primary_gctx));
  CK(cuGreenCtxDestroy(remaining_gctx));
  std::printf("Green Context API: SUPPORTED on this hardware ✓\n");
  return 0;
}
