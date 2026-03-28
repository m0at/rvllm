// RMSNorm kernel: output[i] = input[i] * weight[i] / sqrt(mean(input^2) + eps)
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size/4, 1024), 1, 1)
//   Shared memory: (blockDim.x / 32 + 1) * sizeof(float) (for warp reduction)
//
// Each block processes one token (one row of the input matrix).
// Uses float4 vectorized loads/stores and warp shuffle reduction.
// Requires hidden_size to be a multiple of 4.

extern "C"
__global__ void rms_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int vec_size = hidden_size / 4;

    const float4* x = reinterpret_cast<const float4*>(input + token_idx * hidden_size);
    float4* y = reinterpret_cast<float4*>(output + token_idx * hidden_size);
    const float4* w = reinterpret_cast<const float4*>(weight);

    // Step 1: Compute partial sum of squares with float4 loads
    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += stride) {
        float4 v = x[i];
        local_ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    }

    // Cross-warp reduction via shared memory
    extern __shared__ float sdata[];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (lane == 0) {
        sdata[warp_id] = local_ss;
    }
    __syncthreads();

    // First warp reduces across all warps
    const int num_warps = (stride + 31) >> 5;
    float sum = (tid < num_warps) ? sdata[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Broadcast rms_scale to all threads
    if (tid == 0) {
        sdata[0] = rsqrtf(sum / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms_scale = sdata[0];

    // Step 2: Apply normalization with float4 stores
    for (int i = tid; i < vec_size; i += stride) {
        float4 v = x[i];
        float4 wt = w[i];
        float4 out;
        out.x = v.x * wt.x * rms_scale;
        out.y = v.y * wt.y * rms_scale;
        out.z = v.z * wt.z * rms_scale;
        out.w = v.w * wt.w * rms_scale;
        y[i] = out;
    }
}
