#include <cuda_fp16.h>
#include <math.h>

extern "C"
__global__ void embedding_gather_scale_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ embedding,
    const int* __restrict__ token_ids,
    int hidden_size,
    int vocab_size,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) return;

    const __half* src = embedding + (long long)token_id * hidden_size;
    __half* dst = output + (long long)token_idx * hidden_size;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        dst[i] = __float2half(__half2float(src[i]) * scale);
    }
}

extern "C"
__global__ void head_rms_norm_f16_kernel(
    __half* __restrict__ tensor,
    const __half* __restrict__ weight,
    float eps,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    __half* row = tensor + ((long long)token_idx * num_heads + head_idx) * head_dim;
    extern __shared__ float sdata[];

    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(row[i]);
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        row[i] = __float2half(__half2float(row[i]) * inv * __half2float(weight[i]));
    }
}

extern "C"
__global__ void head_rms_norm_noscale_f16_kernel(
    __half* __restrict__ tensor,
    float eps,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    __half* row = tensor + ((long long)token_idx * num_heads + head_idx) * head_dim;
    extern __shared__ float sdata[];

    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(row[i]);
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        row[i] = __float2half(__half2float(row[i]) * inv);
    }
}

extern "C"
__global__ void scale_f16_inplace_kernel(
    __half* __restrict__ tensor,
    const __half* __restrict__ scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tensor[idx] = __float2half(__half2float(tensor[idx]) * __half2float(scale[0]));
    }
}
