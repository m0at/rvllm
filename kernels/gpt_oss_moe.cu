#include <cuda_fp16.h>

namespace {

constexpr int MAX_EXPERTS = 64;

__device__ __forceinline__ float mxfp4_value(int idx) {
    switch (idx & 0xF) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        case 7: return 6.0f;
        case 8: return -0.0f;
        case 9: return -0.5f;
        case 10: return -1.0f;
        case 11: return -1.5f;
        case 12: return -2.0f;
        case 13: return -3.0f;
        case 14: return -4.0f;
        default: return -6.0f;
    }
}

}  // namespace

extern "C" __global__ void gpt_oss_route_topk_kernel(
    const float* router_logits,
    int* topk_indices,
    float* topk_weights,
    int num_experts,
    int top_k
) {
    const int token_idx = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    if (num_experts <= 0 || num_experts > MAX_EXPERTS || top_k <= 0) {
        return;
    }

    const float* logits = router_logits + token_idx * num_experts;
    int* out_indices = topk_indices + token_idx * top_k;
    float* out_weights = topk_weights + token_idx * top_k;

    float selected_logits[MAX_EXPERTS];
    bool used[MAX_EXPERTS];
    #pragma unroll
    for (int i = 0; i < MAX_EXPERTS; ++i) {
        used[i] = false;
    }

    const int effective_top_k = top_k < num_experts ? top_k : num_experts;
    for (int rank = 0; rank < effective_top_k; ++rank) {
        int best_idx = 0;
        float best_val = -3.402823466e+38f;
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            if (!used[expert_idx] && logits[expert_idx] > best_val) {
                best_val = logits[expert_idx];
                best_idx = expert_idx;
            }
        }
        used[best_idx] = true;
        out_indices[rank] = best_idx;
        selected_logits[rank] = best_val;
    }

    float max_logit = selected_logits[0];
    for (int rank = 1; rank < effective_top_k; ++rank) {
        if (selected_logits[rank] > max_logit) {
            max_logit = selected_logits[rank];
        }
    }
    float sum = 0.0f;
    for (int rank = 0; rank < effective_top_k; ++rank) {
        float w = expf(selected_logits[rank] - max_logit);
        out_weights[rank] = w;
        sum += w;
    }
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int rank = 0; rank < effective_top_k; ++rank) {
        out_weights[rank] *= inv_sum;
    }

    for (int rank = effective_top_k; rank < top_k; ++rank) {
        out_indices[rank] = 0;
        out_weights[rank] = 0.0f;
    }
}

extern "C" __global__ void gpt_oss_select_expert_inputs_kernel(
    float* masked_input,
    float* route_weights,
    const float* input,
    const int* topk_indices,
    const float* topk_weights,
    int hidden_size,
    int top_k,
    int expert_idx
) {
    const int token_idx = blockIdx.x;
    __shared__ float token_weight;

    if (threadIdx.x == 0) {
        float w = 0.0f;
        const int route_base = token_idx * top_k;
        for (int rank = 0; rank < top_k; ++rank) {
            if (topk_indices[route_base + rank] == expert_idx) {
                w = topk_weights[route_base + rank];
                break;
            }
        }
        token_weight = w;
        route_weights[token_idx] = w;
    }
    __syncthreads();

    const float* src = input + token_idx * hidden_size;
    float* dst = masked_input + token_idx * hidden_size;
    for (int hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
        dst[hidden_idx] = token_weight > 0.0f ? src[hidden_idx] : 0.0f;
    }
}

extern "C" __global__ void gpt_oss_dequant_expert_f16_kernel(
    half* output,
    const unsigned char* blocks,
    const unsigned char* scales,
    int expert_idx,
    int out_features,
    int in_features
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_features * in_features;
    if (idx >= total) {
        return;
    }

    const int out_idx = idx / in_features;
    const int in_idx = idx % in_features;
    const int groups = (in_features + 31) / 32;
    const int group_idx = in_idx / 32;
    const int pair_idx = (in_idx % 32) / 2;
    const int expert_row = expert_idx * out_features + out_idx;
    const int block_offset = (expert_row * groups + group_idx) * 16 + pair_idx;
    const unsigned char packed = blocks[block_offset];
    const int nibble = (in_idx & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
    const unsigned int scale_bits = static_cast<unsigned int>(scales[expert_row * groups + group_idx]) << 23;
    const float scale = __uint_as_float(scale_bits);
    output[idx] = __float2half_rn(mxfp4_value(nibble) * scale);
}

extern "C" __global__ void gpt_oss_weighted_add_kernel(
    float* output,
    const float* expert_out,
    const float* route_weights,
    int hidden_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * hidden_size;
    if (idx >= total) {
        return;
    }
    const int token_idx = idx / hidden_size;
    output[idx] += expert_out[idx] * route_weights[token_idx];
}
