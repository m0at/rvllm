// Attention output gate kernels for Qwen3.5 full attention layers.
//
// Qwen3.5 uses attn_output_gate: the q_proj produces [m, 2 * num_heads * head_dim]
// where the layout is PER-HEAD interleaved (matching HuggingFace's
//   .view(-1, head_dim * 2).chunk(2, dim=-1) convention):
//
//   [q_h0(256), gate_h0(256), q_h1(256), gate_h1(256), ..., q_h23(256), gate_h23(256)]
//   |<-- head 0 (512) --->|<-- head 1 (512) ------->| ... |<-- head 23 (512) ---->|
//
// We split into separate query [m, q_dim] and gate [m, q_dim] tensors.
//
// Kernels:
// 1. truncate_q_kernel: extract query (first head_dim per head)
// 2. split_gate_kernel: extract gate (second head_dim per head)
// 3. sigmoid_gate_kernel: output = input * sigmoid(gate)

// Extract the query portion from the q_proj output.
// q_full is [m, num_heads * head_dim * 2] with per-head interleaved layout.
// q_out is [m, q_dim] with contiguous query values (all heads concatenated).
//
// For each output element at (row, head, d):
//   q_out[row * q_dim + head * head_dim + d] =
//     q_full[row * full_dim + head * 2 * head_dim + d]
//
// Parameters:
//   m = number of tokens
//   q_dim = num_heads * head_dim (output dimension per token)
//   head_dim = dimension per head
extern "C"
__global__ void truncate_q_kernel(
    float* __restrict__ q_out,
    const float* __restrict__ q_full,
    int m,
    int q_dim,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * q_dim;
    if (idx < total) {
        int row = idx / q_dim;
        int within_row = idx % q_dim;
        int head = within_row / head_dim;
        int d = within_row % head_dim;
        int full_dim = 2 * q_dim;
        // Per-head interleave: query is the first head_dim elements of each head's 2*head_dim block
        q_out[idx] = q_full[row * full_dim + head * 2 * head_dim + d];
    }
}

// Extract the gate portion from the q_proj output.
// q_full is [m, num_heads * head_dim * 2] with per-head interleaved layout.
// gate is [m, q_dim] with contiguous gate values (all heads concatenated).
//
// For each output element at (row, head, d):
//   gate[row * q_dim + head * head_dim + d] =
//     q_full[row * full_dim + head * 2 * head_dim + head_dim + d]
extern "C"
__global__ void split_gate_kernel(
    float* __restrict__ gate,
    const float* __restrict__ q_full,
    int m,
    int q_dim,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * q_dim;
    if (idx < total) {
        int row = idx / q_dim;
        int within_row = idx % q_dim;
        int head = within_row / head_dim;
        int d = within_row % head_dim;
        int full_dim = 2 * q_dim;
        // Per-head interleave: gate is the second head_dim elements of each head's 2*head_dim block
        gate[idx] = q_full[row * full_dim + head * 2 * head_dim + head_dim + d];
    }
}

// Apply sigmoid gating: output[i] = input[i] * sigmoid(gate[i])
// Both input and gate are [n] elements (flattened [m, q_dim]).
extern "C"
__global__ void sigmoid_gate_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = 1.0f / (1.0f + expf(-gate[idx]));
        output[idx] = input[idx] * g;
    }
}
