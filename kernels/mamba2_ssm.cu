// Gated DeltaNet kernels for Qwen3.5 linear attention decode (single token).
//
// Implements the Gated Delta Rule used by Qwen3.5:
//   - Depthwise causal conv1d with state caching
//   - L2-normalize Q and K
//   - Gated delta rule state update:
//       S_decayed = exp(g) * S
//       S = S_decayed + outer(k, beta * (v - S_decayed^T @ k))
//   - Output: y = S^T @ q * scale
//   - Gated RMSNorm: output = silu(z) * rms_norm(y)
//
// Architecture dimensions (Qwen3.5-27B):
//   num_key_heads=16, num_value_heads=48, key_head_dim=128, value_head_dim=128
//   conv_dim=10240 (qkv output), conv_kernel=4
//   GQA ratio: 48/16 = 3 value heads per key head
//   Recurrent state shape: [num_value_heads, key_head_dim, value_head_dim] = [48, 128, 128]

#include <cuda_fp16.h>
#include <math.h>

// ============================================================================
// Kernel 1: Depthwise causal conv1d update (single token decode)
// ============================================================================
// conv_state: [num_channels, kernel_size] (f32, rolling buffer of last kernel_size values)
// conv_weight: [num_channels, kernel_size] (f32, depthwise conv weights)
// conv_bias: [num_channels] (f32, optional bias -- pass NULL if none)
// input: [num_channels] (f32, current token's projected values)
// output: [num_channels] (f32, convolved + SiLU activated output)
// num_channels: total channels (e.g., 10240 for QKV)
// kernel_size: conv kernel size (e.g., 4)
//
// After computation:
//   1. Shifts conv_state left by 1
//   2. Inserts input as the new rightmost column
//   3. Computes depthwise conv1d over the full buffer + applies SiLU

extern "C" __global__ void mamba2_conv1d_step(
    float* __restrict__ conv_state,   // [num_channels, kernel_size]
    const float* __restrict__ conv_weight, // [num_channels, kernel_size]
    const float* __restrict__ conv_bias,   // [num_channels] or NULL
    const float* __restrict__ input,  // [num_channels]
    float* __restrict__ output,       // [num_channels]
    int num_channels,
    int kernel_size
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= num_channels) return;

    // Update conv_state: shift left and insert new input
    // State layout: [ch * kernel_size + 0..kernel_size-1]
    int base = ch * kernel_size;
    for (int k = 0; k < kernel_size - 1; k++) {
        conv_state[base + k] = conv_state[base + k + 1];
    }
    conv_state[base + kernel_size - 1] = input[ch];

    // Compute conv: sum(weight[k] * state[k]) for k in 0..kernel_size
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        acc += conv_weight[ch * kernel_size + k] * conv_state[base + k];
    }

    // Add bias if present
    if (conv_bias != NULL) {
        acc += conv_bias[ch];
    }

    // SiLU activation: x * sigmoid(x)
    output[ch] = acc / (1.0f + expf(-acc));
}

// ============================================================================
// Kernel 2: Compute gate parameters (beta, g) from in_proj_a/b outputs
// ============================================================================
// For each value head h:
//   beta[h] = sigmoid(b[h])
//   g[h] = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
//
// a: [num_value_heads] (f32, output of in_proj_a @ hidden)
// b: [num_value_heads] (f32, output of in_proj_b @ hidden)
// A_log: [num_value_heads] (f32, log decay rate)
// dt_bias: [num_value_heads] (f32, timestep bias)
// beta_out: [num_value_heads] (f32, write strength)
// g_out: [num_value_heads] (f32, negative log-decay)

extern "C" __global__ void mamba2_compute_gates(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ beta_out,
    float* __restrict__ g_out,
    int num_value_heads
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_value_heads) return;

    // beta = sigmoid(b)
    float b_val = b[h];
    beta_out[h] = 1.0f / (1.0f + expf(-b_val));

    // g = -exp(A_log) * softplus(a + dt_bias)
    float a_val = a[h] + dt_bias[h];
    float sp = logf(1.0f + expf(a_val));  // softplus
    g_out[h] = -expf(A_log[h]) * sp;      // always negative -> exp(g) in (0, 1)
}

// ============================================================================
// Kernel 3: L2-normalize vectors in-place
// ============================================================================
// Normalizes each head's vector: x = x / (||x||_2 + eps)
// data: [num_heads, head_dim] (f32)
// num_heads: number of head vectors
// head_dim: dimension of each vector
// eps: small constant for numerical stability

extern "C" __global__ void mamba2_l2_normalize(
    float* __restrict__ data,
    int num_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= num_heads) return;

    int base = h * head_dim;

    // Compute sum of squares using shared memory reduction
    extern __shared__ float smem[];

    float sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = data[base + d];
        sum_sq += val * val;
    }
    smem[tid] = sum_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(smem[0] + eps);

    // Normalize
    for (int d = tid; d < head_dim; d += blockDim.x) {
        data[base + d] *= inv_norm;
    }
}

// ============================================================================
// Kernel 4: GQA expand (repeat_interleave for key heads -> value heads)
// ============================================================================
// Expands num_key_heads vectors to num_value_heads by repeating each
// key head (num_value_heads / num_key_heads) times.
// input: [num_key_heads, head_dim]
// output: [num_value_heads, head_dim]

extern "C" __global__ void mamba2_gqa_expand(
    const float* __restrict__ input,   // [num_key_heads, head_dim]
    float* __restrict__ output,        // [num_value_heads, head_dim]
    int num_key_heads,
    int num_value_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_value_heads * head_dim;
    if (idx >= total) return;

    int vh = idx / head_dim;
    int d = idx % head_dim;
    int ratio = num_value_heads / num_key_heads;
    int kh = vh / ratio;

    output[idx] = input[kh * head_dim + d];
}

// ============================================================================
// Kernel 5: Gated DeltaNet SSM step (single token decode)
// ============================================================================
// For each value head h:
//   1. Decay: S_decayed[h] = exp(g[h]) * S[h]
//   2. Retrieve: kv_mem[h] = (S_decayed[h] * k[h]).sum(dim=key_dim)  -> [value_head_dim]
//   3. Delta: delta = beta[h] * (v[h] - kv_mem)
//   4. Update: S[h] = S_decayed[h] + outer(k[h], delta)
//   5. Output: y[h] = (S[h] * q[h]).sum(dim=key_dim) * scale
//
// Grid: (num_value_heads, 1, 1)
// Block: (value_head_dim, 1, 1)  -- one thread per VALUE dimension (coalesced!)
//
// Memory layout: state[h][i][j] where i=key_dim, j=value_dim
// Thread j iterates over i, reading state[h*khd*vhd + i*vhd + j].
// Consecutive threads (j, j+1, ...) access consecutive memory addresses
// → perfectly coalesced reads/writes, no atomics needed.
//
// ssm_state: [num_value_heads, key_head_dim, value_head_dim] (f32)
// q: [num_value_heads, key_head_dim] (f32, L2-normed, GQA-expanded)
// k: [num_value_heads, key_head_dim] (f32, L2-normed, GQA-expanded)
// v: [num_value_heads, value_head_dim] (f32)
// beta: [num_value_heads] (f32, write strength from sigmoid)
// g: [num_value_heads] (f32, negative decay factor)
// y_out: [num_value_heads, value_head_dim] (f32)
// scale: 1.0 / sqrt(key_head_dim)

extern "C" __global__ void mamba2_ssm_step(
    float* __restrict__ ssm_state,    // [num_value_heads, key_head_dim, value_head_dim]
    const float* __restrict__ q,      // [num_value_heads, key_head_dim]
    const float* __restrict__ k,      // [num_value_heads, key_head_dim]
    const float* __restrict__ v,      // [num_value_heads, value_head_dim]
    const float* __restrict__ beta,   // [num_value_heads]
    const float* __restrict__ g,      // [num_value_heads]
    float* __restrict__ y_out,        // [num_value_heads, value_head_dim]
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    float scale
) {
    int h = blockIdx.x;   // value head index
    int j = threadIdx.x;  // VALUE dimension index (coalesced!)
    if (h >= num_value_heads || j >= value_head_dim) return;

    float decay = expf(g[h]);
    float beta_h = beta[h];
    float v_val = v[h * value_head_dim + j];

    int head_base = h * key_head_dim * value_head_dim;

    // Pass 1: Decay state and compute kv_mem[j] = sum_i(S_decayed[i][j] * k[i])
    float kv_mem = 0.0f;
    for (int i = 0; i < key_head_dim; i++) {
        int idx = head_base + i * value_head_dim + j;
        float s = ssm_state[idx] * decay;
        ssm_state[idx] = s;
        kv_mem += s * k[h * key_head_dim + i];
    }

    // Compute delta for this j: delta = beta * (v[j] - kv_mem[j])
    float delta = beta_h * (v_val - kv_mem);

    // Pass 2: Update state with delta and compute output
    // y[j] = sum_i(S_updated[i][j] * q[i]) * scale
    float y_val = 0.0f;
    for (int i = 0; i < key_head_dim; i++) {
        int idx = head_base + i * value_head_dim + j;
        float s = ssm_state[idx] + k[h * key_head_dim + i] * delta;
        ssm_state[idx] = s;
        y_val += s * q[h * key_head_dim + i] * scale;
    }

    y_out[h * value_head_dim + j] = y_val;
}

// ============================================================================
// Kernel 6: RMSNorm + SiLU gate (fused) -- per-head normalization
// ============================================================================
// Applies per-head RMSNorm to y, then gates with silu(z):
//   output = silu(z) * rms_norm(y, weight)
//
// y: [num_value_heads * value_head_dim] (f32, SSM output)
// z: [num_value_heads * value_head_dim] (f32, gate values)
// weight: [value_head_dim] (f32, shared across heads)
// output: [num_value_heads * value_head_dim] (f32)
// Grid: (num_value_heads, 1, 1)
// Block: (min(value_head_dim, 128), 1, 1)

extern "C" __global__ void mamba2_norm_gate(
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_value_heads,
    int value_head_dim,
    float eps
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= num_value_heads || tid >= value_head_dim) return;

    int base = head * value_head_dim;

    // Compute RMS for this head
    __shared__ float rms_shared;
    float sum_sq = 0.0f;
    for (int d = tid; d < value_head_dim; d += blockDim.x) {
        float val = y[base + d];
        sum_sq += val * val;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduce
    __shared__ float warp_sums_rms[4];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums_rms[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < (value_head_dim + 31) / 32; w++) {
            total += warp_sums_rms[w];
        }
        rms_shared = rsqrtf(total / (float)value_head_dim + eps);
    }
    __syncthreads();

    // Apply norm + gate
    for (int d = tid; d < value_head_dim; d += blockDim.x) {
        float y_val = y[base + d] * rms_shared * weight[d];
        float z_val = z[base + d];
        float gate = z_val / (1.0f + expf(-z_val));  // silu(z)
        output[base + d] = gate * y_val;
    }
}

// ============================================================================
// Kernel 7: Initialize state to zeros
// ============================================================================
extern "C" __global__ void mamba2_init_state(
    float* __restrict__ data,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0.0f;
}

// ============================================================================
// Kernel 8: Small GEMV -- in_proj_a/b @ hidden (BF16 weight, f32 hidden)
// ============================================================================
// Computes output = weight @ input where weight is stored as f32 (converted from BF16)
// weight: [out_dim, in_dim] (f32)
// input: [in_dim] (f32)
// output: [out_dim] (f32)
// Grid: (out_dim, 1, 1)
// Block: (256, 1, 1) -- threads cooperate on dot product via reduction

extern "C" __global__ void mamba2_small_gemv(
    const float* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_dim,
    int in_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int tid = threadIdx.x;
    extern __shared__ float partial[];

    float sum = 0.0f;
    for (int col = tid; col < in_dim; col += blockDim.x) {
        sum += weight[row * in_dim + col] * input[col];
    }
    partial[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = partial[0];
    }
}
