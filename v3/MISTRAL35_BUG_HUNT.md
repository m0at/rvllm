**Short Answer**
Your #2 is probably not the bug. HF and vLLM both apply YaRN’s attention factor by multiplying both `cos` and `sin`, so Q and K both get scaled and attention scores effectively get `mscale^2 / sqrt(d)`. For this checkpoint, that is expected behavior, not a double-application bug.

The top suspects now are: attention/KV path at nonzero positions, a later-layer projection/dequant issue, or lm_head orientation/data mismatch. RoPE convention and YaRN table math look mostly correct.

**Candidate Ranking**
1. **Layer-by-layer divergence after layer 0**: most likely. Layer-0 scalar ranges are plausible, but that does not prove all 88 layers or all projection kinds are correct.
2. **KV cache / attention scores at last prompt token**: very plausible. The first place where sequential prefill differs from the trivial BOS test is layer-0 attention over `past_len > 1`.
3. **lm_head BF16 GEMM/layout**: plausible if `h_after_final_norm` cosine against vLLM is good but logits are wrong.
4. **RoPE interleaving mismatch**: unlikely. vLLM initializes Llama-family RoPE with `is_neox_style = True`, which is split-half.
5. **YaRN mscale “double application”**: unlikely. vLLM’s YaRN cache does `cos = cos(...) * self.mscale` and `sin = sin(...) * self.mscale`; HF applies `attention_scaling` to both cos and sin.

**Source Findings**
Local rvllm YaRN tables:
- [mistral35_yarn.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_yarn.rs:91): `yarn_mscale()` computes `0.1 * scale * ln(factor) + 1`.
- [mistral35_yarn.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_yarn.rs:125): base inv-freq exponent is `2*i/head_dim`, matching HF/vLLM.
- [mistral35_yarn.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_yarn.rs:293): tables multiply both `cos` and `sin` by `mscale`.

Local RoPE kernel:
- [rope_split_half_bf16.cu](/home/r00t/workspace/upstream/rvllm-serve/kernels/rope_split_half_bf16.cu:37): pairs `i` with `i + half`, i.e. NeoX/split-half.

Forward path:
- [mistral35_bring_up.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_bring_up.rs:1281): `past_len = position + 1`.
- [mistral35_bring_up.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_bring_up.rs:1289): attention scale is `1/sqrt(head_dim)`.
- [mistral35_bring_up.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_bring_up.rs:1458): RoPE is applied to Q and K before dumps.
- [mistral35_bring_up.rs](/home/r00t/workspace/upstream/rvllm-serve/v3/crates/rvllm-runtime/src/mistral35_bring_up.rs:1719): lm_head path is separate BF16 GEMM.

Reference behavior:
- HF RoPE docs say YaRN has an `attention_factor` applied to computed cos/sin: https://huggingface.co/docs/transformers/main/internal/rope_utils
- HF `modeling_rope_utils.py` computes YaRN `attention_factor` from `factor/mscale/mscale_all_dim`, then returns it separately; Llama/Mistral rotary forward multiplies both cos and sin by that scaling.
- vLLM `YaRNScalingRotaryEmbedding` computes `self.mscale`, then multiplies both `cos` and `sin`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py
- vLLM Llama/Mistral-family attention uses `is_neox_style = True`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

**Concrete Next Dumps**
Add dumps for the **last prompt token**, layer 0:

1. `q_pre_rope`, `k_pre_rope`, `v`
2. `q_post_rope`, `k_post_rope`
3. layer-0 `K_cache[0..position]`, `V_cache[0..position]`
4. `scores[head, 0..position]` before softmax
5. softmax weights for a few heads
6. `attn_out`, `o_out`, `h_after_attn`, `h_after_layer0`

In numpy, dequant layer-0 weights and reproduce exactly:
`embed -> rmsnorm -> q/k/v -> rope -> qk scores -> softmax_v -> o_proj -> residual -> mlp`.

The fastest discriminator is:

- If `q_pre_rope/k_pre_rope/v` mismatch: NVFP4 GEMM/dequant or tensor layout.
- If pre-RoPE matches but post-RoPE mismatches: RoPE table/convention.
- If post-RoPE matches but scores mismatch: KV cache layout, GQA mapping, or scale.
- If scores match but `attn_out` mismatches: softmax/V cache.
- If layer 0 matches, dump only `h_after_layerN` cosine for all 88 layers and bisect the first bad layer.
