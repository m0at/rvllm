# 09: Speculative Decoding

## 1. Current State of Speculative Decoding in rvLLM

### 1.1 What Is Implemented

The speculative decoding crate (`crates/rvllm-speculative/`) contains a well-structured but incomplete implementation:

**Core algorithm (verification):** Fully implemented and tested. The file `crates/rvllm-speculative/src/verification.rs` contains the complete Leviathan et al. (2023) speculative sampling algorithm: modified rejection sampling with adjusted distribution resampling on rejection. The K+1 distribution handling is correct (bonus token from the target's K+1-th position on full acceptance). Test coverage includes self-speculation (100% acceptance), zero-target rejection, partial acceptance, probabilistic acceptance rate validation (10,000 trials confirming 0.5 rate for 0.4/0.8 ratio), and empty input.

**Draft model trait:** The `DraftModel` trait in `crates/rvllm-speculative/src/draft.rs` defines the pluggable interface: `generate(input_tokens, num_tokens) -> Vec<DraftToken>` where each `DraftToken` carries `token_id`, `logprob`, and `draft_probs` (full vocabulary distribution). The default `DraftModelRunner` is a placeholder that uses deterministic selection (modular arithmetic), not a real model.

**Self-draft model:** `crates/rvllm-speculative/src/self_draft.rs` implements early-exit speculation using a `PartialForwardFn` callback that runs the first N layers of the target model plus RMSNorm + LM head. Greedy argmax selection with softmax probability extraction. The GPU integration exists in `gpu_engine.rs` (lines 1293-1341) where `build_self_draft` constructs the callback using `forward_logits_partial` on the GPU worker.

**Speculative engine:** `crates/rvllm-speculative/src/engine.rs` wraps the full draft+verify loop with a `TargetModel` trait abstraction. Supports single-request `step()` and multi-step `generate()` with EOS detection and max_tokens limit. Tracks `SpeculativeMetrics` (acceptance rate, speedup ratio).

**GPU target model adapter:** `GpuTargetModel` in `gpu_engine.rs` (lines 1141-1248) wraps the GPU worker's forward pass as a `TargetModel`. Constructs synthetic single-sequence metadata, runs a full prefill-style forward pass, extracts K+1 probability distributions via softmax over logits.

**Batch scheduler:** `crates/rvllm-speculative/src/scheduler.rs` handles multi-sequence speculative decoding: generates draft tokens per sequence, builds concatenated verification input with per-sequence offset tracking, and provides `verify_all()` to demux batch verification results.

**Benchmark harness:** `benchmark_speculative()` in `gpu_engine.rs` (lines 1415-1480) runs both standard decode and speculative decode on the same prompt, reporting comparative tok/s and wall-clock speedup.

### 1.2 What Is Missing

The following are the critical gaps between the current scaffolding and a production-ready speculative decoding system:

1. **No real draft model loading.** `DraftModelRunner` is a placeholder. There is no code to load a second set of model weights (e.g., Qwen2.5-0.5B) as a draft model. The dependency `rvllm-model-runner` is commented out in `Cargo.toml` (line 23).

2. **Self-draft runs full context every step.** The `SelfDraftModel.generate()` runs the partial forward on the entire context for each of the K draft tokens sequentially. There is no incremental/KV-cached draft generation. At context length 512, generating K=5 draft tokens means 5 partial forward passes of 512+ tokens. This is extremely expensive and likely slower than just doing 5 normal decode steps.

3. **No KV cache sharing between draft and verify.** The draft phase and verification phase use independent forward passes with no shared KV cache state. The verification pass recomputes the full context from scratch (it constructs `is_prompt: true` metadata with all tokens). This means every speculative step does a full prefill of the entire context, not an incremental decode.

4. **No CUDA graph integration.** Speculative decode steps cannot use CUDA graph replay. The `GraphRunner` (in `crates/rvllm-worker/src/graph_runner.rs`) only captures decode steps (single token per sequence). The draft phase (K sequential forward passes) and verify phase (K+1 token prefill) have variable shapes that are not currently graphed.

5. **No tree-structured speculation.** Verification is strictly linear (K tokens in sequence). No branching or multiple draft sequences.

6. **No integration with continuous batching.** Speculative decode is a separate path (`generate_speculative`) that bypasses the main engine loop, scheduler, and output processor. It cannot interleave with regular batched decode.

7. **No Medusa/EAGLE/Lookahead/n-gram support.** Only vanilla draft-model and self-draft approaches exist.

8. **No adaptive K.** The number of speculative tokens K is fixed at config time. No runtime adaptation based on observed acceptance rates.

### 1.3 Key Code References

| Component | File | Lines |
|---|---|---|
| Verification algorithm | `crates/rvllm-speculative/src/verification.rs` | 38-175 |
| DraftModel trait | `crates/rvllm-speculative/src/draft.rs` | 27-38 |
| Self-draft model | `crates/rvllm-speculative/src/self_draft.rs` | 17-111 |
| SpeculativeEngine | `crates/rvllm-speculative/src/engine.rs` | 112-271 |
| GPU target adapter | `crates/rvllm-engine/src/gpu_engine.rs` | 1141-1248 |
| Self-draft GPU builder | `crates/rvllm-engine/src/gpu_engine.rs` | 1293-1341 |
| Partial forward (GPU) | `crates/rvllm-model-runner/src/gpu_runner.rs` | 813-894 |
| CUDA graph pool | `crates/rvllm-gpu/src/cuda_graph.rs` | 1-80+ |
| Graph runner | `crates/rvllm-worker/src/graph_runner.rs` | 1-100+ |
| SpeculativeConfig | `crates/rvllm-speculative/src/config.rs` | 1-58 |

---

## 2. N=1 Performance Analysis

### 2.1 Current Numbers

From the latest benchmarks (H100 SXM 80GB, Qwen2.5-7B f16, direct engine, 128 tok/req):

| N | rvLLM (tok/s) | vLLM 0.18 (tok/s) | Ratio |
|---|---|---|---|
| 1 | 98 | 170 | 0.58x |
| 4 | 548 | 665 | 0.82x |
| 16 | 2,122 | 2,202 | 0.96x |

At N=1, rvLLM produces 98 tok/s, meaning each decode step takes approximately **10.2ms**. This is pure memory-bandwidth-bound decode: reading 14GB of FP16 weights (Qwen2.5-7B) from HBM once per token.

**Theoretical minimum decode latency (H100):**
- Qwen2.5-7B FP16 weights: ~14 GB
- H100 HBM3 bandwidth: 3.35 TB/s
- Minimum weight-read time: 14 / 3350 = 4.18 ms per token
- Theoretical peak N=1: 1000 / 4.18 = 239 tok/s

Current 98 tok/s = 41% of bandwidth-limited theoretical peak. The gap comes from: KV cache reads (~1 ms at 512 context), kernel launch overhead (~2.5 ms for 252+ kernels), attention computation, activation memory traffic, and CPU-GPU synchronization.

vLLM achieves 170 tok/s = 71% of theoretical peak, largely through CUDA graph replay (eliminates kernel launch overhead), Triton-optimized GEMMs, and mature scheduling.

### 2.2 Why Speculative Decoding Is the Right Fix for N=1

At N=1, the GPU is severely underutilized. An H100 has 989 TFLOPS of FP16 tensor core compute. A single decode step for Qwen2.5-7B requires approximately 14.3 GFLOP (dominated by GEMM operations across 28 layers). At 989 TFLOPS, the compute takes only 14.5 microseconds. The step is entirely memory-bandwidth-bound.

Speculative decoding converts the problem: instead of one decode step per token, we do (1 draft step + 1 verify step) to produce (1 + accepted) tokens. The verification step processes K+1 tokens in a single forward pass, which reads model weights once but produces K+1 sets of logits. Since the bottleneck is weight reads (not compute), verifying K+1 tokens costs almost the same wall time as generating 1 token.

**The speedup formula:**

```
speedup = E[tokens_per_step] / cost_ratio
where:
  E[tokens_per_step] = sum_{i=0}^{K} (1 - alpha) * alpha^i * (i + 1) + alpha^K * 1
                     = (1 - alpha^{K+1}) / (1 - alpha)    [for alpha < 1]
  cost_ratio = (draft_time + verify_time) / baseline_decode_time
```

If the draft model is negligibly fast relative to the target, `cost_ratio ~ 1`, and the speedup equals `E[tokens_per_step]`.

---

## 3. Speculative Decoding Approaches

### 3.1 Self-Draft (Early Exit / Shallow Layers)

**Concept:** Use the first N layers of the target model as a lightweight draft model. After N layers, apply the final RMSNorm and LM head to get approximate next-token predictions. The quality of these predictions depends on N/total_layers.

**Current rvLLM implementation:** Exists in `self_draft.rs`. The `PartialForwardFn` callback calls `forward_logits_partial()` which runs N layers then applies final norm + LM head. For Qwen2.5-7B (28 layers), the default is `total_layers / 4 = 7` layers.

**Acceptance rate model for self-draft:**

Research (Schuster et al., "Confident Adaptive Language Modeling," 2022; Elhoushi et al., "Layer Skip," 2024) shows that early-exit acceptance rates depend strongly on the ratio N/L and the task:

| Draft layers (N/28) | Expected alpha (greedy) | Expected alpha (sampling T=0.8) |
|---|---|---|
| 7 (25%) | 0.45-0.55 | 0.35-0.45 |
| 10 (36%) | 0.55-0.65 | 0.45-0.55 |
| 14 (50%) | 0.65-0.75 | 0.55-0.65 |

**Cost analysis for Qwen2.5-7B at N=1:**

Each draft step (7 layers) costs 7/28 = 25% of a full forward pass in compute, but the memory-bandwidth cost is: 7/28 of weights read + full LM head read. The LM head is `[vocab_size, hidden_size]` = `[152064, 3584]` = ~1.04 GB in FP16. Total 7B weights are ~14 GB, so 7 layers ~ 3.5 GB, plus 1.04 GB LM head = 4.54 GB. Cost ratio per draft token: 4.54/14 = 0.32.

For K=5 draft tokens + 1 verify pass:
- Draft cost: 5 * 0.32 = 1.6x baseline
- Verify cost: ~1.05x baseline (K+1=6 tokens, still bandwidth-limited at N=1)
- Total cost: 2.65x baseline
- Expected tokens at alpha=0.50, K=5: (1 - 0.50^6) / (1 - 0.50) = 1.97
- Effective speedup: 1.97 / 2.65 = **0.74x (SLOWER)**

**Conclusion:** Self-draft with separate forward passes per draft token is not viable for N=1. The draft model must be either (a) trivially cheap or (b) generate all K tokens in a single pass.

**Critical fix needed:** Incremental KV-cached draft generation. If the draft model reuses a KV cache, each successive draft token requires only a single-token decode through 7 layers (not a full-context forward). This changes the cost:
- Draft token 1: ~0.32x (includes context encoding)
- Draft tokens 2-5: ~0.01x each (single token through 7 layers, KV cached)
- Draft cost: 0.32 + 4*0.01 = 0.36x
- Verify cost: ~1.05x
- Total cost: 1.41x
- Effective speedup: 1.97 / 1.41 = **1.40x**

This requires sharing the KV cache between the draft partial-forward and the target model's full forward, which is architecturally non-trivial because the draft uses only 7 layers' worth of KV cache while the target needs all 28.

### 3.2 Medusa-Style: Multiple Draft Heads

**Concept:** (Cai et al., "Medusa," 2024) Add M small MLP heads on top of the target model's last hidden state. Each head i predicts the token at position t+i independently (not autoregressively). All M heads run in a single forward pass alongside the normal LM head.

**Architecture for Qwen2.5-7B:**
```
                    hidden_states [batch, 3584]
                           |
            /--------------+--------\-----\------\
         LM head        Head 1     Head 2  Head 3  Head 4
      [3584, 152064]  [3584, 152064] ...
           |              |           |       |       |
        token t+0      token t+1   t+2     t+3     t+4
```

Each Medusa head is a small residual MLP: `Linear(3584, 3584) -> SiLU -> Linear(3584, vocab_size)`. Memory per head: `3584*3584 + 3584*152064 = 12.8M + 545M = 558M params * 2 bytes = ~1.1 GB`. Four heads = ~4.4 GB additional memory.

**Acceptance rate model:**

Medusa heads predict independently, so the acceptance rate drops faster with position than autoregressive draft. Empirical results from the Medusa paper on 7B models:

| Position | Head accuracy (top-1) | Head accuracy (top-5) |
|---|---|---|
| t+1 | 0.60-0.70 | 0.85-0.90 |
| t+2 | 0.40-0.50 | 0.70-0.80 |
| t+3 | 0.25-0.35 | 0.55-0.65 |
| t+4 | 0.15-0.25 | 0.40-0.50 |

With tree-structured verification (multiple candidates per position from top-k), Medusa achieves 2.2-2.8 tokens per step on 7B models.

**Cost analysis at N=1:**

The Medusa heads are negligibly cheap compared to the model. The verification pass is a standard K+1 token forward (or tree-structured with multiple candidates). Total cost ratio is approximately 1.0x (heads add ~0.5% to forward time). With 2.5 expected tokens per step:
- Effective speedup: **2.5x** (98 -> ~245 tok/s)

**Disadvantage:** Requires fine-tuning the Medusa heads on representative data. Cannot be used off-the-shelf. Training cost: ~2-4 GPU-hours for a 7B model on 10M tokens.

### 3.3 EAGLE: Feature-Level Draft

**Concept:** (Li et al., "EAGLE," 2024) Instead of training heads that predict tokens from hidden states, EAGLE trains a lightweight autoregressive model that operates on the hidden state space. The draft model takes the target's hidden states at previous positions and autoregressively generates hidden states for future positions, which are then mapped to tokens via the target's LM head.

**Architecture:**
```
target hidden_states[t] --> EAGLE (1-2 transformer layers) --> hidden'[t+1]
hidden'[t+1] --> target LM head --> token[t+1]
hidden'[t+1] --> EAGLE --> hidden'[t+2]
...
```

The EAGLE model is a 1-2 layer transformer with the same hidden dimension as the target (3584 for Qwen2.5-7B). Memory overhead: 2 transformer layers = 2 * (4 * 3584^2 + 2 * 3584 * 18944) * 2 bytes ~ 2 * (51M + 136M) * 2 = ~0.75 GB.

**Acceptance rates:** EAGLE achieves significantly higher acceptance rates than Medusa because it models the autoregressive dependency between future tokens:

| K | EAGLE alpha (greedy) | Medusa alpha (greedy) |
|---|---|---|
| 1 | 0.75-0.85 | 0.60-0.70 |
| 2 | 0.65-0.75 | 0.40-0.50 |
| 3 | 0.55-0.65 | 0.25-0.35 |
| 5 | 0.40-0.50 | 0.10-0.15 |

With tree-structured verification (EAGLE-2), expected tokens per step: 3.0-4.0 for 7B models.

**Cost at N=1:** EAGLE draft is ~2/28 = 7% the cost of a full forward per draft token, plus one LM head evaluation per token (~7% of a full forward). Total draft cost for K=5: 5 * 0.14 = 0.70x. Verify cost: ~1.05x. Total: 1.75x. Expected tokens: 3.5. Effective speedup: 3.5/1.75 = **2.0x** (98 -> ~196 tok/s).

**Disadvantage:** Requires training the EAGLE module. More complex integration than Medusa.

### 3.4 Lookahead Decoding (Jacobi Iteration)

**Concept:** (Fu et al., "Lookahead Decoding," 2024) Instead of a separate draft model, use Jacobi iteration to solve the autoregressive equation simultaneously. Initialize future positions with guesses, then iteratively refine all positions in parallel until convergence.

**Algorithm:**
```
Step 0: x = [x1, guess2, guess3, ..., guessK]
Step 1: Run full model on x -> [_, pred2, pred3, ..., predK, predK+1]
        Accept pred2 if pred2 == guess2, pred3 if pred3 == guess3, etc.
        Update guesses: guess_i = pred_i
Step 2: Run again with updated guesses
...
```

**Acceptance rates:** Low for general text (each position's prediction depends on its left context, which is wrong). Typical acceptance: 0.2-0.3 per position. Better for repetitive/structured text (code, JSON).

**Cost at N=1:** Each lookahead step is a single forward pass with K+1 tokens, costing ~1.0x. If average accepted tokens per step is 1.3, speedup is **1.3x** (98 -> ~127 tok/s). Modest but requires no training and no additional memory.

### 3.5 Draft Model Options

#### 3.5a Same-Family Small Model (Qwen2.5-0.5B as draft for Qwen2.5-7B)

Qwen2.5-0.5B: 24 layers, hidden_size=896, intermediate=4864, 494M params, ~1 GB FP16 weights.

**Acceptance rate:** Same family, same tokenizer, same training data distribution. Expected alpha for greedy decoding: 0.65-0.75 (literature: "Draft & Verify," Chen et al., 2023).

**Cost at N=1:** Draft model decode: 1 GB weight read per token = ~0.30 ms. For K=5: 1.5 ms total draft time. Target verify (K+1=6 tokens): ~10.2 ms (same as one decode step, since bandwidth-limited). Total step: 11.7 ms. Expected tokens at alpha=0.70, K=5: (1 - 0.70^6) / (1 - 0.70) = 2.88. Effective tok/s: 2.88 / 11.7ms = 246 tok/s. **Speedup: 2.51x**.

Memory overhead: 1 GB for draft weights + ~0.1 GB for draft KV cache = 1.1 GB total. Negligible on H100 80GB.

#### 3.5b Quantized Main Model (INT4 Qwen2.5-7B as draft)

Qwen2.5-7B in INT4: ~3.5 GB weights. Same architecture, high agreement with FP16 target.

**Acceptance rate:** Very high for greedy decoding (the quantized model agrees with the full-precision model most of the time). Expected alpha: 0.80-0.90.

**Cost at N=1:** 3.5 GB weight read per token = ~1.04 ms. For K=5: 5.2 ms. Verify: ~10.2 ms. Total: 15.4 ms. Expected tokens at alpha=0.85, K=5: (1 - 0.85^6) / (1 - 0.85) = 3.87. Effective tok/s: 3.87 / 15.4ms = 251 tok/s. **Speedup: 2.56x**.

Memory overhead: 3.5 GB for INT4 weights + INT4 KV cache. Higher than small-model approach, but simpler (no separate model loading, same tokenizer guaranteed).

**Practical issue:** Requires INT4 inference support in rvLLM (currently only FP16 and FP8 are supported).

#### 3.5c N-gram Draft

Use the prompt and previously generated tokens to build an n-gram table. Draft tokens are predicted by matching the most recent n-1 tokens against the table.

**Acceptance rate:** Highly task-dependent. For code completion with repetitive patterns: 0.50-0.70. For creative text: 0.10-0.20. Average across benchmarks: 0.25-0.35.

**Cost at N=1:** Essentially zero GPU cost (CPU-only table lookup). Draft cost: ~0 ms. Verify cost: ~10.2 ms. Total: 10.2 ms. Expected tokens at alpha=0.30, K=5: 1.37. Effective tok/s: 1.37 / 10.2ms = 134 tok/s. **Speedup: 1.37x**.

**Advantage:** Zero additional memory. Zero training. Works with any model. Can be implemented in hours. Good baseline.

---

## 4. Verification Algorithm Details

### 4.1 Verifying K Draft Tokens in One Forward Pass

The verification step is the key insight of speculative decoding. Given context C and K draft tokens [d1, d2, ..., dK], the target model processes the concatenated sequence [C, d1, d2, ..., dK] in a single forward pass.

For the target model, this is processed as a prefill: all K+1 positions (the last context token position through the K-th draft position) produce logits simultaneously. The output is K+1 probability distributions:
- P_target(t | C) at position |C|-1 -> used to verify d1
- P_target(t | C, d1) at position |C| -> used to verify d2
- ...
- P_target(t | C, d1, ..., dK) at position |C|+K-1 -> bonus token distribution

Current rvLLM implementation in `GpuTargetModel::forward_verify()` (gpu_engine.rs:1169-1244):
1. Constructs synthetic single-sequence metadata with `is_prompt: true`
2. Runs full forward pass on `[C ++ drafts]`
3. Extracts the last `num_verify_positions` logit vectors
4. Applies softmax to convert logits to probabilities

**Critical performance issue:** The current implementation treats verification as a full prefill of the entire context + draft tokens. This means reading all model weights once AND computing attention over the full context length. For incremental decode with KV cache, the verification should only process the K draft tokens as new tokens with the existing KV cache from the context. This would make verification cost approximately equal to a K-token prefill chunk, not a full re-prefill.

### 4.2 The Accept/Reject Algorithm

The verification loop in `verification.rs` implements modified rejection sampling:

```
for i in 0..K:
    token = draft_tokens[i]
    dp = draft_probs[i][token]
    tp = target_probs[i][token]

    accept_prob = min(1, tp / dp)
    if random() < accept_prob:
        accept(token)
    else:
        // Resample from max(0, P_target - P_draft), normalized
        bonus = sample(max(0, target_probs[i] - draft_probs[i]))
        return (accepted_tokens, bonus)

// All K accepted: bonus from target_probs[K]
bonus = sample(target_probs[K])
return (all_K_tokens, bonus)
```

This guarantees that the output distribution is identical to sampling directly from the target model, regardless of draft quality. This is the correctness guarantee of speculative decoding.

---

## 5. Acceptance Rate Analysis for Qwen2.5-7B

### 5.1 Expected Tokens Per Step

For a per-token acceptance rate alpha and K draft tokens, the expected number of tokens produced per speculative step (including the bonus token) is:

```
E[tokens] = sum_{i=0}^{K-1} alpha^i * (1-alpha) * (i+1) + alpha^K * (K+1)
          = (1 - alpha^{K+1}) / (1 - alpha)
```

Table of E[tokens] for various alpha and K:

| alpha \ K | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| 0.50 | 1.88 | 1.94 | 1.97 | 1.98 | 1.99 | 2.00 |
| 0.60 | 2.18 | 2.31 | 2.39 | 2.43 | 2.46 | 2.48 |
| 0.65 | 2.36 | 2.53 | 2.65 | 2.74 | 2.80 | 2.85 |
| 0.70 | 2.56 | 2.79 | 2.95 | 3.07 | 3.17 | 3.24 |
| 0.75 | 2.78 | 3.09 | 3.32 | 3.49 | 3.63 | 3.74 |
| 0.80 | 3.04 | 3.43 | 3.74 | 3.99 | 4.19 | 4.36 |
| 0.85 | 3.33 | 3.83 | 4.25 | 4.59 | 4.88 | 5.13 |
| 0.90 | 3.69 | 4.30 | 4.86 | 5.35 | 5.79 | 6.19 |

### 5.2 Per-Approach Projections for Qwen2.5-7B at N=1

**Baseline:** 98 tok/s (10.2 ms/token)

| Approach | alpha | K | E[tokens] | Draft cost (ms) | Verify cost (ms) | Step time (ms) | Projected tok/s | Speedup |
|---|---|---|---|---|---|---|---|---|
| N-gram | 0.30 | 5 | 1.37 | ~0 | 10.2 | 10.2 | 134 | 1.37x |
| Self-draft (7L, KV-cached) | 0.50 | 5 | 1.97 | 2.5 | 10.2 | 12.7 | 155 | 1.58x |
| Qwen2.5-0.5B draft | 0.70 | 5 | 2.95 | 1.5 | 10.2 | 11.7 | 252 | 2.57x |
| Qwen2.5-0.5B draft | 0.70 | 7 | 3.17 | 2.1 | 10.2 | 12.3 | 258 | 2.63x |
| INT4 self-quantized | 0.85 | 5 | 3.87 | 5.2 | 10.2 | 15.4 | 251 | 2.56x |
| Medusa (4 heads) | 0.55* | 5* | 2.50* | ~0 | 10.2 | 10.2 | 245 | 2.50x |
| EAGLE (2 layers) | 0.65 | 5 | 2.65 | 1.4 | 10.2 | 11.6 | 228 | 2.33x |
| EAGLE-2 (tree) | 0.65 | 5 | 3.50* | 1.4 | 11.5 | 12.9 | 271 | 2.77x |

*Medusa uses tree-structured verification with multiple candidates per position, so effective E[tokens] is higher than the per-position alpha would suggest. The EAGLE-2 tree also boosts expected tokens through branching.

**Key insight:** The Qwen2.5-0.5B draft model approach is projected to deliver 250+ tok/s, beating vLLM's 170 tok/s by ~48%. This requires no training, just loading a second model.

### 5.3 Where the Draft Cost Estimates Come From

**Draft model decode latency** is memory-bandwidth-limited at N=1:

```
draft_time_per_token = draft_model_size_bytes / H100_bandwidth
Qwen2.5-0.5B: 0.99 GB / 3.35 TB/s = 0.30 ms/tok
Qwen2.5-1.5B: 3.0 GB / 3.35 TB/s = 0.90 ms/tok
Self-draft 7L: (3.5 + 1.04) GB / 3.35 TB/s = 1.35 ms/tok (first token; subsequent ~0.50 ms with KV cache)
```

**Verification latency** is also bandwidth-limited but with K+1 tokens contributing compute:

For K=5, the verification forward pass reads 14 GB of weights and processes 6 tokens. At N=1, this is still bandwidth-limited (6 tokens is far too few to saturate tensor cores on GEMM). The cost is approximately: weight read time + K * incremental_attention_time. At 512 context, attention for 6 extra tokens is negligible. So verify cost is approximately equal to one baseline decode step: ~10.2 ms.

---

## 6. Batch Speculative Decoding (N > 1)

### 6.1 How Speculation Interacts with Batching

At higher concurrency, speculative decoding's value proposition changes:

1. **The verify step becomes cheaper relative to normal decode.** At N=32, a normal decode step reads weights once for 32 tokens. A speculative verify step reads weights once for 32*(K+1) tokens. Since both are bandwidth-limited (at N=32), the verify step costs about the same as a normal step but produces K+1x more tokens per sequence.

2. **The draft step cost scales with N.** Running the draft model on N sequences costs N * draft_cost_per_sequence. If using a separate draft model, the draft model's forward pass can also batch all N sequences.

3. **At high N, the GPU becomes compute-bound.** Once N * K exceeds the point where GEMMs saturate tensor cores (roughly N=64 for 7B), the verify step costs significantly more than a normal decode step.

### 6.2 Speedup vs Concurrency Projections

Using Qwen2.5-0.5B draft, K=5, alpha=0.70:

| N | Baseline tok/s | Draft cost (ms) | Verify cost (ms) | Step time (ms) | Spec tok/s | Speedup | vs vLLM |
|---|---|---|---|---|---|---|---|
| 1 | 98 | 1.5 | 10.2 | 11.7 | 252 | 2.57x | 1.48x |
| 4 | 548 | 1.5 | 10.2 | 11.7 | 1,008 | 1.84x | 1.52x |
| 16 | 2,122 | 1.8 | 11.0 | 12.8 | 3,681 | 1.73x | 1.67x |
| 32 | 3,957 | 2.5 | 12.5 | 15.0 | 6,293 | 1.59x | 1.37x |
| 64 | 7,451 | 4.0 | 16.0 | 20.0 | 9,440 | 1.27x | 1.20x |
| 128 | 12,312 | 7.0 | 24.0 | 31.0 | 11,900 | 0.97x | 0.82x |

**Crossover point:** Around N=64-128, speculative decoding becomes compute-bound and the overhead of the draft model (which is now also compute-bound) makes speculation break even or slightly worse. The sweet spot is N=1 through N=32.

### 6.3 Interaction with Continuous Batching

The main challenge is that speculative decode produces a variable number of tokens per sequence per step (1 to K+1). In continuous batching, sequences that accept fewer tokens are "done" faster than those that accept all K+1, creating an imbalance.

**Approach 1: Uniform speculation.** All sequences in the batch speculate with the same K. After verification, sequences that rejected early have wasted compute on the verify pass but move forward correctly. The scheduler treats speculative decode as a special batch type.

**Approach 2: Spec-decode only for low-concurrency.** When the running batch has fewer than a threshold number of sequences (e.g., N < 32), use speculative decoding. When the batch is large enough that the GPU is well-utilized, fall back to normal decode. This is the simplest integration path and captures the N=1 use case where speculation matters most.

**Approach 3: Heterogeneous batching.** Some sequences in the batch use speculation, others do normal decode. The scheduler assigns speculation to sequences with high acceptance rates (measured online). This is complex but theoretically optimal.

For rvLLM's near-term goals, Approach 2 is recommended: speculation below N=32, normal decode above.

---

## 7. Tree-Structured Speculation

### 7.1 Concept

Instead of a single draft sequence [d1, d2, d3, d4, d5], generate a tree of candidates:

```
         d1
       /    \
      d2a    d2b
     / \      |
   d3a d3b   d3c
    |   |     |
   d4a d4b   d4c
```

The tree is verified in a single forward pass by flattening the tree into a sequence with appropriate attention masking (each node attends only to its ancestors). This allows the verifier to evaluate multiple hypotheses at each position.

### 7.2 Expected Tokens Per Step with Trees

For a binary tree of depth K with alpha=0.70:

| Tree structure | Total candidates | Verify tokens | E[tokens] | Improvement over linear |
|---|---|---|---|---|
| Linear K=5 | 5 | 6 | 2.95 | baseline |
| Binary K=3, width=2 | 7 | 8 | 3.35 | +14% |
| Top-2 at each level, K=4 | 15 | 16 | 3.82 | +29% |
| Top-3 at first, top-2 rest, K=4 | 21 | 22 | 4.10 | +39% |

The tradeoff: more candidates in the tree means more tokens in the verify pass, which increases verify cost. At N=1 (bandwidth-limited), the extra tokens in the verify pass are essentially free up to ~32 tokens. Beyond that, GEMMs start to become nontrivial.

### 7.3 Attention Masking for Tree Verification

The key implementation challenge is constructing the correct attention mask. In a tree, token at position j should attend to all tokens on its path from root, not to siblings or cousins. This requires a custom attention mask that is passed to FlashAttention.

Current rvLLM FlashAttention kernels (FA3 v3) use a causal mask. Tree verification requires a custom boolean mask, which the FA3 kernel would need to support. This is a significant engineering effort but provides the highest speedup ceiling.

---

## 8. CUDA Graph Integration for Speculative Decode

### 8.1 Why It Matters

At N=1, kernel launch overhead is a major fraction of step time. The 252+ kernel launches per forward pass take ~2.5 ms. With speculative decode, we have:
- K draft forward passes (K * 252 launches for self-draft; K * ~60 launches for 0.5B draft)
- 1 verification forward pass (252+ launches)

Without CUDA graphs, this means (K+1) * 252 = ~1,764 launches for K=5 self-draft, adding ~17.6 ms of pure launch overhead. This would completely negate the benefit of speculation.

### 8.2 What to Graph

**Draft model decode:** If using a separate draft model (Qwen2.5-0.5B), graph the single-token decode step for the draft. The draft model's decode has fixed input shape (1 token per sequence) and can be captured the same way as the target model's decode.

**Verification pass:** The verification pass has K+1 tokens, which is a fixed size for a given K. This can be captured as a CUDA graph for each K value. For tree verification, the token count varies by tree shape but can be padded to fixed sizes.

**Full spec-decode step:** Ideally, graph the entire speculative step: K draft decodes + 1 verification + accept/reject logic. The accept/reject is CPU-side, so this would require host-device synchronization within the graph, which CUDA graphs do not support. Instead, graph the draft and verify phases separately.

### 8.3 Implementation Plan for CUDA Graph Spec-Decode

1. Pre-allocate persistent GPU buffers for draft model metadata (same pattern as main model's graph runner).
2. Capture draft decode at batch_size=1 (or N for batched draft).
3. Capture verify pass at token_count=(K+1) (or padded).
4. In the speculative step:
   a. Update draft metadata buffers via memcpy_htod
   b. Replay draft graph K times
   c. Copy draft logits back to CPU
   d. CPU: select draft tokens, build verify input
   e. Update verify metadata buffers
   f. Replay verify graph
   g. Copy verify logits back to CPU
   h. CPU: run accept/reject, produce output tokens

Steps (d) and (h) require CPU-GPU synchronization (stream sync after logits copy). This is unavoidable but adds only ~2 synchronization points per speculative step vs ~K+1 without graphing.

---

## 9. Memory Overhead Analysis

### 9.1 Draft Model Weights

| Draft approach | Weight memory | Notes |
|---|---|---|
| Self-draft (N layers) | 0 GB | Shares target weights |
| Qwen2.5-0.5B FP16 | 1.0 GB | Separate model |
| Qwen2.5-1.5B FP16 | 3.0 GB | Separate model |
| INT4 Qwen2.5-7B | 3.5 GB | Quantized copy |
| Medusa (4 heads) | 4.4 GB | Additional parameters |
| EAGLE (2 layers) | 0.75 GB | Additional parameters |
| N-gram | ~0.001 GB | CPU-side hash table |

### 9.2 Draft KV Cache

The draft model needs its own KV cache for incremental decoding:

```
KV_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * sizeof(dtype)

Qwen2.5-0.5B: 2 * 24 * 2 * 128 * 2 = 24,576 bytes/token
  At 512 context: 12 MB per sequence
  At N=1: 12 MB total (negligible)

Self-draft (7 layers): 2 * 7 * 4 * 128 * 2 = 14,336 bytes/token
  At 512 context: 7 MB per sequence
```

### 9.3 Total Memory Budget (H100 80GB, Qwen2.5-7B)

```
Target model weights:        14.0 GB
Target KV cache (N=1, 512):   0.2 GB
Draft model (0.5B):           1.0 GB
Draft KV cache:               0.012 GB
CUDA kernels/scratch:         1.0 GB
Total:                       16.2 GB (20% of 80 GB)
```

Ample headroom. Even at N=32 with 2048 context, total memory is ~25 GB.

---

## 10. Implementation Plan: Fastest Path to Beating vLLM at N=1

### Phase 1: N-gram Draft (1-2 days) -- Target: 130-140 tok/s

**Why first:** Zero model loading, zero additional memory, zero training. Validates the full speculative decode pipeline end-to-end.

Implementation:
1. Add `NgramDraftModel` implementing `DraftModel` trait in `crates/rvllm-speculative/src/ngram.rs`
2. Build n-gram table from prompt tokens during prefill
3. Update table with generated tokens during decode
4. Look up most recent (n-1) tokens to predict next token
5. Fix verification to use incremental KV cache (not full re-prefill)
6. Wire into `generate_speculative()` as default when no draft model specified

Expected outcome: 1.3-1.4x speedup. Not enough to beat vLLM (170 tok/s), but validates the pipeline.

### Phase 2: Separate Draft Model -- Qwen2.5-0.5B (3-5 days) -- Target: 250+ tok/s

**Why second:** Highest projected speedup with existing infrastructure. Same tokenizer as target. No training needed.

Implementation:
1. Extend model loader to load a second model simultaneously
2. Create `SmallModelDraftRunner` that wraps a separate `GpuModelRunner` for the draft model
3. Implement incremental KV-cached decode for the draft model
4. The draft model gets its own CUDA stream for potential overlap with target model
5. Wire into `generate_speculative()` with `RVLLM_SPECULATIVE_DRAFT=/path/to/Qwen2.5-0.5B`

Critical subtask: The verification pass (`GpuTargetModel::forward_verify`) must be changed from full-context prefill to incremental KV-cached verification. The K draft tokens should be processed as a K-token "chunked prefill" appended to the existing KV cache, not re-processing the entire context.

### Phase 3: CUDA Graph Integration (2-3 days) -- Target: 280+ tok/s

Graph both the draft decode step and verification step:
1. Draft model decode graph at batch_size=1
2. Verification graph at token_count=(K+1) for standard K values (3, 5, 7)
3. Reduces kernel launch overhead from ~5+ ms per speculative step to ~0.05 ms

### Phase 4: Tree-Structured Verification (3-5 days) -- Target: 300+ tok/s

1. Generate top-2 candidates at each draft position
2. Flatten tree into sequence with custom attention mask
3. Modify FA3 kernel to accept custom mask (or use padding + causal mask tricks)
4. Expected boost: +15-30% over linear verification

### Phase 5: Medusa/EAGLE Heads (1-2 weeks) -- Target: 300-350 tok/s

Train Medusa or EAGLE heads for the Qwen2.5-7B model:
1. Implement Medusa head architecture in `crates/rvllm-model-runner/src/layers/medusa.rs`
2. Training script (PyTorch, distillation from target model)
3. Load trained heads alongside main model
4. Combine with tree-structured verification

### Phase 6: Adaptive K + Continuous Batching Integration (1 week)

1. Monitor online acceptance rate per sequence
2. Adjust K dynamically: increase K when alpha is high, decrease when low
3. Integrate speculative decode into the main engine loop (not a separate path)
4. Use speculation for low-N batches, standard decode for high-N

---

## 11. Expected End-State Performance

### After Phase 2 (most impactful single change):

| N | Current rvLLM | With 0.5B Draft (K=5) | vLLM 0.18 | rvLLM vs vLLM |
|---|---|---|---|---|
| 1 | 98 | **252** | 170 | **1.48x faster** |
| 4 | 548 | **1,008** | 665 | **1.52x faster** |
| 16 | 2,122 | **3,681** | 2,202 | **1.67x faster** |
| 32 | 3,957 | **6,293** | 4,585 | **1.37x faster** |
| 64 | 7,451 | 7,451 (no spec) | 7,888 | 0.94x |
| 128 | 12,312 | 12,312 (no spec) | 14,528 | 0.85x |

### After Phase 4 (tree verification):

| N | rvLLM + Tree Spec | vLLM 0.18 | rvLLM vs vLLM |
|---|---|---|---|
| 1 | **310** | 170 | **1.82x faster** |
| 4 | **1,240** | 665 | **1.86x faster** |

This transforms N=1 from rvLLM's weakest point (0.58x vLLM) to its strongest point (1.5-1.8x faster than vLLM).

---

## 12. Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| Acceptance rate lower than projected | Speedup reduced proportionally | N-gram fallback always available; adaptive K reduces wasted speculation |
| KV cache integration complexity | Phase 2 takes longer | Start with full-context verify (slower but correct), optimize incrementally |
| Draft model loading doubles startup time | UX regression | Lazy-load draft model after first request; async model loading |
| Memory pressure at high N with draft model | Fewer concurrent sequences | Disable speculation above N threshold; unload draft model dynamically |
| vLLM also adds speculative decoding | Competitive advantage disappears | vLLM already has spec-decode; our advantage is Rust-native efficiency of the scheduling loop |
| CUDA graph + spec-decode interaction bugs | Correctness regression | Extensive coherency testing at each phase; graph capture gated behind feature flag |

---

## 13. References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
2. Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318.
3. Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML.
4. Li, Y., et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML.
5. Fu, Y., et al. (2024). "Lookahead Decoding: Breaking the Sequential Dependency of LLM Decoding with Jacobi Iteration."
6. Miao, X., et al. (2024). "SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification." ASPLOS.
7. Elhoushi, M., et al. (2024). "Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding." arXiv:2404.16710.
8. Sun, Z., et al. (2024). "TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding."

---

### Critical Files for Implementation
- `crates/rvllm-speculative/src/engine.rs` -- Core speculative engine that needs incremental KV cache support for verification, adaptive K, and draft model pluggability
- `crates/rvllm-engine/src/gpu_engine.rs` -- GPU engine integration point (GpuTargetModel adapter, build_self_draft, generate_speculative) that must be reworked from full-context re-prefill to incremental KV-cached verification
- `crates/rvllm-speculative/src/draft.rs` -- DraftModel trait and DraftModelRunner placeholder that needs a real implementation wrapping a separate GpuModelRunner for the 0.5B draft model
- `crates/rvllm-model-runner/src/gpu_runner.rs` -- GPU forward pass orchestrator containing forward_partial (lines 813-894) that needs KV cache integration for incremental draft token generation
- `crates/rvllm-speculative/src/verification.rs` -- Verification algorithm (correct as-is for linear speculation) that will need extension for tree-structured verification with custom attention masks
