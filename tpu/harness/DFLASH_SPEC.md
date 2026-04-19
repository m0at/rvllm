# DFlash Block Diffusion Speculative Decoding -- rvLLM TPU

Port of DFlash (arxiv 2602.06036) + DDTree (arxiv 2604.12989) to JAX/TPU v6e-4.
Target: Gemma 4 31B. Goal: tau=5-6, 160-200+ tok/s at B=1.

## Why DFlash over EAGLE-3

EAGLE-3 drafts K tokens SEQUENTIALLY (K forward passes through 1 layer).
DFlash drafts B tokens IN PARALLEL (1 forward pass through 5 layers).

| | EAGLE-3 (current) | DFlash (target) |
|---|---|---|
| Draft tokens per cycle | K=5 | B=16 |
| Draft forward passes | 5 sequential | 1 parallel |
| Drafter depth | 1 layer | 5 layers |
| Drafter params | 450M | ~1.2B |
| Published tau (8B model) | 3.0-3.4 | 6.5 |
| Published speedup | 1.7-2.0x | 4.9x |
| Verify step | identical | identical |

The verify step (31ms on our v6e-4) is THE SAME. The entire speedup comes from
higher tau through better draft quality + more tokens per cycle.

At our 31ms verify + ~3ms DFlash draft = ~34ms cycle:
- tau=5: 5/34ms = 147 tok/s (1.8x)
- tau=6: 6/34ms = 176 tok/s (2.2x)
- tau=8 (with DDTree): 8/34ms = 235 tok/s (2.9x)

## DFlash Drafter Architecture

### Model (for Gemma 4 31B, H=5376)

```
Input per cycle:
  block_output_ids: [B=16] int32
    position 0 = last accepted token (anchor)
    positions 1-15 = mask token ID (noise)

  target_features: [context_len, 5*H]
    hidden states from 5 uniformly-spaced target layers
    layer_ids = [1, 13, 25, 37, 49] for 60-layer Gemma 4

Architecture:
  embed = target.embed_tokens(block_output_ids)     # [16, H], frozen
  ctx = hidden_norm(fc(target_features))             # [ctx, H]
  
  For each of 5 DFlash layers:
    Q = q_proj(draft_hidden)                         # [16, q_dim]
    K = concat(k_proj(ctx), k_proj(draft_hidden))    # [ctx+16, kv_dim]
    V = concat(v_proj(ctx), v_proj(draft_hidden))    # [ctx+16, kv_dim]
    # NON-CAUSAL attention within draft block
    # Causal only for context (draft positions cannot see future context)
    attn_out = attention(Q, K, V, mask=bidirectional_block_mask)
    draft_hidden = MLP(attn_out)                     # standard pre-norm + SwiGLU
  
  logits = target.lm_head(norm(draft_hidden))        # [16, vocab], frozen LM head
  draft_tokens = argmax(logits[1:])                  # 15 draft tokens
```

### Attention Pattern: [L, L+B]

The mask for each draft position i (0-indexed within block):
- Can see ALL context positions [0, ctx) -- full causal context
- Can see ALL other draft positions [ctx, ctx+B) -- BIDIRECTIONAL
- Cannot see future context beyond what's in the KV cache

This is the key difference from EAGLE-3. The bidirectional attention within
the block lets each position use information from all other positions,
giving much higher draft quality than autoregressive.

### Parameter Count

| Component | Shape | Params |
|---|---|---|
| fc (feature fusion) | [5*5376, 5376] | 144.5M |
| hidden_norm | [5376] | 5.4K |
| 5x self_attn (Q/K/V/O) | 5 * 4 * [5376, 5376] | 578.8M |
| 5x MLP (gate/up/down) | 5 * 3 * [5376, 21504] | 1,037.5M |
| 5x layer norms | 5 * 2 * [5376] | 53.8K |
| output norm | [5376] | 5.4K |
| **Total (trained)** | | **~1,761M** |
| embed + lm_head (frozen) | shared with target | 0 (reuse) |

~1.76B params, ~3.5GB bf16. Fits in v6e-4 HBM alongside the 31B target.

Note: can reduce by using smaller intermediate_size (10752 instead of 21504 = half MLP).
That gives ~1.1B params, ~2.2GB. More practical for training speed.

### Reduced variant (recommended for first iteration)

| Component | Shape | Params |
|---|---|---|
| fc | [5*5376, 5376] | 144.5M |
| 5x self_attn | 5 * 4 * [5376, 5376] | 578.8M |
| 5x MLP (half intermediate) | 5 * 3 * [5376, 10752] | 518.7M |
| norms | | ~0.1M |
| **Total** | | **~1,242M** |

~1.24B params, ~2.5GB bf16. More reasonable for v6e-4 training.

## Training Recipe

### Data
- Generate responses with Gemma 4 31B on conversation prompts
- Paper uses 800K samples, 6 epochs
- Start with 50K samples (3072 tokens each), 3 epochs
- Self-distilled: run target model to generate responses

### Loss
Cross-entropy with exponential position decay:
```
L = sum(w_k * CE(logits[k], target_token[k]))
w_k = exp(-(k-1) / gamma)    gamma=7 for block_size=16
```
Earlier positions weighted more heavily (they're verified first).

### Training attention
During training, pack multiple blocks per sequence. Each block:
- 1 anchor token (ground truth) + 15 positions to predict
- Bidirectional attention within block
- Attend to target features
- No cross-block attention

### Hyperparameters
- AdamW, lr=6e-4 (10x higher than EAGLE-3's 5e-5)
- Gradient clipping: 1.0
- Cosine schedule with 4% warmup
- bf16 throughout

## Integration with Existing Code

### What stays the same
- Target model forward (verify step) -- identical to EAGLE-3
- Feature extraction (layers 1, 13, 25, 37, 49 instead of 2, 30, 59)
- Fused while_loop structure
- KV cache management
- On-device acceptance logic

### What changes
- Draft function: sequential chain -> single parallel pass
- Draft model: 1 layer -> 5 layers, non-causal attention
- Block size: K=5 -> B=16 (verify T=17 instead of T=6)
- Feature count: 3 layers -> 5 layers
- Training: TTT loss -> position-decay CE loss

### Verify step impact
With B=16, the verify step processes T=17 positions instead of T=6.
At our measured rates:
- T=6: 31ms (current)
- T=17: estimated ~40-45ms (KV reads and all-reduce scale with T)
  - Weight read: 9.5ms (unchanged)
  - KV reads: 11.5ms * 17/6 = ~32.6ms
  - All-reduce: 9.5ms * 17/6 = ~26.9ms
  
Wait, that gives ~69ms. Too slow.

### CRITICAL: T=17 verify may be too expensive on v6e-4

The verify cost scales linearly with T (positions). At T=17:
- 69ms cycle, tau=6: 6/69 = 87 tok/s (WORSE than baseline 80 tok/s)

We need block_size <= 10 for the verify cost to be manageable:
- T=11 (B=10): ~45ms, tau=5: 5/45 = 111 tok/s (1.4x)
- T=9 (B=8): ~38ms, tau=4.5: 4.5/38 = 118 tok/s (1.5x)
- T=7 (B=6): ~33ms, tau=3.5: 3.5/33 = 106 tok/s (1.3x)

DFlash's advantage on GPU (4.9x) comes from the GPU having MUCH faster
verify (compute-bound, not BW-bound). On TPU v6e-4 where verify is
BW-bound, the advantage shrinks dramatically.

### Revised block size recommendation: B=8

- T=9 verify: ~38ms
- Expected tau at B=8: ~4.0-4.5 (paper shows block_size=8 with gamma=4)
- tok/s: 4.5/38 = ~118 tok/s (1.5x baseline)

This is only marginally better than EAGLE-3 projected at tau=3.5 (113 tok/s).

## Honest Assessment

DFlash's massive speedups (4.9x) are on GPU where the verify step is fast
(~3-5ms). On TPU v6e-4, our verify step is 31ms and scales linearly with T.
This fundamentally limits the benefit of larger block sizes.

The DFlash advantage on our hardware is:
1. Better draft quality (tau per position) due to 5-layer depth + bidirectional
2. All-parallel drafting (no sequential penalty)
3. BUT: larger T means proportionally more expensive verify

Net effect on v6e-4: modest improvement over EAGLE-3, not the 4.9x seen on GPU.

If we can get int8 KV cache working (cutting verify from 31ms to 25ms), then:
- EAGLE-3 at tau=3.5: 3.5/27ms = 130 tok/s
- DFlash B=8 at tau=4.5: 4.5/30ms = 150 tok/s
- DFlash B=12 at tau=5.5: 5.5/35ms = 157 tok/s

The gap narrows to ~20% improvement from DFlash over EAGLE-3, not 2.4x.

## Implementation Plan

### Phase 1: DFlash drafter module (2 days)
- dflash_draft.py: 5-layer Transformer with non-causal [L, L+B] attention
- Feature extraction from 5 target layers
- Single-pass forward producing B draft tokens

### Phase 2: Training pipeline (2 days)
- dflash_train.py: position-decay CE loss
- Self-distilled data generation (run Gemma 4 on prompts)
- Training on v6e-4

### Phase 3: Inference integration (1 day)
- Replace EAGLE-3 draft_chain with DFlash single-pass
- Verify at T=B+1
- Fused while_loop

### Phase 4: DDTree (2 days, optional)
- Tree construction from per-position distributions
- Tree verification with ancestor-only attention mask
- KV cache compaction

Total: ~1 week for DFlash, +2 days for DDTree.

## Files
- tpu/harness/DFLASH_SPEC.md -- this file
- tpu/harness/dflash_draft.py -- drafter model (planned)
- tpu/harness/dflash_train.py -- training pipeline (planned)
- tpu/harness/dflash_infer.py -- inference with DFlash draft (planned)
