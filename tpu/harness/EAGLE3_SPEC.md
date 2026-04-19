# EAGLE-3 Speculative Decoding -- rvLLM TPU Add-on

Target: Gemma 4 31B on v6e-4 (TP=4). B=1 only.
Goal: 200-280 tok/s single-user (up from 80 tok/s baseline).

## Confirmed Model Values

| Param | Value |
|---|---|
| d_model (H) | 5376 |
| num_layers | 60 |
| vocab_size | 262144 |
| intermediate_size | 21504 |
| Layer pattern | sliding default, global at (i+1)%6==0: indices 5,11,17,23,29,35,41,47,53,59 |
| Sliding | head_dim=256, num_kv_heads=16, q_dim=8192 |
| Global | head_dim=512, num_kv_heads=4, q_dim=16384, k_eq_v=true |
| LM head | tied to embed_tokens |
| Softcap | 30*tanh(x/30) |

## Feature Capture Layers

Segmented scan splits 60 layers into 3 segments:

| Feature | Layer | Segment | Type |
|---|---|---|---|
| Low (syntactic) | 2 | layers[0:3] | sliding |
| Mid (relational) | 30 | layers[3:31] | sliding |
| High (semantic) | 59 | layers[31:60] | global |

## Draft Head Architecture (~450M params, ~900MB bf16)

```
FC_fuse:    Linear(3*5376 -> 5376) + bias     87M
FC_in:      Linear(2*5376 -> 5376) + bias     58M
DraftLayer: 1 transformer block (sliding-style dims)
  Q_proj:   [8192, 5376]                      44M
  K_proj:   [4096, 5376]                      22M
  V_proj:   [4096, 5376]                      22M
  O_proj:   [5376, 8192]                      44M
  Gate:     [10752, 5376]                     58M
  Up:       [10752, 5376]                     58M
  Down:     [5376, 10752]                     58M
  QK-norm, 2x RMSNorm, RoPE(theta=10k, dim=256)
LM head:    reuse target embed_tokens (tied)
```

On v6e-4: REPLICATED across all 4 chips (~3.6GB total, no TP collectives).

## Draft-Verify Cycle (Greedy)

```
1. Draft:  K=5 sequential steps through draft head (~2ms)
   Input per step: (embedding of last token, fused feature g)
   g_0 = FC_fuse(feat_low, feat_mid, feat_high) from last verified position
   g_{k+1} = draft head hidden state (self-approximation, TTT trained)

2. Verify: K+1=6 positions through target in ONE pass (~14ms)
   Input: [last_token, d0, d1, d2, d3, d4] at positions [pos..pos+5]
   Same weight read as single-step decode; 6 positions nearly free at B=1
   Captures features at layers 2/30/59 for all 6 positions

3. Accept: greedy prefix match
   Compare d_i vs argmax(target_logits[i]) for i=0..K-1
   First mismatch at j: accept d_0..d_{j-1}, correction = argmax(logits[j])
   All match: bonus = argmax(logits[K])
   Output: min 1, max K+1 = 6 tokens per cycle

4. KV rollback: implicit
   Next cycle overwrites garbage entries; causal mask prevents reading them
   No explicit rollback needed
```

## Expected Performance

| tau | tok/s | speedup |
|---|---|---|
| 2.0 | 143 | 1.8x |
| 3.0 | 214 | 2.7x |
| 3.5 | 250 | 3.1x |
| 4.0 | 286 | 3.6x |
| 5.0 | 357 | 4.5x |

Realistic tau on chat after training: 3.5-4.5 -> 250-320 tok/s.

## Files

- `tpu/harness/EAGLE3_SPEC.md` -- this file
- `tpu/harness/eagle3_infer.py` -- inference module (self-contained)
- `tpu/harness/eagle3_train.py` -- training script (phase 2)

## Usage

```bash
# Pipeline test with random draft weights (~0% acceptance, validates wiring):
python3 eagle3_infer.py --model-dir /path/to/gemma-4-31B-it \
    --max-tokens 32 --prompt "Hello" --random-draft

# Train draft head (online: loads target model, captures features on-the-fly):
python3 eagle3_train.py \
    --model-dir /path/to/gemma-4-31B-it \
    --data-file /path/to/train.jsonl \
    --output-dir /path/to/eagle3-head \
    --max-seq 512 --epochs 3 --lr 5e-5 --batch-size 4

# With trained draft head:
python3 eagle3_infer.py --model-dir /path/to/gemma-4-31B-it \
    --draft-dir /path/to/eagle3-head --max-tokens 256 --prompt "Hello"
```

## Training (eagle3_train.py)

Online approach: target model stays loaded, features captured on-the-fly per sequence.
No disk-based feature caching needed.

TTT (Training-Time Test) loss:
- 5-depth draft chain per starting position
- Depths 0-1: use target features (g from fuse(fl, fm, fh) at starting position)
- Depths 2-4: use draft hidden state (self-approximation)
- Embeddings always use ground-truth tokens
- Cross-entropy at each depth against next ground-truth token

Training data format: JSONL, each line `{"text": "..."}` or `{"conversations": [...]}`

Hyperparameters: AdamW lr=5e-5, grad clip=0.5, 1000-step linear warmup, 2-4 epochs
Compute: ~2-6 hours on v6e-4 with 100K sequences at max_seq=512
Checkpoints saved every N steps + end of each epoch
Final weights exported as safetensors for eagle3_infer.py

If tau < 3.5 after training: scale to 500K sequences or add domain-specific data

## v6e-1 Note

31B int8 = ~31GB > 32GB HBM. v6e-1 only works with smaller targets (7B class).
The EAGLE-3 code handles TP=1 automatically via mesh detection.
