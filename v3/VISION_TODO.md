# Qwen 3.6 vision — known bugs (from vLLM source review 2026-05-04)

## Symptom
Both 224×224 magenta circle and 640×488 photo produce identical
"Logo mit stilisiertem S" answer → vision tokens carry no
image-discriminative signal.

## Root causes (verified against vllm-git/vllm/model_executor/models/qwen3_vl.py)

### Bug 1: Missing learned absolute position embedding
- vLLM has `pos_embed = nn.Embedding(num_position_embeddings=2304, hidden_size=1152)`
- Bilinearly interpolated to grid_h × grid_w via `fast_pos_embed_interpolate()`
- **Added** to `hidden_states` AFTER patch_embed, BEFORE encoder blocks
  (qwen3_vl.py:801: `hidden_states = hidden_states + pos_embeds`)
- Weight name in safetensors: `model.visual.pos_embed.weight` shape `[2304, 1152]` bf16
- `num_grid_per_side = 48` (= sqrt(2304))
- Our `forward_qwen_vision` skips this entirely → patches have no spatial
  identity until rotary inside attention, which is much weaker signal
  than additive learned pos_embed.

### Bug 2: Rotary applied to full head_dim instead of half
- vLLM: `get_rope(..., rope_parameters={"partial_rotary_factor": 0.5})`
- Means only first `head_dim/2 = 36` channels get rotated; remaining 36
  pass through unchanged.
- vLLM `cos_combined = cos[pos_ids].flatten(1)` produces
  `[seq_len, 2 * inv_freq_dim]` where `inv_freq_dim = rotary_dim/2 = 18`,
  so the per-token cos/sin table has width **36**, not 72.
- Layout inside the 36-wide table (NeoX-style):
  - dims  0..18 = cos/sin from row position (h_pos)
  - dims 18..36 = cos/sin from col position (w_pos)
- Our kernel + caller use `head_dim`-wide tables and rotate the whole
  head. This corrupts the upper half (channels 36..72) that should be
  identity.

## Fix plan (next iteration)

### Phase A: partial-rotary fix (smaller, local)
1. Rewrite `kernels/vit_rotary_2d_f16.cu`:
   - Add new param `rotary_dim` (= head_dim/2)
   - Block size = `rotary_dim/2` (= 18 for head_dim=72)
   - cos/sin tables shape `[seq_len, rotary_dim]`
   - Rotate only channels `[0, rotary_dim)`, leave `[rotary_dim, head_dim)`
     untouched.
   - Inside rotated region: `i ∈ [0, rotary_dim/2)` rotates
     against `j = i + rotary_dim/2`; tables encode row-freq lower,
     col-freq upper, same as before.
2. Caller `forward_qwen_vision` (qwen36_bring_up.rs ~3334):
   - `rope_dim_half = head_dim / 4` (NOT head_dim/2)  — that's
     `inv_freq_dim = 18`
   - Build cos/sin tables shape `[n_tokens, head_dim/2 = 36]`
   - Lower half (0..18) uses h_pos, upper half (18..36) uses w_pos
3. Update launch: block dim = head_dim/4 = 18

### Phase B: pos_embed addition (bigger, needs new kernel + loader)
1. Loader (`qwen36_load.rs`):
   - Add field `Qwen36Vision::pos_embed: F16Weight` shape `[2304, 1152]`
     bf16→f16 (`model.visual.pos_embed.weight`)
2. New kernel `kernels/vit_pos_embed_interp_f16.cu`:
   - Inputs:
     - pos_table: `[2304, 1152] f16`
     - grid_h, grid_w (post-merge: `H*spatial_merge_size`)
     - num_grid_per_side = 48
     - spatial_merge_size = 2
   - Output: `add` to `hidden_states[seq_len, 1152]` in place
   - For each output position (h, w) in `[0, grid_h) × [0, grid_w)`:
     1. Compute float coords in 48×48 grid:
        `gh = h * (num_grid_per_side - 1) / (grid_h - 1)` (or
        non-merged `grid_h / spatial_merge_size`?)
     2. Bilinear interpolate over 4 nbrs in pos_table
     3. Add to `hidden_states[token_idx]` where token_idx follows the
        same merge-aware ordering as pos_h/pos_w
   - Reference: `pos_embed_interpolate_native` in qwen3_vl.py:276
3. Hook in `forward_qwen_vision` between patch_embed and the 27-block
   loop:
   ```rust
   // Step 4.5 (new): add learned absolute position embedding
   launch_vit_pos_embed_interp(hidden_region, vision.pos_embed, grid_h, grid_w);
   ```

### Phase C: validation
- Re-run E2E with test_224.png and a real photo; expect content-aware
  answers (cat/circle/etc), not "Logo mit S" for everything.
- Sanity: vision probe L2 should change (currently 264.694).
- Compare vision-tower output (n_tokens, hidden) cosine similarity
  vs HF reference dump for one fixture image (Phase D, separate).

## Reference files
- vllm-git/vllm/model_executor/models/qwen3_vl.py:518 (transformer entry)
- vllm-git/vllm/model_executor/models/qwen3_vl.py:276 (`pos_embed_interpolate_native`)
- vllm-git/vllm/model_executor/models/qwen3_vl.py:651 (`rot_pos_emb`)
- transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- HF apply_rotary_pos_emb_vision: standard rotate_half on first
  rotary_dim channels only

## Status
- Phase A partial-rotary fix: not yet done (current code rotates full head)
- Phase B pos_embed: not yet done (entire feature missing)
- Phase 3b Gemma vision forward: not yet done
- Phase 5 numeric audit: blocked on Phase A+B
