#!/usr/bin/env python3
"""EAGLE-3 draft head training for Gemma 4 on TPU v6e.

Online training: loads target model, captures features on-the-fly,
trains draft head with TTT (Training-Time Test) loss.

Usage:
    python3 eagle3_train.py \
        --model-dir /path/to/gemma-4-31B-it \
        --data-file /path/to/train.jsonl \
        --output-dir /path/to/eagle3-head \
        --max-seq 512 --epochs 3 --lr 5e-5 --batch-size 4
"""
import argparse, json, os, struct, sys, time, math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes

from eagle3_infer import (
    H, NH, INTER, VOCAB, NL, WINDOW, SOFTCAP_VAL, EPS,
    MAX_Q, MAX_KV, MAX_O, MAX_NORM_HD,
    S_Q, S_KV, S_HD, S_KVH, S_GQA,
    G_Q, G_KV, G_HD, G_KVH, G_GQA,
    LAYER_IS_GLOBAL, K_DRAFT, FEAT_LAYERS, DRAFT_INTER,
    rms_norm, head_norm, head_norm_noscale, rope,
    precompute_rope, int8_matmul,
    make_mesh, load_model, load_tokenizer,
    verify_one_layer_feats,
    init_random_draft, to_np_bf16,
)

# ── data loading ──

def load_data(path, tokenizer, max_seq, max_examples=None):
    sequences = []
    with open(path) as f:
        for line_num, line in enumerate(f):
            if max_examples and len(sequences) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith('{'):
                obj = json.loads(line)
                if 'text' in obj:
                    text = obj['text']
                elif 'conversations' in obj:
                    text = '\n'.join(t.get('value', '') for t in obj['conversations'])
                else:
                    text = str(obj)
            else:
                text = line

            tokens = [2] + tokenizer.encode(text).ids
            if len(tokens) < K_DRAFT + 2:
                continue
            tokens = tokens[:max_seq]
            sequences.append(np.array(tokens, dtype=np.int32))

    print(f"loaded {len(sequences)} sequences from {path}", file=sys.stderr)
    if sequences:
        lens = [len(s) for s in sequences]
        print(f"  lengths: min={min(lens)} max={max(lens)} mean={np.mean(lens):.0f}", file=sys.stderr)
    return sequences

def pad_sequence(tokens, max_seq):
    if len(tokens) >= max_seq:
        return tokens[:max_seq], min(len(tokens), max_seq)
    padded = np.zeros(max_seq, dtype=np.int32)
    padded[:len(tokens)] = tokens
    return padded, len(tokens)

# ── feature capture (target model prefill) ──

def prefill_features(tokens, embed, final_norm, weights, zero_kc, zero_vc,
                     cos_s, sin_s, cos_g, sin_g):
    T = tokens.shape[0]
    x = embed[tokens] * jnp.sqrt(jnp.float32(H))
    ctx = T
    fl = jnp.zeros_like(x)
    fm = jnp.zeros_like(x)
    fh = jnp.zeros_like(x)
    init = (x, jnp.int32(0), jnp.int32(ctx), cos_s, sin_s, cos_g, sin_g, fl, fm, fh)
    layer_xs = {**weights, 'kc': zero_kc, 'vc': zero_vc,
                'li': jnp.arange(NL, dtype=jnp.int32)}

    final, _ = jax.lax.scan(verify_one_layer_feats, init, layer_xs)
    return final[7], final[8], final[9]

# ── batched draft step (training) ──

def batched_draft_step(e, g, dw, dk, dv, draft_pos, embed, cos_s, sin_s):
    """Draft head forward for P parallel chains.

    e: [P, H]  token embeddings
    g: [P, H]  fused features or draft hidden states
    dk, dv: [P, K_DRAFT, S_KV]  per-chain draft KV caches
    draft_pos: scalar int  depth index (0..K-1)
    Returns: x [P, H], logits [P, VOCAB], dk, dv
    """
    P_ = e.shape[0]
    u = jnp.concatenate([e, g], axis=-1)
    u = u @ dw['fc_in_w'].T + dw['fc_in_b']

    residual = u
    h = rms_norm(u, dw['d_ln1'])

    q = h @ dw['d_qw'].T
    k = h @ dw['d_kw'].T
    v = h @ dw['d_vw'].T

    q = head_norm(q.reshape(P_, NH, S_HD), dw['d_qn'])
    k = head_norm(k.reshape(P_, S_KVH, S_HD), dw['d_kn'])
    v = head_norm_noscale(v.reshape(P_, S_KVH, S_HD))

    c = cos_s[draft_pos][None, None, :]
    s = sin_s[draft_pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)

    dk = dk.at[:, draft_pos, :].set(k.reshape(P_, S_KV))
    dv = dv.at[:, draft_pos, :].set(v.reshape(P_, S_KV))

    k_ctx = dk.reshape(P_, K_DRAFT, S_KVH, S_HD)
    v_ctx = dv.reshape(P_, K_DRAFT, S_KVH, S_HD)
    q_g = q.reshape(P_, S_KVH, S_GQA, S_HD)
    sc = jnp.einsum('pghd,ptgd->pght', q_g.astype(jnp.float32), k_ctx.astype(jnp.float32))

    t = jnp.arange(K_DRAFT)
    valid = t < (draft_pos + 1)
    sc = jnp.where(valid[None, None, None, :], sc, jnp.float32(-1e9))
    prob = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    attn_out = jnp.einsum('pght,ptgd->pghd', prob, v_ctx).reshape(P_, S_Q)

    x = residual + attn_out @ dw['d_ow'].T

    residual = x
    h = rms_norm(x, dw['d_ln2'])
    gate = h @ dw['d_gw'].T
    up = h @ dw['d_uw'].T
    h = jax.nn.gelu(gate, approximate=True) * up
    x = residual + h @ dw['d_dw'].T

    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)

    return x, logits, dk, dv

# ── TTT loss ──

def ttt_loss(dw, tokens, feat_low, feat_mid, feat_high, positions, embed, cos_s, sin_s):
    """Training-Time Test loss for one sequence.

    tokens:    [seq_len]
    feat_*:    [seq_len, H]
    positions: [P] starting positions to evaluate
    """
    P_ = positions.shape[0]

    fl = feat_low[positions]
    fm = feat_mid[positions]
    fh = feat_high[positions]
    concat = jnp.concatenate([fl, fm, fh], axis=-1)
    g_target = concat @ dw['fc_fuse_w'].T + dw['fc_fuse_b']

    dk = jnp.zeros((P_, K_DRAFT, S_KV), dtype=jnp.bfloat16)
    dv = jnp.zeros((P_, K_DRAFT, S_KV), dtype=jnp.bfloat16)

    total_loss = jnp.float32(0.0)
    x_prev = jnp.zeros((P_, H), dtype=jnp.bfloat16)

    for d in range(K_DRAFT):
        if d < 2:
            g = g_target
        else:
            g = jax.lax.stop_gradient(x_prev)

        tok_idx = tokens[positions + d]
        e = embed[tok_idx] * jnp.sqrt(jnp.float32(H))

        x, logits, dk, dv = batched_draft_step(e, g, dw, dk, dv, d, embed, cos_s, sin_s)

        target_toks = tokens[positions + d + 1]
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
        ce = -log_probs[jnp.arange(P_), target_toks]
        total_loss = total_loss + jnp.mean(ce)

        x_prev = x

    return total_loss / K_DRAFT

# ── optimizer (manual AdamW, no optax dependency) ──

def init_optimizer(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return m, v, jnp.int32(0)

def adamw_step(params, grads, m, v, step, lr, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    step = step + 1
    lr_t = lr * jnp.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
    m = jax.tree.map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: beta2 * v_ + (1 - beta2) * g ** 2, v, grads)
    params = jax.tree.map(
        lambda p, m_, v_: p - lr_t * (m_ / (jnp.sqrt(v_) + eps)) - lr * wd * p,
        params, m, v)
    return params, m, v, step

def clip_grad_norm(grads, max_norm=0.5):
    leaves = jax.tree.leaves(grads)
    total_sq = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in leaves)
    total_norm = jnp.sqrt(total_sq)
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), total_norm

# ── safetensors export ──

WEIGHT_NAME_MAP = {
    'fc_fuse_w': 'fc_fuse.weight',
    'fc_fuse_b': 'fc_fuse.bias',
    'fc_in_w': 'fc_in.weight',
    'fc_in_b': 'fc_in.bias',
    'd_ln1': 'draft_layer.input_layernorm.weight',
    'd_qw': 'draft_layer.self_attn.q_proj.weight',
    'd_kw': 'draft_layer.self_attn.k_proj.weight',
    'd_vw': 'draft_layer.self_attn.v_proj.weight',
    'd_ow': 'draft_layer.self_attn.o_proj.weight',
    'd_qn': 'draft_layer.self_attn.q_norm.weight',
    'd_kn': 'draft_layer.self_attn.k_norm.weight',
    'd_ln2': 'draft_layer.pre_feedforward_layernorm.weight',
    'd_gw': 'draft_layer.mlp.gate_proj.weight',
    'd_uw': 'draft_layer.mlp.up_proj.weight',
    'd_dw': 'draft_layer.mlp.down_proj.weight',
}

def write_safetensors(dw, path):
    header = {}
    offset = 0
    data_parts = []

    for internal_name, st_name in sorted(WEIGHT_NAME_MAP.items()):
        arr = np.array(dw[internal_name])
        if arr.dtype != ml_dtypes.bfloat16:
            arr = arr.astype(ml_dtypes.bfloat16)
        raw = arr.view(np.uint16).tobytes()
        header[st_name] = {
            'dtype': 'BF16',
            'shape': list(arr.shape),
            'data_offsets': [offset, offset + len(raw)]
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)

    total_bytes = 8 + len(header_json) + offset
    print(f"saved {path} ({total_bytes / 1e6:.1f} MB)", file=sys.stderr)

def save_checkpoint(dw, output_dir, epoch, step_num):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'draft_head_e{epoch}_s{step_num}.safetensors')
    write_safetensors(dw, path)

def save_final(dw, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'draft_head.safetensors')
    write_safetensors(dw, path)

# ── evaluation ──

def eval_loss(dw, val_sequences, embed, final_norm, weights, zero_kc, zero_vc,
              cos_s, sin_s, cos_g, sin_g, max_seq, num_positions):
    prefill_jit = jax.jit(prefill_features)
    total_loss = 0.0
    count = 0

    for seq in val_sequences[:50]:
        padded, seq_len = pad_sequence(seq, max_seq)
        if seq_len < K_DRAFT + 2:
            continue
        tokens = jnp.array(padded, dtype=jnp.int32)

        fl, fm, fh = prefill_jit(tokens, embed, final_norm, weights,
                                 zero_kc, zero_vc, cos_s, sin_s, cos_g, sin_g)

        max_start = max(seq_len - K_DRAFT - 1, 1)
        positions = jnp.minimum(jnp.arange(num_positions, dtype=jnp.int32), max_start - 1)

        loss = ttt_loss(dw, tokens, fl, fm, fh, positions, embed, cos_s, sin_s)
        total_loss += float(loss)
        count += 1

    return total_loss / max(count, 1)

# ── main training loop ──

def main():
    parser = argparse.ArgumentParser(description='EAGLE-3 draft head training')
    parser.add_argument('--model-dir', required=True, help='Target model directory')
    parser.add_argument('--data-file', required=True, help='Training data (JSONL)')
    parser.add_argument('--val-file', default=None, help='Validation data (JSONL)')
    parser.add_argument('--output-dir', required=True, help='Output directory for trained head')
    parser.add_argument('--max-seq', type=int, default=512, help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=1, help='Sequences per step')
    parser.add_argument('--positions-per-seq', type=int, default=64,
                        help='Random starting positions sampled per sequence')
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000, help='Checkpoint every N steps')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint dir')
    args = parser.parse_args()

    mesh = make_mesh()
    print(f"mesh: {mesh} ({len(jax.devices())} devices)", file=sys.stderr)

    tokenizer = load_tokenizer(args.model_dir)
    if tokenizer is None:
        print("ERROR: tokenizer.json not found", file=sys.stderr)
        sys.exit(1)

    print("loading training data...", file=sys.stderr)
    train_data = load_data(args.data_file, tokenizer, args.max_seq, args.max_examples)
    if not train_data:
        print("ERROR: no valid training sequences", file=sys.stderr)
        sys.exit(1)

    val_data = []
    if args.val_file:
        val_data = load_data(args.val_file, tokenizer, args.max_seq, max_examples=200)

    print("loading target model...", file=sys.stderr)
    embed, final_norm, weights, _ = load_model(args.model_dir, mesh, args.max_seq)

    cos_s_np, sin_s_np = precompute_rope(10000.0, S_HD, args.max_seq)
    cos_g_np, sin_g_np = precompute_rope(1000000.0, 128, args.max_seq)
    rope_sh = NamedSharding(mesh, P(None, None))
    cos_s = jax.device_put(jnp.array(cos_s_np), rope_sh)
    sin_s = jax.device_put(jnp.array(sin_s_np), rope_sh)
    cos_g = jax.device_put(jnp.array(cos_g_np), rope_sh)
    sin_g = jax.device_put(jnp.array(sin_g_np), rope_sh)

    kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
    zero_kc = jax.device_put(jnp.zeros((NL, args.max_seq, MAX_KV), dtype=jnp.bfloat16), kv_sh)
    zero_vc = jax.device_put(jnp.zeros((NL, args.max_seq, MAX_KV), dtype=jnp.bfloat16), kv_sh)

    if args.resume:
        from eagle3_infer import load_draft_weights
        dw = load_draft_weights(args.resume, mesh)
        print("resumed from checkpoint", file=sys.stderr)
    else:
        print("initializing random draft weights...", file=sys.stderr)
        dw = init_random_draft(mesh)

    m, v, opt_step = init_optimizer(dw)
    total_params = sum(x.size for x in jax.tree.leaves(dw))
    print(f"draft head: {total_params:,} params ({total_params * 2 / 1e6:.0f} MB bf16)", file=sys.stderr)

    prefill_jit = jax.jit(prefill_features)

    # All arrays use fixed shapes to avoid JIT recompilation.
    # Tokens always padded to max_seq; positions always positions_per_seq.
    print("compiling prefill...", file=sys.stderr, flush=True)
    t0 = time.time()
    test_padded, _ = pad_sequence(train_data[0], args.max_seq)
    test_tok = jnp.array(test_padded, dtype=jnp.int32)
    _fl, _fm, _fh = prefill_jit(test_tok, embed, final_norm, weights,
                                 zero_kc, zero_vc, cos_s, sin_s, cos_g, sin_g)
    _fl.block_until_ready()
    print(f"prefill compiled: {time.time()-t0:.1f}s, features shape: {_fl.shape}", file=sys.stderr)
    del _fl, _fm, _fh

    loss_grad_fn = jax.jit(jax.value_and_grad(
        lambda dw_, tokens_, fl_, fm_, fh_, positions_:
            ttt_loss(dw_, tokens_, fl_, fm_, fh_, positions_, embed, cos_s, sin_s)))

    print("compiling training step...", file=sys.stderr, flush=True)
    t0 = time.time()
    test_fl, test_fm, test_fh = prefill_jit(test_tok, embed, final_norm, weights,
                                             zero_kc, zero_vc, cos_s, sin_s, cos_g, sin_g)
    test_pos = jnp.zeros(args.positions_per_seq, dtype=jnp.int32)
    test_loss, test_grads = loss_grad_fn(dw, test_tok, test_fl, test_fm, test_fh, test_pos)
    test_loss.block_until_ready()
    print(f"training compiled: {time.time()-t0:.1f}s, initial loss: {float(test_loss):.3f}", file=sys.stderr)
    del test_fl, test_fm, test_fh, test_loss, test_grads

    update_fn = jax.jit(lambda dw_, grads_, m_, v_, step_, lr_:
                        adamw_step(dw_, grads_, m_, v_, step_, lr_))
    clip_fn = jax.jit(clip_grad_norm)

    # ── training loop ──

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        rng = np.random.default_rng(epoch)
        indices = rng.permutation(len(train_data))

        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch_start in range(0, len(train_data), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(train_data))
            batch_indices = indices[batch_start:batch_end]

            batch_loss_sum = 0.0
            batch_grad_sum = None

            for idx in batch_indices:
                seq = train_data[idx]
                padded, seq_len = pad_sequence(seq, args.max_seq)
                if seq_len < K_DRAFT + 2:
                    continue

                tokens = jnp.array(padded, dtype=jnp.int32)
                fl, fm, fh = prefill_jit(tokens, embed, final_norm, weights,
                                          zero_kc, zero_vc, cos_s, sin_s, cos_g, sin_g)

                max_start = max(seq_len - K_DRAFT - 1, 1)
                pos_key = jax.random.PRNGKey(global_step * 1000 + int(idx))
                positions = jax.random.randint(pos_key, shape=(args.positions_per_seq,),
                                               minval=0, maxval=max_start)

                loss, grads = loss_grad_fn(dw, tokens, fl, fm, fh, positions)

                batch_loss_sum += float(loss)
                if batch_grad_sum is None:
                    batch_grad_sum = grads
                else:
                    batch_grad_sum = jax.tree.map(lambda a, b: a + b, batch_grad_sum, grads)

            if batch_grad_sum is None:
                continue

            n_seqs = batch_end - batch_start
            avg_grads = jax.tree.map(lambda g: g / n_seqs, batch_grad_sum)
            avg_grads, grad_norm = clip_fn(avg_grads)

            lr_t = args.lr
            if global_step < args.warmup_steps:
                lr_t = args.lr * (global_step + 1) / args.warmup_steps

            dw, m, v, opt_step = update_fn(dw, avg_grads, m, v, opt_step, jnp.float32(lr_t))

            avg_loss = batch_loss_sum / n_seqs
            epoch_loss += avg_loss
            epoch_steps += 1
            global_step += 1

            if global_step % 10 == 0:
                elapsed = time.time() - t_epoch
                seqs_done = batch_end
                seqs_total = len(train_data)
                eta = elapsed / max(seqs_done, 1) * (seqs_total - seqs_done)
                print(f"  epoch {epoch} step {global_step} | loss {avg_loss:.3f} | "
                      f"grad_norm {float(grad_norm):.3f} | lr {lr_t:.2e} | "
                      f"{seqs_done}/{seqs_total} seqs | eta {eta:.0f}s",
                      file=sys.stderr)

            if global_step % args.save_every == 0:
                save_checkpoint(dw, args.output_dir, epoch, global_step)

        epoch_avg = epoch_loss / max(epoch_steps, 1)
        epoch_time = time.time() - t_epoch
        print(f"\nepoch {epoch} done: avg_loss={epoch_avg:.3f}, time={epoch_time:.0f}s, "
              f"steps={epoch_steps}", file=sys.stderr)

        if val_data:
            print("evaluating...", file=sys.stderr, flush=True)
            val_loss = eval_loss(dw, val_data, embed, final_norm, weights,
                                 zero_kc, zero_vc, cos_s, sin_s, cos_g, sin_g,
                                 args.max_seq, args.positions_per_seq)
            print(f"  val_loss={val_loss:.3f}", file=sys.stderr)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(dw, args.output_dir, epoch, global_step)
                print(f"  new best! saved checkpoint", file=sys.stderr)

        save_checkpoint(dw, args.output_dir, epoch, global_step)

    save_final(dw, args.output_dir)
    print(f"\ntraining complete. final weights: {args.output_dir}/draft_head.safetensors", file=sys.stderr)
    print(f"\nto run inference:\n  python3 eagle3_infer.py --model-dir {args.model_dir} "
          f"--draft-dir {args.output_dir} --max-tokens 256 --prompt 'Hello'", file=sys.stderr)

if __name__ == '__main__':
    main()
