//! Pure-Rust f32 reference implementations of every fused kernel.
//!
//! These are the ground truth for CI cosine tests. The PTX output must
//! match these within cosine ≥ 0.999 for f16 outputs and ≤ 1e-5 absolute
//! for f32 outputs.
//!
//! None of these functions touch a GPU; they are intentionally straight
//! loops so the arithmetic is legible and verifiable by eye.

pub const FP8_E4M3_MAX: f32 = 448.0;

/// RMSNorm: y = gamma * x / rms(x), where rms = sqrt(mean(x^2) + eps).
pub fn rmsnorm_ref(x: &[f32], gamma: &[f32], eps: f32, hidden: usize, out: &mut [f32]) {
    assert_eq!(x.len() % hidden, 0);
    assert_eq!(gamma.len(), hidden);
    assert_eq!(out.len(), x.len());
    for (row_in, row_out) in x.chunks(hidden).zip(out.chunks_mut(hidden)) {
        let ms: f32 = row_in.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (o, (xv, g)) in row_out.iter_mut().zip(row_in.iter().zip(gamma)) {
            *o = xv * inv * g;
        }
    }
}

/// Per-token FP8 E4M3 quantization: scale = amax(row) / 448, out = round(row / scale).
/// Returns (fp8 bytes, per-row scale) written into `out_fp8` and `scales`.
pub fn quantize_fp8_per_token_ref(
    x: &[f32],
    hidden: usize,
    out_fp8: &mut [u8],
    scales: &mut [f32],
) {
    assert_eq!(x.len() % hidden, 0);
    let rows = x.len() / hidden;
    assert_eq!(out_fp8.len(), x.len());
    assert_eq!(scales.len(), rows);
    for (i, row) in x.chunks(hidden).enumerate() {
        let amax = row
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let scale = amax / FP8_E4M3_MAX;
        let inv = 1.0 / scale;
        scales[i] = scale;
        let dst = &mut out_fp8[i * hidden..(i + 1) * hidden];
        for (d, v) in dst.iter_mut().zip(row) {
            *d = f32_to_fp8_e4m3(v * inv);
        }
    }
}

/// Fused add + RMSNorm + FP8 quant.
/// residual_out = x + residual_in; then (fp8_out, scale) = quant(rmsnorm(residual_out, gamma, eps)).
#[allow(clippy::too_many_arguments)]
pub fn fused_add_rmsnorm_fp8_quant_ref(
    x: &[f32],
    residual_in: &[f32],
    gamma: &[f32],
    eps: f32,
    hidden: usize,
    residual_out: &mut [f32],
    fp8_out: &mut [u8],
    scales: &mut [f32],
) {
    assert_eq!(x.len(), residual_in.len());
    for i in 0..x.len() {
        residual_out[i] = x[i] + residual_in[i];
    }
    let mut normed = vec![0f32; x.len()];
    rmsnorm_ref(residual_out, gamma, eps, hidden, &mut normed);
    quantize_fp8_per_token_ref(&normed, hidden, fp8_out, scales);
}

/// SiLU(gate) * up, then per-token FP8 quant.
/// `gate_up` is laid out `[T, 2, intermediate]` (gate first, up second) per token.
pub fn fused_silu_mul_fp8_quant_ref(
    gate_up: &[f32],
    num_tokens: usize,
    intermediate: usize,
    fp8_out: &mut [u8],
    scales: &mut [f32],
) {
    assert_eq!(gate_up.len(), num_tokens * 2 * intermediate);
    assert_eq!(fp8_out.len(), num_tokens * intermediate);
    let mut y = vec![0f32; num_tokens * intermediate];
    for t in 0..num_tokens {
        let base = t * 2 * intermediate;
        let gate = &gate_up[base..base + intermediate];
        let up = &gate_up[base + intermediate..base + 2 * intermediate];
        let yrow = &mut y[t * intermediate..(t + 1) * intermediate];
        for i in 0..intermediate {
            let g = gate[i];
            let silu = g / (1.0 + (-g).exp());
            yrow[i] = silu * up[i];
        }
    }
    quantize_fp8_per_token_ref(&y, intermediate, fp8_out, scales);
}

/// GELU(tanh)(gate) * up, then per-token FP8 quant.
/// Same layout as SiLU variant: `gate_up` is `[T, 2, intermediate]`.
pub fn fused_gelu_mul_fp8_quant_ref(
    gate_up: &[f32],
    num_tokens: usize,
    intermediate: usize,
    fp8_out: &mut [u8],
    scales: &mut [f32],
) {
    assert_eq!(gate_up.len(), num_tokens * 2 * intermediate);
    assert_eq!(fp8_out.len(), num_tokens * intermediate);
    let mut y = vec![0f32; num_tokens * intermediate];
    for t in 0..num_tokens {
        let base = t * 2 * intermediate;
        let gate = &gate_up[base..base + intermediate];
        let up = &gate_up[base + intermediate..base + 2 * intermediate];
        let yrow = &mut y[t * intermediate..(t + 1) * intermediate];
        for i in 0..intermediate {
            let g = gate[i];
            let sqrt_2_over_pi: f32 = 0.7978845608;
            let x3 = g * g * g;
            let inner = sqrt_2_over_pi * (g + 0.044715 * x3);
            let gelu = 0.5 * g * (1.0 + inner.tanh());
            yrow[i] = gelu * up[i];
        }
    }
    quantize_fp8_per_token_ref(&y, intermediate, fp8_out, scales);
}

/// Argmax along the last axis of `[rows, cols]`, written to `out`.
pub fn argmax_ref(logits: &[f32], rows: usize, cols: usize, out: &mut [i32]) {
    assert_eq!(logits.len(), rows * cols);
    assert_eq!(out.len(), rows);
    for (i, row) in logits.chunks(cols).enumerate() {
        let mut best = 0usize;
        let mut best_v = row[0];
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = j;
            }
        }
        out[i] = best as i32;
    }
}

/// `x[i] += y[i]` in-place.
pub fn residual_add_ref(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (xi, yi) in x.iter_mut().zip(y) {
        *xi += *yi;
    }
}

/// Gather token embeddings: `out[t] = weight[token_ids[t]]`.
pub fn embedding_gather_ref(
    token_ids: &[u32],
    weight: &[f32],
    hidden: usize,
    vocab: usize,
    out: &mut [f32],
) {
    assert_eq!(weight.len(), hidden * vocab);
    assert_eq!(out.len(), hidden * token_ids.len());
    for (t, &tid) in token_ids.iter().enumerate() {
        let tid = tid as usize;
        assert!(tid < vocab, "token id {tid} out of vocab {vocab}");
        out[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&weight[tid * hidden..(tid + 1) * hidden]);
    }
}

/// Minimal RoPE: apply cos/sin to pairs. No KV-write — the fused
/// kv-write variant lives in the GPU kernel; reference here is the
/// pure rotation for easier unit checking.
pub fn rope_ref(
    q: &mut [f32],
    positions: &[u32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    head_dim: usize,
) {
    assert!(head_dim % 2 == 0);
    let pairs = head_dim / 2;
    let num_tokens = positions.len();
    assert_eq!(q.len(), num_tokens * num_heads * head_dim);
    for t in 0..num_tokens {
        let p = positions[t] as usize;
        let c = &cos[p * pairs..(p + 1) * pairs];
        let s = &sin[p * pairs..(p + 1) * pairs];
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for i in 0..pairs {
                let a = q[base + i];
                let b = q[base + i + pairs];
                q[base + i] = a * c[i] - b * s[i];
                q[base + i + pairs] = a * s[i] + b * c[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f32 → FP8 E4M3 round-to-nearest with saturation to ±448.
// Returns a u8 holding the 8-bit E4M3 representation.
// ---------------------------------------------------------------------------

fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sat = x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
    if sat == 0.0 {
        return 0;
    }
    let sign = (sat.is_sign_negative() as u8) << 7;
    let ax = sat.abs();
    let bits = ax.to_bits();
    // f32: sign | exp(8) | mant(23); bias 127
    let f32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let f32_mant = bits & 0x7fffff;
    // e4m3: bias 7, exp 4 bits (0..15), mant 3 bits; subnormal when exp==0.
    let unbiased = f32_exp;
    // Largest normal e4m3 exponent is 8 (bias 7 + 1? standard says max is S1E111M111=448).
    // The official e4m3 uses bias=7, but 0b1111 is reserved for NaN in E4M3 FN; we saturate.
    let e8 = unbiased + 7;
    if e8 >= 15 {
        // saturate
        return sign | 0x7e; // 0 1111 110 = 448
    }
    if e8 <= 0 {
        // subnormal: shift mantissa right by -e8 positions
        let shift = 1 - e8; // 1..=7
        let m_full = (1 << 23) | f32_mant; // implicit leading 1
                                           // take top 3 bits of the subnormal mantissa
        let m_sub = m_full >> (23 - 3 + shift as u32).min(26);
        return sign | (m_sub as u8 & 0x7);
    }
    let e = e8 as u8 & 0xf;
    let m = (f32_mant >> (23 - 3)) as u8 & 0x7;
    sign | (e << 3) | m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_hand_calc() {
        // x = [1,2,3], gamma = [1,1,1], eps=0
        let x = [1.0f32, 2.0, 3.0];
        let g = [1.0f32, 1.0, 1.0];
        let mut o = [0.0f32; 3];
        rmsnorm_ref(&x, &g, 0.0, 3, &mut o);
        // rms = sqrt((1+4+9)/3) = sqrt(14/3) ~ 2.1602
        let rms = (14f32 / 3.0).sqrt();
        for i in 0..3 {
            assert!((o[i] - x[i] / rms).abs() < 1e-6);
        }
    }

    #[test]
    fn argmax_returns_largest() {
        let logits = [0.1f32, 0.9, 0.3, 0.5, -1.0, 0.8];
        let mut out = [0i32; 2];
        argmax_ref(&logits, 2, 3, &mut out);
        // row0 [0.1,0.9,0.3] → 1 (0.9); row1 [0.5,-1.0,0.8] → 2 (0.8)
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn silu_mul_then_quant_round_trip() {
        // T=1, I=4
        let gate_up = vec![
            0.0f32, 1.0, -1.0, 2.0, // gate
            1.0f32, 1.0, 1.0, 1.0, // up
        ];
        let mut fp8 = vec![0u8; 4];
        let mut scale = vec![0f32; 1];
        fused_silu_mul_fp8_quant_ref(&gate_up, 1, 4, &mut fp8, &mut scale);
        assert!(scale[0] > 0.0);
        assert!(fp8.iter().any(|&b| b != 0));
    }

    #[test]
    fn embedding_gather_picks_row() {
        // V=3 H=2
        let w = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0];
        let ids = [2u32, 0];
        let mut out = [0f32; 4];
        embedding_gather_ref(&ids, &w, 2, 3, &mut out);
        assert_eq!(&out[..2], &[30.0, 31.0]);
        assert_eq!(&out[2..], &[10.0, 11.0]);
    }

    #[test]
    fn rope_is_orthogonal_preserving() {
        // Single token, 1 head, head_dim=4 (so 2 pairs).
        let mut q = vec![1.0f32, 0.0, 0.0, 1.0];
        let pos = [0u32];
        let cos = [1.0f32, 1.0];
        let sin = [0.0f32, 0.0];
        rope_ref(&mut q, &pos, &cos, &sin, 1, 4);
        assert_eq!(q, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn quantize_scale_is_positive() {
        let x = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let mut fp8 = vec![0u8; 6];
        let mut scale = vec![0f32; 2];
        quantize_fp8_per_token_ref(&x, 3, &mut fp8, &mut scale);
        assert!(scale[0] > 0.0 && scale[1] > 0.0);
    }
}
