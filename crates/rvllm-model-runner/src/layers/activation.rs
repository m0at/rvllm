//! Activation functions: SiLU, GELU.
//! Optimized for Apple M5 NEON autovectorization via chunk-of-8 processing.

use half::f16;

use crate::bridge::{GpuBuffer, Result};

const CHUNK: usize = 8;
const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
const GELU_COEFF: f32 = 0.044715_f32;

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

#[inline(always)]
fn gelu_scalar(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// SiLU (Swish): x * sigmoid(x). Used in Llama / Mistral / Qwen2.
#[inline(always)]
pub fn silu(input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
    let data = &input.data;
    let len = data.len();
    let mut out = Vec::with_capacity(len);

    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i].to_f32();
        }
        for i in 0..CHUNK {
            tmp[i] = silu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            out.push(f16::from_f32(tmp[i]));
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let x = data[base + i].to_f32();
        out.push(f16::from_f32(silu_scalar(x)));
    }

    Ok(GpuBuffer::from_vec(out, input.shape.clone()))
}

/// SiLU in-place -- avoids allocation.
#[inline(always)]
pub fn silu_inplace(buf: &mut GpuBuffer<f16>) {
    let data = &mut buf.data;
    let len = data.len();
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i].to_f32();
        }
        for i in 0..CHUNK {
            tmp[i] = silu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            data[base + i] = f16::from_f32(tmp[i]);
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let x = data[base + i].to_f32();
        data[base + i] = f16::from_f32(silu_scalar(x));
    }
}

/// SiLU in-place on raw f32 slice -- no f16 conversion overhead.
#[inline(always)]
pub fn silu_inplace_f32(data: &mut [f32]) {
    let len = data.len();
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i];
        }
        for i in 0..CHUNK {
            tmp[i] = silu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            data[base + i] = tmp[i];
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        data[base + i] = silu_scalar(data[base + i]);
    }
}

/// GELU in-place on raw f32 slice -- no f16 conversion overhead.
#[inline(always)]
pub fn gelu_inplace_f32(data: &mut [f32]) {
    let len = data.len();
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i];
        }
        for i in 0..CHUNK {
            tmp[i] = gelu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            data[base + i] = tmp[i];
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        data[base + i] = gelu_scalar(data[base + i]);
    }
}

/// Fused silu(gate) * up -- single pass, saves one full traversal.
/// gate and up must have the same length.
#[inline(always)]
pub fn fused_silu_mul(gate: &[f16], up: &[f16]) -> Vec<f16> {
    debug_assert_eq!(gate.len(), up.len());
    let len = gate.len();
    let mut out = Vec::with_capacity(len);

    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut g = [0.0f32; CHUNK];
        let mut u = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            g[i] = gate[base + i].to_f32();
            u[i] = up[base + i].to_f32();
        }
        for i in 0..CHUNK {
            g[i] = silu_scalar(g[i]) * u[i];
        }
        for i in 0..CHUNK {
            out.push(f16::from_f32(g[i]));
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let gv = gate[base + i].to_f32();
        let uv = up[base + i].to_f32();
        out.push(f16::from_f32(silu_scalar(gv) * uv));
    }

    out
}

/// GELU (Gaussian Error Linear Unit), approximate tanh variant.
#[inline(always)]
pub fn gelu(input: &GpuBuffer<f16>) -> Result<GpuBuffer<f16>> {
    let data = &input.data;
    let len = data.len();
    let mut out = Vec::with_capacity(len);

    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i].to_f32();
        }
        for i in 0..CHUNK {
            tmp[i] = gelu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            out.push(f16::from_f32(tmp[i]));
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let x = data[base + i].to_f32();
        out.push(f16::from_f32(gelu_scalar(x)));
    }

    Ok(GpuBuffer::from_vec(out, input.shape.clone()))
}

/// GELU in-place -- avoids allocation.
#[inline(always)]
pub fn gelu_inplace(buf: &mut GpuBuffer<f16>) {
    let data = &mut buf.data;
    let len = data.len();
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let base = c * CHUNK;
        let mut tmp = [0.0f32; CHUNK];
        for i in 0..CHUNK {
            tmp[i] = data[base + i].to_f32();
        }
        for i in 0..CHUNK {
            tmp[i] = gelu_scalar(tmp[i]);
        }
        for i in 0..CHUNK {
            data[base + i] = f16::from_f32(tmp[i]);
        }
    }

    let base = chunks * CHUNK;
    for i in 0..remainder {
        let x = data[base + i].to_f32();
        data[base + i] = f16::from_f32(gelu_scalar(x));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(vals: &[f32]) -> GpuBuffer<f16> {
        GpuBuffer::from_vec(vals.iter().map(|&v| f16::from_f32(v)).collect(), vec![vals.len()])
    }

    #[test]
    fn silu_zero() {
        let buf = make_buf(&[0.0]);
        let out = silu(&buf).unwrap();
        assert!((out.data[0].to_f32()).abs() < 0.01);
    }

    #[test]
    fn silu_positive() {
        let buf = make_buf(&[2.0]);
        let out = silu(&buf).unwrap();
        // silu(2) = 2 * sigmoid(2) ~ 2 * 0.8808 ~ 1.7616
        assert!((out.data[0].to_f32() - 1.762).abs() < 0.05);
    }

    #[test]
    fn silu_negative() {
        let buf = make_buf(&[-2.0]);
        let out = silu(&buf).unwrap();
        // silu(-2) = -2 * sigmoid(-2) ~ -2 * 0.1192 ~ -0.2384
        assert!((out.data[0].to_f32() + 0.238).abs() < 0.05);
    }

    #[test]
    fn silu_inplace_matches() {
        let vals = [0.0, 2.0, -2.0, 1.0, -1.0, 3.0, -3.0, 0.5, -0.5];
        let buf = make_buf(&vals);
        let allocating = silu(&buf).unwrap();
        let mut inplace = buf;
        silu_inplace(&mut inplace);
        for (a, b) in allocating.data.iter().zip(inplace.data.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 0.001);
        }
    }

    #[test]
    fn fused_silu_mul_matches_separate() {
        let gate_vals: Vec<f16> = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 0.0, 1.5]
            .iter().map(|&v| f16::from_f32(v)).collect();
        let up_vals: Vec<f16> = [0.5f32, 1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, -0.5]
            .iter().map(|&v| f16::from_f32(v)).collect();
        let fused = fused_silu_mul(&gate_vals, &up_vals);
        for i in 0..gate_vals.len() {
            let g = gate_vals[i].to_f32();
            let u = up_vals[i].to_f32();
            let expected = silu_scalar(g) * u;
            assert!((fused[i].to_f32() - expected).abs() < 0.01,
                "mismatch at {}: got {} expected {}", i, fused[i].to_f32(), expected);
        }
    }

    #[test]
    fn gelu_zero() {
        let buf = make_buf(&[0.0]);
        let out = gelu(&buf).unwrap();
        assert!((out.data[0].to_f32()).abs() < 0.01);
    }

    #[test]
    fn gelu_positive() {
        let buf = make_buf(&[1.0]);
        let out = gelu(&buf).unwrap();
        // gelu(1) ~ 0.8412
        assert!((out.data[0].to_f32() - 0.841).abs() < 0.05);
    }

    #[test]
    fn gelu_negative() {
        let buf = make_buf(&[-1.0]);
        let out = gelu(&buf).unwrap();
        // gelu(-1) ~ -0.1588
        assert!((out.data[0].to_f32() + 0.159).abs() < 0.05);
    }

    #[test]
    fn gelu_inplace_matches() {
        let vals = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
        let buf = make_buf(&vals);
        let allocating = gelu(&buf).unwrap();
        let mut inplace = buf;
        gelu_inplace(&mut inplace);
        for (a, b) in allocating.data.iter().zip(inplace.data.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 0.001);
        }
    }

    #[test]
    fn silu_inplace_f32_matches() {
        let vals = [0.0f32, 2.0, -2.0, 1.0, -1.0, 3.0, -3.0, 0.5, -0.5];
        let expected: Vec<f32> = vals.iter().map(|&x| silu_scalar(x)).collect();
        let mut data = vals.to_vec();
        silu_inplace_f32(&mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn gelu_inplace_f32_matches() {
        let vals = [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
        let expected: Vec<f32> = vals.iter().map(|&x| gelu_scalar(x)).collect();
        let mut data = vals.to_vec();
        gelu_inplace_f32(&mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {} expected {}", a, b);
        }
    }
}
