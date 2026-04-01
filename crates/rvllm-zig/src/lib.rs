//! Zig SIMD acceleration for rvLLM.
//!
//! SIMD-optimized sampling ops (softmax, argmax, temperature scaling)
//! and weight conversion (bf16->f16, f32->f16).

extern "C" {
    fn rvz_softmax(logits: *const f32, out: *mut f32, n: usize);
    fn rvz_log_softmax(logits: *const f32, out: *mut f32, n: usize);
    fn rvz_argmax_f32(data: *const f32, n: usize) -> u32;
    fn rvz_max_f32(data: *const f32, n: usize) -> f32;
    fn rvz_temperature_scale(logits: *mut f32, n: usize, inv_temp: f32);
    fn rvz_bf16_to_f16(src: *const u16, dst: *mut u16, n: usize);
    fn rvz_f32_to_f16(src: *const f32, dst: *mut u16, n: usize);
}

#[inline]
pub fn softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert!(out.len() >= logits.len());
    unsafe { rvz_softmax(logits.as_ptr(), out.as_mut_ptr(), logits.len()) }
}

#[inline]
pub fn log_softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert!(out.len() >= logits.len());
    unsafe { rvz_log_softmax(logits.as_ptr(), out.as_mut_ptr(), logits.len()) }
}

#[inline]
pub fn argmax(logits: &[f32]) -> u32 {
    unsafe { rvz_argmax_f32(logits.as_ptr(), logits.len()) }
}

#[inline]
pub fn max_f32(data: &[f32]) -> f32 {
    unsafe { rvz_max_f32(data.as_ptr(), data.len()) }
}

#[inline]
pub fn temperature_scale(logits: &mut [f32], inv_temp: f32) {
    unsafe { rvz_temperature_scale(logits.as_mut_ptr(), logits.len(), inv_temp) }
}

#[inline]
pub fn bf16_to_f16(src: &[u16], dst: &mut [u16]) {
    debug_assert!(dst.len() >= src.len());
    unsafe { rvz_bf16_to_f16(src.as_ptr(), dst.as_mut_ptr(), src.len()) }
}

#[inline]
pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert!(dst.len() >= src.len());
    unsafe { rvz_f32_to_f16(src.as_ptr(), dst.as_mut_ptr(), src.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        softmax_into(&logits, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 2);
    }

    #[test]
    fn test_temperature_scale() {
        let mut logits = vec![1.0, 2.0, 3.0];
        temperature_scale(&mut logits, 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f16() {
        let src = vec![0x3F80u16, 0x4000u16];
        let mut dst = vec![0u16; 2];
        bf16_to_f16(&src, &mut dst);
        assert_eq!(dst[0], 0x3C00);
        assert_eq!(dst[1], 0x4000);
    }

    #[test]
    fn test_f32_to_f16() {
        let src = vec![1.0f32, 2.0f32];
        let mut dst = vec![0u16; 2];
        f32_to_f16(&src, &mut dst);
        assert_eq!(dst[0], 0x3C00);
        assert_eq!(dst[1], 0x4000);
    }
}
