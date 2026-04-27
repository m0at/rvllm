//! Element types the runtime is allowed to see at tensor boundaries.
//!
//! No implicit conversions; every cast is an explicit kernel.

#[derive(
    Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum DType {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    I32,
    U32,
    U8,
    /// FP8 E4M3. Paired with a per-row or per-tensor `F32` scale at the
    /// tensor boundary (the scale tensor is separate).
    Fp8E4M3,
    /// FP8 E5M2 — not currently used in the decode path, reserved.
    Fp8E5M2,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn bytes(self) -> usize {
        match self {
            DType::F16 | DType::Bf16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 => 8,
            DType::I8 | DType::U8 | DType::Fp8E4M3 | DType::Fp8E5M2 => 1,
        }
    }

    /// True iff this dtype needs an external scale tensor (FP8 family).
    pub const fn needs_scale(self) -> bool {
        matches!(self, DType::Fp8E4M3 | DType::Fp8E5M2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(DType::F16.bytes(), 2);
        assert_eq!(DType::I8.bytes(), 1);
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::Fp8E4M3.bytes(), 1);
    }

    #[test]
    fn fp8_needs_scale() {
        assert!(DType::Fp8E4M3.needs_scale());
        assert!(!DType::F16.needs_scale());
    }
}
