//! Element types the runtime is allowed to see at tensor boundaries.
//!
//! No implicit conversions; every cast is an explicit kernel.

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F16,
    Bf16,
    F32,
    F64,
    I32,
    U32,
    U8,
    /// FP8 E4M3. Paired with a per-row or per-tensor `F32` scale at the
    /// tensor boundary (the scale tensor is separate).
    Fp8E4M3,
    /// FP8 E5M2 — not currently used in the decode path, reserved.
    Fp8E5M2,
    /// 4-bit unsigned integer, packed two-per-byte. Cycle 39 (AWQ
    /// W4A16): low nibble = element 2i, high nibble = element 2i+1.
    /// Always paired with per-group F16/BF16 scales and (optionally)
    /// per-group U4 zero-points stored as separate tensors —
    /// compressed-tensors AWQ format. The scale tensor is always
    /// dimensioned `[out_features, in_features / group_size]`.
    U4Packed,
    /// 4-bit signed integer, packed two-per-byte. Same layout as
    /// U4Packed but interpreted as int4 two's-complement. Some AWQ
    /// checkpoints store weights as I4 with shifted zero-points;
    /// either form is viable since both are dequantized via
    /// `value = (packed_nibble - zero_point) * scale`.
    I4Packed,
}

impl DType {
    /// Size of one element in bytes. NOTE: U4Packed/I4Packed return 0
    /// because they are sub-byte — use [`Self::bits`] for sub-byte
    /// dtypes and compute storage as `(elements * bits + 7) / 8`.
    pub const fn bytes(self) -> usize {
        match self {
            DType::F16 | DType::Bf16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 => 8,
            DType::U8 | DType::Fp8E4M3 | DType::Fp8E5M2 => 1,
            DType::U4Packed | DType::I4Packed => 0,
        }
    }

    /// Bits per element. 4 for sub-byte packed dtypes, otherwise
    /// `bytes() * 8`. Caller computes storage from this.
    pub const fn bits(self) -> usize {
        match self {
            DType::U4Packed | DType::I4Packed => 4,
            _ => self.bytes() * 8,
        }
    }

    /// True iff this dtype is a sub-byte packed format. Storage
    /// computation needs to use `bits()` not `bytes()`.
    pub const fn is_sub_byte(self) -> bool {
        matches!(self, DType::U4Packed | DType::I4Packed)
    }

    /// True iff this dtype needs an external scale tensor (FP8 + INT4
    /// quantized families).
    pub const fn needs_scale(self) -> bool {
        matches!(
            self,
            DType::Fp8E4M3 | DType::Fp8E5M2 | DType::U4Packed | DType::I4Packed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(DType::F16.bytes(), 2);
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::Fp8E4M3.bytes(), 1);
    }

    #[test]
    fn fp8_needs_scale() {
        assert!(DType::Fp8E4M3.needs_scale());
        assert!(!DType::F16.needs_scale());
    }

    // === Cycle 39 sub-byte dtype tests ===

    #[test]
    fn u4_packed_is_sub_byte() {
        assert!(DType::U4Packed.is_sub_byte());
        assert!(DType::I4Packed.is_sub_byte());
        assert!(!DType::Fp8E4M3.is_sub_byte());
        assert!(!DType::F16.is_sub_byte());
    }

    #[test]
    fn u4_packed_bits_and_bytes() {
        assert_eq!(DType::U4Packed.bits(), 4);
        assert_eq!(DType::I4Packed.bits(), 4);
        // bytes() is 0 for sub-byte — caller must use bits() for storage
        assert_eq!(DType::U4Packed.bytes(), 0);
        assert_eq!(DType::I4Packed.bytes(), 0);
    }

    #[test]
    fn fp8_bits_consistent() {
        assert_eq!(DType::Fp8E4M3.bits(), 8);
        assert_eq!(DType::F16.bits(), 16);
        assert_eq!(DType::F32.bits(), 32);
    }

    #[test]
    fn u4_needs_scale() {
        assert!(DType::U4Packed.needs_scale());
        assert!(DType::I4Packed.needs_scale());
    }
}
