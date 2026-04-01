use std::fmt;

/// Data types supported for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U8,
    Q4_0,
    #[allow(non_camel_case_types)]
    Q4_K_M,
    /// FP8 E4M3 format (1 sign, 4 exponent, 3 mantissa bits). 1 byte per element.
    #[allow(non_camel_case_types)]
    F8_E4M3,
}

impl DType {
    /// Size in bytes of a single element for this dtype.
    /// Quantized types report the average bytes per element.
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::U8 => 1,
            DType::F8_E4M3 => 1,
            // Q4_0: 32 values packed in 18 bytes (16 nibbles + 2 byte scale) = 0.5625 bytes/elem
            // We report the block size; callers use total_bytes() on WeightTensor for exact sizing.
            DType::Q4_0 => 1,
            // Q4_K_M: similar quantization granularity
            DType::Q4_K_M => 1,
        }
    }

    /// Parse from a safetensors dtype string.
    pub fn from_safetensors_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I32" => Some(DType::I32),
            "U8" | "BOOL" => Some(DType::U8),
            "F8_E4M3" => Some(DType::F8_E4M3),
            _ => None,
        }
    }

    /// Parse from a GGUF type code.
    pub fn from_gguf_type(code: u32) -> Option<Self> {
        match code {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            // GGUF type 14 = Q4_K_M (approximate; real mapping depends on ggml version)
            14 => Some(DType::Q4_K_M),
            7 => Some(DType::I32),
            8 => Some(DType::U8),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::I32 => write!(f, "I32"),
            DType::U8 => write!(f, "U8"),
            DType::F8_E4M3 => write!(f, "F8_E4M3"),
            DType::Q4_0 => write!(f, "Q4_0"),
            DType::Q4_K_M => write!(f, "Q4_K_M"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_of_standard_types() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::U8.size_of(), 1);
    }

    #[test]
    fn from_safetensors_str_known() {
        assert_eq!(DType::from_safetensors_str("F32"), Some(DType::F32));
        assert_eq!(DType::from_safetensors_str("F16"), Some(DType::F16));
        assert_eq!(DType::from_safetensors_str("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors_str("UNKNOWN"), None);
    }

    #[test]
    fn from_gguf_type_known() {
        assert_eq!(DType::from_gguf_type(0), Some(DType::F32));
        assert_eq!(DType::from_gguf_type(1), Some(DType::F16));
        assert_eq!(DType::from_gguf_type(2), Some(DType::Q4_0));
        assert_eq!(DType::from_gguf_type(255), None);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", DType::Q4_K_M), "Q4_K_M");
        assert_eq!(format!("{}", DType::F16), "F16");
    }
}
