use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMethod {
    None,
    GPTQ,
    AWQ,
    SqueezeLLM,
    FP8,
    GgufQ4_0,
    GgufQ4KM,
    GgufQ5_0,
    GgufQ5KM,
    GgufQ8_0,
}

impl QuantMethod {
    pub fn bits(&self) -> u32 {
        match self {
            Self::None => 16,
            Self::GPTQ => 4,
            Self::AWQ => 4,
            Self::SqueezeLLM => 4,
            Self::FP8 => 8,
            Self::GgufQ4_0 => 4,
            Self::GgufQ4KM => 4,
            Self::GgufQ5_0 => 5,
            Self::GgufQ5KM => 5,
            Self::GgufQ8_0 => 8,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn group_size(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::GPTQ => Some(128),
            Self::AWQ => Some(128),
            Self::SqueezeLLM => Some(128),
            Self::FP8 => None,
            Self::GgufQ4_0 => Some(32),
            Self::GgufQ4KM => Some(32),
            Self::GgufQ5_0 => Some(32),
            Self::GgufQ5KM => Some(32),
            Self::GgufQ8_0 => Some(32),
        }
    }
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::GPTQ => write!(f, "gptq"),
            Self::AWQ => write!(f, "awq"),
            Self::SqueezeLLM => write!(f, "squeezellm"),
            Self::FP8 => write!(f, "fp8"),
            Self::GgufQ4_0 => write!(f, "gguf_q4_0"),
            Self::GgufQ4KM => write!(f, "gguf_q4_k_m"),
            Self::GgufQ5_0 => write!(f, "gguf_q5_0"),
            Self::GgufQ5KM => write!(f, "gguf_q5_k_m"),
            Self::GgufQ8_0 => write!(f, "gguf_q8_0"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_not_quantized() {
        assert!(!QuantMethod::None.is_quantized());
        assert_eq!(QuantMethod::None.bits(), 16);
        assert_eq!(QuantMethod::None.group_size(), None);
    }

    #[test]
    fn gptq_properties() {
        let m = QuantMethod::GPTQ;
        assert!(m.is_quantized());
        assert_eq!(m.bits(), 4);
        assert_eq!(m.group_size(), Some(128));
    }

    #[test]
    fn gguf_q4_0_properties() {
        let m = QuantMethod::GgufQ4_0;
        assert!(m.is_quantized());
        assert_eq!(m.bits(), 4);
        assert_eq!(m.group_size(), Some(32));
    }

    #[test]
    fn fp8_no_group_size() {
        let m = QuantMethod::FP8;
        assert!(m.is_quantized());
        assert_eq!(m.bits(), 8);
        assert_eq!(m.group_size(), None);
    }

    #[test]
    fn all_quantized_except_none() {
        let all = [
            QuantMethod::None,
            QuantMethod::GPTQ,
            QuantMethod::AWQ,
            QuantMethod::SqueezeLLM,
            QuantMethod::FP8,
            QuantMethod::GgufQ4_0,
            QuantMethod::GgufQ4KM,
            QuantMethod::GgufQ5_0,
            QuantMethod::GgufQ5KM,
            QuantMethod::GgufQ8_0,
        ];
        for m in &all {
            if matches!(m, QuantMethod::None) {
                assert!(!m.is_quantized());
            } else {
                assert!(m.is_quantized());
            }
        }
    }
}
