use crate::method::QuantMethod;

#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub method: QuantMethod,
    pub group_size: usize,
    pub bits: u32,
    pub has_zeros: bool,
}

impl QuantConfig {
    pub fn new(method: QuantMethod, group_size: usize, bits: u32, has_zeros: bool) -> Self {
        Self {
            method,
            group_size,
            bits,
            has_zeros,
        }
    }

    pub fn unquantized() -> Self {
        Self {
            method: QuantMethod::None,
            group_size: 0,
            bits: 16,
            has_zeros: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquantized_config() {
        let c = QuantConfig::unquantized();
        assert!(!c.method.is_quantized());
        assert_eq!(c.bits, 16);
        assert!(!c.has_zeros);
    }

    #[test]
    fn gptq_config() {
        let c = QuantConfig::new(QuantMethod::GPTQ, 128, 4, true);
        assert!(c.method.is_quantized());
        assert_eq!(c.group_size, 128);
        assert_eq!(c.bits, 4);
        assert!(c.has_zeros);
    }
}
