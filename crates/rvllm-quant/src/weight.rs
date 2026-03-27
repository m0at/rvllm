use crate::config::QuantConfig;

#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Option<Vec<f32>>,
    pub shape: (usize, usize),
    pub quant_config: QuantConfig,
}

impl QuantizedWeight {
    pub fn new(
        data: Vec<u8>,
        scales: Vec<f32>,
        zeros: Option<Vec<f32>>,
        shape: (usize, usize),
        quant_config: QuantConfig,
    ) -> Self {
        Self {
            data,
            scales,
            zeros,
            shape,
            quant_config,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn num_groups(&self) -> usize {
        if self.quant_config.group_size == 0 {
            return 1;
        }
        let cols = self.shape.1;
        (cols + self.quant_config.group_size - 1) / self.quant_config.group_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::QuantMethod;

    #[test]
    fn weight_basic() {
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(vec![0u8; 64], vec![1.0; 4], None, (4, 32), cfg);
        assert_eq!(w.num_elements(), 128);
        assert_eq!(w.num_groups(), 1);
    }

    #[test]
    fn weight_multiple_groups() {
        let cfg = QuantConfig::new(QuantMethod::GPTQ, 128, 4, true);
        let w = QuantizedWeight::new(
            vec![0u8; 256],
            vec![1.0; 8],
            Some(vec![0.0; 8]),
            (4, 256),
            cfg,
        );
        assert_eq!(w.num_elements(), 1024);
        assert_eq!(w.num_groups(), 2);
    }
}
