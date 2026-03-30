use rvllm_core::prelude::*;

use crate::dequant;
use crate::gemm::gemm_quantized;
use crate::method::QuantMethod;
use crate::weight::QuantizedWeight;

/// Drop-in replacement for a linear layer that operates on quantized weights.
pub struct QuantizedLinear {
    weight: QuantizedWeight,
}

impl QuantizedLinear {
    pub fn new(weight: QuantizedWeight) -> Self {
        tracing::debug!(
            method = %weight.quant_config.method,
            shape = ?weight.shape,
            "created QuantizedLinear"
        );
        Self { weight }
    }

    /// Forward pass: fused dequantize + GEMM.
    /// input: flat f32 slice of length == weight.shape.1 (cols).
    #[inline]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        gemm_quantized(input, &self.weight)
    }

    /// Full dequantization of the weight matrix to f32.
    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let w = &self.weight;
        let empty_zeros;
        let zeros = match &w.zeros {
            Some(z) => z.as_slice(),
            None => {
                empty_zeros = vec![0.0f32; w.scales.len()];
                &empty_zeros
            }
        };

        match w.quant_config.method {
            QuantMethod::None => Err(LLMError::ModelError(
                "dequantize called on unquantized weight".into(),
            )),
            QuantMethod::GgufQ4_0 => Ok(dequant::dequantize_q4_0(&w.data, &w.scales, w.shape)),
            QuantMethod::GgufQ4KM => Ok(dequant::dequantize_q4_k_m(
                &w.data, &w.scales, zeros, w.shape,
            )),
            QuantMethod::GPTQ | QuantMethod::SqueezeLLM => Ok(dequant::dequantize_gptq(
                &w.data,
                &w.scales,
                zeros,
                w.quant_config.group_size,
                w.quant_config.bits,
                w.shape,
            )),
            QuantMethod::AWQ => Ok(dequant::dequantize_awq(
                &w.data,
                &w.scales,
                zeros,
                w.quant_config.group_size,
                w.shape,
            )),
            QuantMethod::FP8 => Ok(dequant::dequantize_fp8(&w.data, &w.scales, w.shape)),
            QuantMethod::MXFP4
            | QuantMethod::GgufQ5_0
            | QuantMethod::GgufQ5KM
            | QuantMethod::GgufQ8_0 => {
                Err(LLMError::ModelError(format!(
                    "dequantization not yet implemented for {}",
                    w.quant_config.method
                )))
            }
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.weight.shape
    }

    pub fn quant_method(&self) -> QuantMethod {
        self.weight.quant_config.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantConfig;
    use crate::dequant::q4::quantize_q4_0;

    #[test]
    fn linear_forward_q4_0() {
        let rows = 2;
        let cols = 32;
        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let (data, scales) = quantize_q4_0(&original, 32);
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(data, scales, None, (rows, cols), cfg);
        let layer = QuantizedLinear::new(w);

        let input = vec![1.0f32; 32];
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn linear_dequantize_q4_0() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let (data, scales) = quantize_q4_0(&original, 32);
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(data, scales, None, (1, 32), cfg);
        let layer = QuantizedLinear::new(w);

        let deq = layer.dequantize().unwrap();
        assert_eq!(deq.len(), 32);
        for (o, d) in original.iter().zip(deq.iter()) {
            assert!((o - d).abs() < 0.3);
        }
    }

    #[test]
    fn linear_shape() {
        let cfg = QuantConfig::new(QuantMethod::GgufQ4_0, 32, 4, false);
        let w = QuantizedWeight::new(vec![0u8; 16], vec![1.0], None, (1, 32), cfg);
        let layer = QuantizedLinear::new(w);
        assert_eq!(layer.shape(), (1, 32));
        assert_eq!(layer.quant_method(), QuantMethod::GgufQ4_0);
    }
}
