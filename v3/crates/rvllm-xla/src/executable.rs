use rvllm_core::{ConfigError, Result, RvllmError};

use crate::ffi::PjrtElementType;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PjrtProgramFormat {
    Mlir,
    Hlo,
}

impl PjrtProgramFormat {
    pub const fn as_pjrt_format(self) -> &'static [u8] {
        match self {
            Self::Mlir => b"mlir",
            Self::Hlo => b"hlo",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PjrtTensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub nbytes: usize,
}

impl PjrtTensorSpec {
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<i64>>,
        dtype: PjrtElementType,
    ) -> Result<Self> {
        let shape = shape.into();
        let nbytes = tensor_nbytes(&shape, dtype)?;
        Ok(Self {
            name: name.into(),
            shape,
            dtype,
            nbytes,
        })
    }

    pub fn anonymous(shape: impl Into<Vec<i64>>, dtype: PjrtElementType) -> Result<Self> {
        Self::new("", shape, dtype)
    }

    pub fn from_parts(
        name: impl Into<String>,
        shape: impl Into<Vec<i64>>,
        dtype: PjrtElementType,
        nbytes: usize,
    ) -> Result<Self> {
        let spec = Self::new(name, shape, dtype)?;
        if spec.nbytes != nbytes {
            return Err(invalid_owned(
                "nbytes",
                format!(
                    "{} byte length mismatch: got {nbytes}, expected {}",
                    spec.display_name(),
                    spec.nbytes
                ),
            ));
        }
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        let expected = tensor_nbytes(&self.shape, self.dtype)?;
        if self.nbytes != expected {
            return Err(invalid_owned(
                "nbytes",
                format!(
                    "{} byte length mismatch: got {}, expected {expected}",
                    self.display_name(),
                    self.nbytes
                ),
            ));
        }
        Ok(())
    }

    pub fn validate_byte_len(&self, len: usize) -> Result<()> {
        self.validate()?;
        if len != self.nbytes {
            return Err(invalid_owned(
                "buffer",
                format!(
                    "{} host byte length mismatch: got {len}, expected {}",
                    self.display_name(),
                    self.nbytes
                ),
            ));
        }
        Ok(())
    }

    fn display_name(&self) -> &str {
        if self.name.is_empty() {
            "tensor"
        } else {
            &self.name
        }
    }
}

pub struct PjrtHostBuffer<'a> {
    pub data: &'a [u8],
    pub spec: PjrtTensorSpec,
}

impl<'a> PjrtHostBuffer<'a> {
    pub fn new(data: &'a [u8], spec: PjrtTensorSpec) -> Result<Self> {
        spec.validate_byte_len(data.len())?;
        Ok(Self { data, spec })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PjrtExecutableSignature {
    pub inputs: Vec<PjrtTensorSpec>,
    pub outputs: Vec<PjrtTensorSpec>,
}

impl PjrtExecutableSignature {
    pub fn new(inputs: Vec<PjrtTensorSpec>, outputs: Vec<PjrtTensorSpec>) -> Result<Self> {
        let signature = Self { inputs, outputs };
        signature.validate()?;
        Ok(signature)
    }

    pub fn validate(&self) -> Result<()> {
        for spec in self.inputs.iter().chain(self.outputs.iter()) {
            spec.validate()?;
        }
        Ok(())
    }

    pub fn validate_arguments(&self, actual: &[PjrtTensorSpec]) -> Result<()> {
        validate_argument_specs(&self.inputs, actual)
    }
}

pub fn tensor_nbytes(shape: &[i64], dtype: PjrtElementType) -> Result<usize> {
    let elem_bytes = dtype
        .byte_size()
        .ok_or_else(|| invalid("dtype", "invalid PJRT element type"))?;
    let mut elems = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(invalid("shape", "dimensions must be non-negative"));
        }
        elems = elems
            .checked_mul(dim as usize)
            .ok_or_else(|| invalid("shape", "element count overflow"))?;
    }
    elems
        .checked_mul(elem_bytes)
        .ok_or_else(|| invalid("nbytes", "byte length overflow"))
}

pub fn validate_argument_specs(
    expected: &[PjrtTensorSpec],
    actual: &[PjrtTensorSpec],
) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(invalid_owned(
            "arguments",
            format!(
                "got {} arguments, expected {}",
                actual.len(),
                expected.len()
            ),
        ));
    }
    for (idx, (want, got)) in expected.iter().zip(actual).enumerate() {
        want.validate()?;
        got.validate()?;
        if got.dtype != want.dtype {
            return Err(invalid_owned(
                "arguments",
                format!(
                    "argument {idx} {} dtype mismatch: got {:?}, expected {:?}",
                    want.display_name(),
                    got.dtype,
                    want.dtype
                ),
            ));
        }
        if got.shape != want.shape {
            return Err(invalid_owned(
                "arguments",
                format!(
                    "argument {idx} {} shape mismatch: got {:?}, expected {:?}",
                    want.display_name(),
                    got.shape,
                    want.shape
                ),
            ));
        }
        if got.nbytes != want.nbytes {
            return Err(invalid_owned(
                "arguments",
                format!(
                    "argument {idx} {} byte length mismatch: got {}, expected {}",
                    want.display_name(),
                    got.nbytes,
                    want.nbytes
                ),
            ));
        }
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    invalid_owned(field, reason.to_string())
}

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
        },
        "pjrt_executable",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_spec_computes_byte_lengths() {
        let spec = PjrtTensorSpec::new("logits", vec![2, 200_064], PjrtElementType::BF16).unwrap();
        assert_eq!(spec.nbytes, 800_256);
        assert_eq!(tensor_nbytes(&[], PjrtElementType::F32).unwrap(), 4);
        assert_eq!(PjrtProgramFormat::Mlir.as_pjrt_format(), b"mlir");
        assert_eq!(PjrtProgramFormat::Hlo.as_pjrt_format(), b"hlo");
    }

    #[test]
    fn tensor_spec_rejects_invalid_shapes_and_dtypes() {
        assert!(PjrtTensorSpec::new("bad", vec![-1], PjrtElementType::S32).is_err());
        assert!(PjrtTensorSpec::new("bad", vec![1], PjrtElementType::INVALID).is_err());
        assert!(PjrtTensorSpec::from_parts("bad", vec![2], PjrtElementType::S32, 4).is_err());
    }

    #[test]
    fn host_buffer_checks_byte_length() {
        let spec = PjrtTensorSpec::new("token_ids", vec![4], PjrtElementType::S32).unwrap();
        assert!(PjrtHostBuffer::new(&[0u8; 16], spec.clone()).is_ok());
        assert!(PjrtHostBuffer::new(&[0u8; 12], spec).is_err());
    }

    #[test]
    fn signature_argument_validation_checks_count_shape_dtype_and_bytes() {
        let want_token = PjrtTensorSpec::new("token_ids", vec![8], PjrtElementType::S32).unwrap();
        let want_kv = PjrtTensorSpec::new("kv_cache", vec![16], PjrtElementType::S8).unwrap();
        let sig = PjrtExecutableSignature::new(
            vec![want_token.clone(), want_kv.clone()],
            vec![PjrtTensorSpec::new("next_token", vec![8], PjrtElementType::S32).unwrap()],
        )
        .unwrap();

        assert!(sig
            .validate_arguments(&[want_token.clone(), want_kv.clone()])
            .is_ok());
        assert!(sig.validate_arguments(&[want_token.clone()]).is_err());
        assert!(sig
            .validate_arguments(&[
                PjrtTensorSpec::new("token_ids", vec![7], PjrtElementType::S32).unwrap(),
                want_kv.clone(),
            ])
            .is_err());
        assert!(sig
            .validate_arguments(&[
                PjrtTensorSpec::new("token_ids", vec![8], PjrtElementType::U32).unwrap(),
                want_kv,
            ])
            .is_err());
    }
}
