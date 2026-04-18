use crate::device::XlaDeviceId;
use crate::Result;

#[cfg(feature = "tpu")]
use crate::ffi::PjrtElementType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlaDtype {
    F32,
    F16,
    BF16,
    U8,
    U16,
    U32,
    I32,
    I64,
}

impl XlaDtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::U16 => 2,
            Self::U8 => 1,
            Self::I64 => 8,
        }
    }

    #[cfg(feature = "tpu")]
    pub fn to_pjrt(&self) -> PjrtElementType {
        match self {
            Self::F32 => PjrtElementType::F32,
            Self::F16 => PjrtElementType::F16,
            Self::BF16 => PjrtElementType::BF16,
            Self::U8 => PjrtElementType::U8,
            Self::U16 => PjrtElementType::U16,
            Self::U32 => PjrtElementType::U32,
            Self::I32 => PjrtElementType::S32,
            Self::I64 => PjrtElementType::S64,
        }
    }
}

pub struct XlaBuffer {
    shape: Vec<i64>,
    dtype: XlaDtype,
    device: XlaDeviceId,
    #[cfg(feature = "tpu")]
    pjrt_handle: Option<crate::client::PjrtBufferHandle>,
}

impl XlaBuffer {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn dtype(&self) -> XlaDtype {
        self.dtype
    }

    pub fn device(&self) -> XlaDeviceId {
        self.device
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }

    #[cfg(feature = "tpu")]
    pub(crate) fn pjrt_handle(&self) -> Option<&crate::client::PjrtBufferHandle> {
        self.pjrt_handle.as_ref()
    }

    #[cfg(feature = "tpu")]
    pub(crate) fn from_pjrt_handle(
        handle: crate::client::PjrtBufferHandle,
        device_idx: usize,
    ) -> Self {
        Self {
            shape: vec![],
            dtype: XlaDtype::F32,
            device: XlaDeviceId(device_idx),
            pjrt_handle: Some(handle),
        }
    }

    #[cfg(feature = "tpu")]
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let handle = self.pjrt_handle.as_ref().ok_or_else(|| {
            crate::LLMError::GpuError("XlaBuffer has no PJRT handle".into())
        })?;
        handle.client().buffer_to_host(handle, dst)
    }

    #[cfg(not(feature = "tpu"))]
    pub fn copy_to_host(&self, _dst: &mut [u8]) -> Result<()> {
        Err(crate::LLMError::GpuError(
            "PJRT FFI not enabled -- build with --features tpu".into(),
        ))
    }

    #[cfg(feature = "tpu")]
    pub fn copy_from_host(
        client: &crate::client::PjrtClientHandle,
        src: &[u8],
        shape: &[i64],
        dtype: XlaDtype,
        device: XlaDeviceId,
    ) -> Result<Self> {
        let handle = client.buffer_from_host(src, shape, dtype.to_pjrt(), device.0)?;
        Ok(Self {
            shape: shape.to_vec(),
            dtype,
            device,
            pjrt_handle: Some(handle),
        })
    }

    #[cfg(not(feature = "tpu"))]
    pub fn copy_from_host(
        _src: &[u8],
        _shape: &[i64],
        _dtype: XlaDtype,
        _device: XlaDeviceId,
    ) -> Result<Self> {
        Err(crate::LLMError::GpuError(
            "PJRT FFI not enabled -- build with --features tpu".into(),
        ))
    }
}

unsafe impl Send for XlaBuffer {}
unsafe impl Sync for XlaBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(XlaDtype::F32.size_bytes(), 4);
        assert_eq!(XlaDtype::BF16.size_bytes(), 2);
        assert_eq!(XlaDtype::U8.size_bytes(), 1);
        assert_eq!(XlaDtype::I64.size_bytes(), 8);
    }

    #[test]
    fn buffer_size_calc() {
        let buf = XlaBuffer {
            shape: vec![128, 4096],
            dtype: XlaDtype::BF16,
            device: XlaDeviceId(0),
            #[cfg(feature = "tpu")]
            pjrt_handle: None,
        };
        assert_eq!(buf.num_elements(), 128 * 4096);
        assert_eq!(buf.size_bytes(), 128 * 4096 * 2);
    }
}
