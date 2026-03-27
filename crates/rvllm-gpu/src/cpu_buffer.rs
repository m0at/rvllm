//! Host-side buffer for CPU/GPU data transfers.
//!
//! Mirrors the `GpuBuffer` interface but lives on the CPU side.
//! For true pinned (page-locked) memory that enables DMA transfers at
//! ~2x bandwidth, see `pinned_memory::PinnedBuffer`.
//! Under `mock-gpu` this is a plain `Vec<T>`.

use bytemuck::Pod;

use crate::Result;

pub struct CpuBuffer<T: Pod + Send> {
    data: Vec<T>,
}

impl<T: Pod + Send> CpuBuffer<T> {
    pub fn new(count: usize) -> Self {
        Self {
            data: vec![T::zeroed(); count],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.data.len() {
            return Err(crate::LLMError::MemoryError(format!(
                "CpuBuffer copy_from_host: source len {} != buffer len {}",
                src.len(),
                self.data.len()
            )));
        }
        self.data.copy_from_slice(src);
        Ok(())
    }

    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        Ok(self.data.clone())
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_roundtrip() {
        let mut buf = CpuBuffer::<f32>::new(4);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.size_bytes(), 16);

        let src = [1.0f32, 2.0, 3.0, 4.0];
        buf.copy_from_host(&src).unwrap();
        let out = buf.copy_to_host().unwrap();
        assert_eq!(out, &src);
    }

    #[test]
    fn size_mismatch_error() {
        let mut buf = CpuBuffer::<u8>::new(2);
        assert!(buf.copy_from_host(&[1, 2, 3]).is_err());
    }

    #[test]
    fn empty_buffer() {
        let buf = CpuBuffer::<f64>::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.size_bytes(), 0);
    }
}
