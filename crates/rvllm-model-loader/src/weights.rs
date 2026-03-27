use std::collections::HashMap;
use std::marker::PhantomData;

use rvllm_core::error::{LLMError, Result};

use crate::dtype::DType;

/// Opaque GPU buffer holding raw bytes.
/// In production this wraps a real device allocation from rvllm-gpu.
/// Here we use a Vec<u8> stand-in that the rest of the crate programs against.
#[derive(Debug, Clone)]
pub struct GpuBuffer<T = u8> {
    data: Vec<u8>,
    _marker: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            data: bytes,
            _marker: PhantomData,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Trait for allocating GPU buffers. Downstream crates provide real implementations.
pub trait GpuAllocator: Send + Sync {
    fn allocate(&self, size_bytes: usize) -> Result<GpuBuffer<u8>>;
    fn upload(&self, data: &[u8]) -> Result<GpuBuffer<u8>>;
}

/// A single named weight tensor stored on the GPU.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    name: String,
    shape: Vec<usize>,
    dtype: DType,
    data: GpuBuffer<u8>,
}

impl WeightTensor {
    pub fn new(name: String, shape: Vec<usize>, dtype: DType, data: GpuBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            dtype,
            data,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn data(&self) -> &GpuBuffer<u8> {
        &self.data
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes of the tensor data.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Collection of model weights keyed by name.
#[derive(Debug)]
pub struct ModelWeights {
    weights: HashMap<String, WeightTensor>,
}

impl ModelWeights {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self { weights }
    }

    pub fn insert(&mut self, tensor: WeightTensor) {
        self.weights.insert(tensor.name.clone(), tensor);
    }

    /// Look up a weight by name.
    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.weights.get(name)
    }

    /// Look up a weight and reinterpret its buffer as typed.
    /// Returns an error if the name is missing or the dtype doesn't match.
    pub fn get_typed<T: TypedWeight>(&self, name: &str) -> Result<&GpuBuffer<u8>> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| LLMError::ModelError(format!("weight not found: {}", name)))?;
        if tensor.dtype != T::DTYPE {
            return Err(LLMError::ModelError(format!(
                "dtype mismatch for {}: expected {}, got {}",
                name,
                T::DTYPE,
                tensor.dtype
            )));
        }
        Ok(&tensor.data)
    }

    /// Iterate over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Number of weight tensors loaded.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker trait associating a Rust type with a DType for typed access.
pub trait TypedWeight {
    const DTYPE: DType;
}

impl TypedWeight for f32 {
    const DTYPE: DType = DType::F32;
}

impl TypedWeight for u8 {
    const DTYPE: DType = DType::U8;
}

impl TypedWeight for i32 {
    const DTYPE: DType = DType::I32;
}

/// Mock allocator for testing.
#[derive(Debug)]
pub struct MockGpuAllocator;

impl GpuAllocator for MockGpuAllocator {
    fn allocate(&self, size_bytes: usize) -> Result<GpuBuffer<u8>> {
        Ok(GpuBuffer::from_bytes(vec![0u8; size_bytes]))
    }

    fn upload(&self, data: &[u8]) -> Result<GpuBuffer<u8>> {
        Ok(GpuBuffer::from_bytes(data.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_tensor_basics() {
        let buf = GpuBuffer::from_bytes(vec![0u8; 16]);
        let wt = WeightTensor::new("layer.0.weight".into(), vec![2, 2], DType::F32, buf);
        assert_eq!(wt.name(), "layer.0.weight");
        assert_eq!(wt.shape(), &[2, 2]);
        assert_eq!(wt.dtype(), DType::F32);
        assert_eq!(wt.numel(), 4);
        assert_eq!(wt.size_bytes(), 16);
    }

    #[test]
    fn model_weights_insert_get() {
        let mut mw = ModelWeights::new();
        let buf = GpuBuffer::from_bytes(vec![0u8; 8]);
        let wt = WeightTensor::new("test".into(), vec![4], DType::F16, buf);
        mw.insert(wt);
        assert_eq!(mw.num_weights(), 1);
        assert!(mw.get("test").is_some());
        assert!(mw.get("missing").is_none());
    }

    #[test]
    fn model_weights_names() {
        let mut mw = ModelWeights::new();
        for i in 0..3 {
            let buf = GpuBuffer::from_bytes(vec![0u8; 4]);
            mw.insert(WeightTensor::new(
                format!("w{}", i),
                vec![1],
                DType::F32,
                buf,
            ));
        }
        let mut names: Vec<&str> = mw.names().collect();
        names.sort();
        assert_eq!(names, vec!["w0", "w1", "w2"]);
    }

    #[test]
    fn get_typed_success() {
        let mut mw = ModelWeights::new();
        let buf = GpuBuffer::from_bytes(vec![0u8; 4]);
        mw.insert(WeightTensor::new("a".into(), vec![1], DType::F32, buf));
        let result = mw.get_typed::<f32>("a");
        assert!(result.is_ok());
    }

    #[test]
    fn get_typed_wrong_dtype() {
        let mut mw = ModelWeights::new();
        let buf = GpuBuffer::from_bytes(vec![0u8; 4]);
        mw.insert(WeightTensor::new("a".into(), vec![1], DType::F16, buf));
        let result = mw.get_typed::<f32>("a");
        assert!(result.is_err());
    }

    #[test]
    fn get_typed_missing() {
        let mw = ModelWeights::new();
        let result = mw.get_typed::<f32>("nope");
        assert!(result.is_err());
    }

    #[test]
    fn mock_allocator() {
        let alloc = MockGpuAllocator;
        let buf = alloc.allocate(64).unwrap();
        assert_eq!(buf.len(), 64);
        let buf2 = alloc.upload(&[1, 2, 3]).unwrap();
        assert_eq!(buf2.as_bytes(), &[1, 2, 3]);
    }
}
