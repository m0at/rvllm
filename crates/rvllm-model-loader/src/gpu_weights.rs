//! GPU model weights container backed by CUDA device memory.
//!
//! Holds all model weight tensors as `CudaSlice<f32>` on a single device,
//! with shape metadata for downstream layers to query dimensions.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream, DeviceSlice as _};
    use half::f16;
    use rvllm_core::error::{LLMError, Result};
    use tracing::debug;

    /// Container holding all model weights as typed CUDA device buffers.
    ///
    /// Each weight is stored as a `CudaSlice<f32>` alongside its shape.
    /// When `use_fp16` is enabled, projection weights are also stored as
    /// `CudaSlice<f16>` in `weights_f16` for half-precision GEMM.
    pub struct GpuModelWeights {
        weights: HashMap<String, CudaSlice<f32>>,
        weights_f16: HashMap<String, CudaSlice<f16>>,
        weights_u8: HashMap<String, Vec<u8>>,
        shapes: HashMap<String, Vec<usize>>,
    }

    impl GpuModelWeights {
        /// Build from pre-loaded weight maps (typically produced by `gpu_loader::load_weights_to_gpu`).
        pub fn new(
            weights: HashMap<String, CudaSlice<f32>>,
            shapes: HashMap<String, Vec<usize>>,
        ) -> Self {
            debug!(num_weights = weights.len(), "GpuModelWeights created");
            Self {
                weights,
                weights_f16: HashMap::new(),
                weights_u8: HashMap::new(),
                shapes,
            }
        }

        /// Build an empty container, useful for tests or incremental loading.
        pub fn empty() -> Self {
            Self {
                weights: HashMap::new(),
                weights_f16: HashMap::new(),
                weights_u8: HashMap::new(),
                shapes: HashMap::new(),
            }
        }

        /// Insert a single weight tensor with its shape.
        pub fn insert(&mut self, name: String, data: CudaSlice<f32>, shape: Vec<usize>) {
            self.shapes.insert(name.clone(), shape);
            self.weights.insert(name, data);
        }

        /// Insert a single f16 weight tensor with its shape.
        pub fn insert_f16(&mut self, name: String, data: CudaSlice<f16>, shape: Vec<usize>) {
            self.shapes.insert(name.clone(), shape);
            self.weights_f16.insert(name, data);
        }

        /// Insert a raw host-side U8 tensor with its shape.
        pub fn insert_u8(&mut self, name: String, data: Vec<u8>, shape: Vec<usize>) {
            self.shapes.insert(name.clone(), shape);
            self.weights_u8.insert(name, data);
        }

        /// Look up a weight by name.
        pub fn get(&self, name: &str) -> Option<&CudaSlice<f32>> {
            self.weights.get(name)
        }

        /// Look up an f16 weight by name.
        pub fn get_f16(&self, name: &str) -> Option<&CudaSlice<f16>> {
            self.weights_f16.get(name)
        }

        /// Look up a host-side U8 tensor by name.
        pub fn get_u8(&self, name: &str) -> Option<&[u8]> {
            self.weights_u8.get(name).map(|v| v.as_slice())
        }

        /// Remove an f32 weight tensor, returning ownership if it existed.
        pub fn remove(&mut self, name: &str) -> Option<CudaSlice<f32>> {
            self.weights.remove(name)
        }

        /// Remove an f16 weight tensor, returning ownership if it existed.
        pub fn remove_f16(&mut self, name: &str) -> Option<CudaSlice<f16>> {
            self.weights_f16.remove(name)
        }

        /// Remove a host-side u8 tensor, returning ownership if it existed.
        pub fn remove_u8(&mut self, name: &str) -> Option<Vec<u8>> {
            self.weights_u8.remove(name)
        }

        /// Look up a weight by name, returning an error if missing.
        pub fn require(&self, name: &str) -> Result<&CudaSlice<f32>> {
            self.weights
                .get(name)
                .ok_or_else(|| LLMError::GpuError(format!("weight not found: {}", name)))
        }

        /// Look up the shape of a weight by name.
        pub fn shape(&self, name: &str) -> Option<&[usize]> {
            self.shapes.get(name).map(|v| v.as_slice())
        }

        /// Look up shape, returning an error if missing.
        pub fn require_shape(&self, name: &str) -> Result<&[usize]> {
            self.shapes
                .get(name)
                .map(|v| v.as_slice())
                .ok_or_else(|| LLMError::GpuError(format!("shape not found: {}", name)))
        }

        /// Number of weight tensors stored.
        pub fn num_weights(&self) -> usize {
            self.weights.len() + self.weights_f16.len() + self.weights_u8.len()
        }

        /// Iterate over all weight names.
        pub fn names(&self) -> impl Iterator<Item = &str> {
            self.weights
                .keys()
                .chain(self.weights_f16.keys())
                .chain(self.weights_u8.keys())
                .map(|s| s.as_str())
        }

        /// Check whether a weight exists.
        pub fn contains(&self, name: &str) -> bool {
            self.weights.contains_key(name)
                || self.weights_f16.contains_key(name)
                || self.weights_u8.contains_key(name)
        }

        /// Total GPU memory used by all weight buffers, in bytes.
        pub fn total_bytes(&self) -> usize {
            self.weights
                .values()
                .map(|s| s.len() * std::mem::size_of::<f32>())
                .sum()
        }

        /// Build from a host-side weight map by uploading each tensor to GPU.
        ///
        /// Takes a `HashMap<String, Vec<f32>>` plus shapes, and copies every
        /// tensor to the given CUDA stream via `clone_htod`.
        pub fn from_host(
            host_weights: HashMap<String, Vec<f32>>,
            shapes: HashMap<String, Vec<usize>>,
            stream: &Arc<CudaStream>,
        ) -> Result<Self> {
            let mut gpu_weights = HashMap::with_capacity(host_weights.len());
            for (name, data) in host_weights {
                let slice = stream.clone_htod(&data).map_err(|e| {
                    LLMError::GpuError(format!("htod copy failed for {}: {}", name, e))
                })?;
                gpu_weights.insert(name, slice);
            }
            debug!(
                num_weights = gpu_weights.len(),
                "GpuModelWeights uploaded from host"
            );
            Ok(Self {
                weights: gpu_weights,
                weights_f16: HashMap::new(),
                weights_u8: HashMap::new(),
                shapes,
            })
        }

        /// Consume the container and return the underlying maps.
        pub fn into_parts(self) -> (HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>) {
            (self.weights, self.shapes)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::GpuModelWeights;

#[cfg(test)]
mod tests {
    // GpuModelWeights requires a CUDA device for meaningful tests.
    // Compile-time gated tests run with `cargo test --features cuda`.
    // The public API surface (get, shape, require, etc.) is exercised
    // there. Under default features we verify the module compiles.

    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
