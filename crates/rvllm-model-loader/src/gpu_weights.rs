//! GPU model weights container backed by CUDA device memory.
//!
//! Holds all model weight tensors as `CudaSlice<f32>` on a single device,
//! with shape metadata for downstream layers to query dimensions.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice as _};
    use half::f16;
    use rvllm_core::error::{LLMError, Result};
    use tracing::debug;

    /// Container holding all model weights as typed CUDA device buffers.
    ///
    /// Each weight is stored as a `CudaSlice<f32>` alongside its shape.
    /// When `use_fp16` is enabled, projection weights are also stored as
    /// `CudaSlice<f16>` in `weights_f16` for half-precision GEMM.
    /// When `use_fp8` is enabled, projection weights are stored as
    /// `CudaSlice<u8>` (FP8 E4M3) with optional per-tensor f32 scales.
    pub struct GpuModelWeights {
        weights: HashMap<String, CudaSlice<f32>>,
        weights_f16: HashMap<String, CudaSlice<f16>>,
        weights_fp8: HashMap<String, CudaSlice<u8>>,
        weight_scales: HashMap<String, CudaSlice<f32>>,
        /// Per-group zero points for GPTQ INT4 weights (not used for FP8).
        weight_zeros: HashMap<String, CudaSlice<f32>>,
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
                weights_fp8: HashMap::new(),
                weight_scales: HashMap::new(),
                weight_zeros: HashMap::new(),
                shapes,
            }
        }

        /// Build an empty container, useful for tests or incremental loading.
        pub fn empty() -> Self {
            Self {
                weights: HashMap::new(),
                weights_f16: HashMap::new(),
                weights_fp8: HashMap::new(),
                weight_scales: HashMap::new(),
                weight_zeros: HashMap::new(),
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

        /// Insert a single FP8 (u8) weight tensor with its shape.
        pub fn insert_fp8(&mut self, name: String, data: CudaSlice<u8>, shape: Vec<usize>) {
            self.shapes.insert(name.clone(), shape);
            self.weights_fp8.insert(name, data);
        }

        /// Insert a per-tensor scale factor for an FP8 weight.
        pub fn insert_scale(&mut self, name: String, data: CudaSlice<f32>) {
            self.weight_scales.insert(name, data);
        }

        /// Insert per-group zero points for a GPTQ INT4 weight.
        pub fn insert_zeros(&mut self, name: String, data: CudaSlice<f32>) {
            self.weight_zeros.insert(name, data);
        }

        /// Look up a weight by name.
        pub fn get(&self, name: &str) -> Option<&CudaSlice<f32>> {
            self.weights.get(name)
        }

        /// Look up an f16 weight by name.
        pub fn get_f16(&self, name: &str) -> Option<&CudaSlice<f16>> {
            self.weights_f16.get(name)
        }

        /// Look up an FP8 (u8) weight by name.
        pub fn get_fp8(&self, name: &str) -> Option<&CudaSlice<u8>> {
            self.weights_fp8.get(name)
        }

        /// Look up the per-tensor scale for an FP8 weight by name.
        pub fn get_scale(&self, name: &str) -> Option<&CudaSlice<f32>> {
            self.weight_scales.get(name)
        }

        /// Look up per-group zero points for a GPTQ weight by name.
        pub fn get_zeros(&self, name: &str) -> Option<&CudaSlice<f32>> {
            self.weight_zeros.get(name)
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
            self.weights.len()
        }

        /// Iterate over all weight names.
        pub fn names(&self) -> impl Iterator<Item = &str> {
            self.weights.keys().map(|s| s.as_str())
        }

        /// Check whether a weight exists.
        pub fn contains(&self, name: &str) -> bool {
            self.weights.contains_key(name)
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
        /// tensor to the given CUDA device via `htod_sync_copy`.
        pub fn from_host(
            host_weights: HashMap<String, Vec<f32>>,
            shapes: HashMap<String, Vec<usize>>,
            device: &Arc<CudaDevice>,
        ) -> Result<Self> {
            let mut gpu_weights = HashMap::with_capacity(host_weights.len());
            for (name, data) in host_weights {
                let slice = device.htod_sync_copy(&data).map_err(|e| {
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
                weights_fp8: HashMap::new(),
                weight_scales: HashMap::new(),
                weight_zeros: HashMap::new(),
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
