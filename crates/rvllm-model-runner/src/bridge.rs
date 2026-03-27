//! Bridge / compatibility layer re-exporting types from upstream crates.
//!
//! Types whose upstream API matches are re-exported directly. Types with API
//! mismatches (e.g. GpuBuffer, AttentionBackend) keep a local shim until the
//! upstream crates unify on a common shape-aware buffer abstraction.

use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// rvllm-core: error / result (direct re-export, fully compatible)
// ---------------------------------------------------------------------------
pub use rvllm_core::error::{LLMError, Result};

// ---------------------------------------------------------------------------
// rvllm-gpu: re-export what we can directly
// ---------------------------------------------------------------------------
// The real GpuAllocator / GpuBuffer from rvllm-gpu use Pod + Send bounds and
// an opaque inner buffer without data/shape fields. Model-runner code needs
// direct data/shape access for CPU-mock execution, so we keep a thin local
// GpuBuffer. We re-export the upstream traits/types under qualified names so
// callers can migrate incrementally.
pub mod upstream_gpu {
    pub use rvllm_gpu::prelude::{
        GpuAllocator as RealGpuAllocator,
        GpuBuffer as RealGpuBuffer,
        GpuStream as RealGpuStream,
        MemoryInfo,
    };
    // MockGpuAllocator is available when rvllm-gpu has mock-gpu feature (its default)
    pub use rvllm_gpu::mock::MockGpuAllocator as RealMockGpuAllocator;
}

// ---------------------------------------------------------------------------
// rvllm-attention: re-export the real trait + metadata under qualified name
// ---------------------------------------------------------------------------
pub mod upstream_attention {
    pub use rvllm_attention::{
        AttentionBackend as RealAttentionBackend,
        AttentionMetadata as RealAttentionMetadata,
        MockAttentionBackend as RealMockAttentionBackend,
    };
}

// ---------------------------------------------------------------------------
// rvllm-kv-cache: broken upstream (gpu module deleted, engine/ops not updated)
// TODO: re-add once rvllm-kv-cache compiles:
// pub mod upstream_kv_cache {
//     pub use rvllm_kv_cache::{CacheEngine as RealCacheEngine, KVCache as RealKVCache};
// }
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// rvllm-model-loader: re-export real weight types
// ---------------------------------------------------------------------------
pub mod upstream_model_loader {
    pub use rvllm_model_loader::weights::{
        ModelWeights as RealModelWeights,
        WeightTensor as RealWeightTensor,
    };
}

// ---------------------------------------------------------------------------
// GPU buffer shim -- local struct with data/shape for CPU-mock forward passes.
// TODO: Unify with rvllm_gpu::GpuBuffer once it gains shape metadata and
//       host-accessible data views.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct GpuBuffer<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: Clone + Default> GpuBuffer<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![T::default(); len],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// GPU allocator shim -- compatible with the local GpuBuffer.
// TODO: Replace with rvllm_gpu::prelude::GpuAllocator once buffer APIs unify.
// ---------------------------------------------------------------------------
pub trait GpuAllocator: Send + Sync {
    fn alloc_f16(&self, num_elements: usize) -> Result<GpuBuffer<f16>>;
    fn alloc_f32(&self, num_elements: usize) -> Result<GpuBuffer<f32>>;
    fn free_gpu_bytes(&self) -> usize;
}

/// Opaque GPU stream.
/// TODO: Replace with rvllm_gpu::prelude::GpuStream.
pub struct GpuStream {
    _id: u32,
}

impl GpuStream {
    pub fn new(id: u32) -> Self {
        Self { _id: id }
    }
}

/// Mock allocator for tests.
/// TODO: Replace with rvllm_gpu::mock::MockGpuAllocator once buffer APIs unify.
pub struct MockGpuAllocator {
    pub free_bytes: usize,
}

impl MockGpuAllocator {
    pub fn new(free_bytes: usize) -> Arc<Self> {
        Arc::new(Self { free_bytes })
    }
}

impl GpuAllocator for MockGpuAllocator {
    fn alloc_f16(&self, num_elements: usize) -> Result<GpuBuffer<f16>> {
        Ok(GpuBuffer::zeros(&[num_elements]))
    }

    fn alloc_f32(&self, num_elements: usize) -> Result<GpuBuffer<f32>> {
        Ok(GpuBuffer::zeros(&[num_elements]))
    }

    fn free_gpu_bytes(&self) -> usize {
        self.free_bytes
    }
}

// ---------------------------------------------------------------------------
// Attention shim -- model-runner uses a 5-arg forward() while the real
// rvllm-attention AttentionBackend uses a 7-arg signature with separate
// key_cache/value_cache/block_tables/context_lens/scale parameters.
// TODO: Align model-runner forward pass to call the real AttentionBackend
//       signature, passing cache tensors and metadata fields directly.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AttentionMetadata {
    pub slot_mapping: Vec<u32>,
    pub context_lens: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
    pub max_context_len: u32,
}

pub trait AttentionBackend: Send + Sync {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key: &GpuBuffer<f16>,
        value: &GpuBuffer<f16>,
        metadata: &AttentionMetadata,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f16>>;
}

/// Mock attention backend that returns zeros.
pub struct MockAttentionBackend;

impl AttentionBackend for MockAttentionBackend {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        _key: &GpuBuffer<f16>,
        _value: &GpuBuffer<f16>,
        _metadata: &AttentionMetadata,
        _layer_idx: usize,
    ) -> Result<GpuBuffer<f16>> {
        Ok(GpuBuffer::zeros(&query.shape))
    }
}

// ---------------------------------------------------------------------------
// KV cache shim -- the real CacheEngine requires an Arc<dyn GpuAllocator>
// from rvllm-kv-cache (with a different allocator trait). Keep the simple
// version for now.
// TODO: Wire to rvllm_kv_cache::CacheEngine once allocator traits unify.
// ---------------------------------------------------------------------------
pub struct KVCache {
    pub key_cache: GpuBuffer<f16>,
    pub value_cache: GpuBuffer<f16>,
}

pub struct CacheEngine {
    pub caches: Vec<KVCache>,
}

impl CacheEngine {
    pub fn new(num_layers: usize, cache_elements_per_layer: usize) -> Self {
        let caches = (0..num_layers)
            .map(|_| KVCache {
                key_cache: GpuBuffer::zeros(&[cache_elements_per_layer]),
                value_cache: GpuBuffer::zeros(&[cache_elements_per_layer]),
            })
            .collect();
        Self { caches }
    }

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }
}

// ---------------------------------------------------------------------------
// Model loader shim -- the real ModelWeights uses GpuBuffer<u8> (raw bytes)
// with typed access, while model-runner needs f16 data with shape metadata.
// TODO: Wire to rvllm_model_loader::weights once a typed buffer conversion
//       layer exists.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub name: String,
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightTensor>,
}

impl ModelWeights {
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| LLMError::ModelError(format!("weight not found: {}", name)))
    }

    pub fn get_as_buffer(&self, name: &str) -> Result<GpuBuffer<f16>> {
        let t = self.get(name)?;
        Ok(GpuBuffer::from_vec(t.data.clone(), t.shape.clone()))
    }
}
