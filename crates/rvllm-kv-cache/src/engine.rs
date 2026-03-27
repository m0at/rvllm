//! CacheEngine: manages per-layer GPU and CPU KV caches.

use half::f16;
use tracing::{debug, info};
use rvllm_core::prelude::{BlockId, LLMError, Result};
use rvllm_gpu::prelude::{CpuBuffer, GpuAllocator, GpuBuffer, GpuStream};

/// Per-layer paged KV cache engine.
///
/// Manages GPU and CPU caches for all transformer layers and provides
/// block-level copy, swap-in, and swap-out operations.
pub struct CacheEngine {
    pub gpu_cache: Vec<(GpuBuffer<f16>, GpuBuffer<f16>)>,
    pub cpu_cache: Vec<(CpuBuffer<f16>, CpuBuffer<f16>)>,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
}

impl CacheEngine {
    pub fn new<A: GpuAllocator>(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        allocator: &A,
    ) -> Result<Self> {
        let elements_per_block = block_size * num_heads * head_dim;
        let gpu_total = num_gpu_blocks * elements_per_block;
        let cpu_total = num_cpu_blocks * elements_per_block;

        info!(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            "initializing cache engine"
        );

        let mut gpu_cache = Vec::with_capacity(num_layers);
        let mut cpu_cache = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            debug!(layer, gpu_total, "allocating GPU KV cache");
            let key_gpu = allocator.alloc::<f16>(gpu_total).map_err(|e| {
                LLMError::MemoryError(format!("GPU key alloc layer {layer}: {e}"))
            })?;
            let val_gpu = allocator.alloc::<f16>(gpu_total).map_err(|e| {
                LLMError::MemoryError(format!("GPU value alloc layer {layer}: {e}"))
            })?;
            gpu_cache.push((key_gpu, val_gpu));

            debug!(layer, cpu_total, "allocating CPU KV cache");
            let key_cpu = CpuBuffer::<f16>::new(cpu_total);
            let val_cpu = CpuBuffer::<f16>::new(cpu_total);
            cpu_cache.push((key_cpu, val_cpu));
        }

        Ok(Self {
            gpu_cache,
            cpu_cache,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
        })
    }

    fn elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    /// Copy blocks within GPU cache. Each (src, dst) pair copies a full block
    /// across all layers.
    pub fn copy_blocks(
        &mut self,
        mapping: &[(BlockId, BlockId)],
        _stream: &GpuStream,
    ) -> Result<()> {
        let epb = self.elements_per_block();

        for &(src_id, dst_id) in mapping {
            let src = src_id.0 as usize;
            let dst = dst_id.0 as usize;

            if src >= self.num_gpu_blocks || dst >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "copy_blocks: block index out of range (src={src}, dst={dst}, max={})",
                    self.num_gpu_blocks
                )));
            }

            let src_off = src * epb;
            let dst_off = dst * epb;

            for (key_buf, val_buf) in &mut self.gpu_cache {
                let mut key_data = key_buf.copy_to_host()?;
                let src_key: Vec<f16> = key_data[src_off..src_off + epb].to_vec();
                key_data[dst_off..dst_off + epb].copy_from_slice(&src_key);
                key_buf.copy_from_host(&key_data)?;

                let mut val_data = val_buf.copy_to_host()?;
                let src_val: Vec<f16> = val_data[src_off..src_off + epb].to_vec();
                val_data[dst_off..dst_off + epb].copy_from_slice(&src_val);
                val_buf.copy_from_host(&val_data)?;
            }
        }

        debug!(pairs = mapping.len(), "copy_blocks complete");
        Ok(())
    }

    /// Swap blocks from CPU cache into GPU cache.
    /// Each (cpu_block, gpu_block) copies CPU -> GPU across all layers.
    pub fn swap_in(
        &mut self,
        mapping: &[(BlockId, BlockId)],
        _stream: &GpuStream,
    ) -> Result<()> {
        let epb = self.elements_per_block();

        for &(cpu_id, gpu_id) in mapping {
            let cpu_idx = cpu_id.0 as usize;
            let gpu_idx = gpu_id.0 as usize;

            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "swap_in: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }
            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "swap_in: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }

            let cpu_off = cpu_idx * epb;
            let gpu_off = gpu_idx * epb;

            for (layer_idx, ((key_gpu, val_gpu), (key_cpu, val_cpu))) in self
                .gpu_cache
                .iter_mut()
                .zip(self.cpu_cache.iter())
                .enumerate()
            {
                let mut key_data = key_gpu.copy_to_host()?;
                key_data[gpu_off..gpu_off + epb]
                    .copy_from_slice(&key_cpu.as_slice()[cpu_off..cpu_off + epb]);
                key_gpu.copy_from_host(&key_data)?;

                let mut val_data = val_gpu.copy_to_host()?;
                val_data[gpu_off..gpu_off + epb]
                    .copy_from_slice(&val_cpu.as_slice()[cpu_off..cpu_off + epb]);
                val_gpu.copy_from_host(&val_data)?;

                let _ = layer_idx;
            }
        }

        debug!(pairs = mapping.len(), "swap_in complete");
        Ok(())
    }

    /// Swap blocks from GPU cache out to CPU cache.
    /// Each (gpu_block, cpu_block) copies GPU -> CPU across all layers.
    pub fn swap_out(
        &mut self,
        mapping: &[(BlockId, BlockId)],
        _stream: &GpuStream,
    ) -> Result<()> {
        let epb = self.elements_per_block();

        for &(gpu_id, cpu_id) in mapping {
            let gpu_idx = gpu_id.0 as usize;
            let cpu_idx = cpu_id.0 as usize;

            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "swap_out: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }
            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::MemoryError(format!(
                    "swap_out: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }

            let gpu_off = gpu_idx * epb;
            let cpu_off = cpu_idx * epb;

            for ((key_gpu, val_gpu), (key_cpu, val_cpu)) in
                self.gpu_cache.iter().zip(self.cpu_cache.iter_mut())
            {
                let key_data = key_gpu.copy_to_host()?;
                key_cpu.as_mut_slice()[cpu_off..cpu_off + epb]
                    .copy_from_slice(&key_data[gpu_off..gpu_off + epb]);

                let val_data = val_gpu.copy_to_host()?;
                val_cpu.as_mut_slice()[cpu_off..cpu_off + epb]
                    .copy_from_slice(&val_data[gpu_off..gpu_off + epb]);
            }
        }

        debug!(pairs = mapping.len(), "swap_out complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_gpu::prelude::MockGpuAllocator;

    fn make_engine(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> CacheEngine {
        let alloc = MockGpuAllocator::new(1 << 30);
        CacheEngine::new(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            &alloc,
        )
        .unwrap()
    }

    #[test]
    fn engine_creation() {
        let engine = make_engine(2, 4, 64, 8, 10, 5);
        assert_eq!(engine.gpu_cache.len(), 2);
        assert_eq!(engine.cpu_cache.len(), 2);
        let epb = 8 * 4 * 64;
        assert_eq!(engine.gpu_cache[0].0.len(), 10 * epb);
        assert_eq!(engine.cpu_cache[0].0.len(), 5 * epb);
    }

    #[test]
    fn copy_blocks_within_gpu() {
        let mut engine = make_engine(1, 2, 4, 2, 4, 2);
        let epb = engine.elements_per_block();
        let stream = GpuStream::new(0).unwrap();

        // Write known pattern into block 0
        let mut data = engine.gpu_cache[0].0.copy_to_host().unwrap();
        for i in 0..epb {
            data[i] = f16::from_f32(i as f32);
        }
        engine.gpu_cache[0].0.copy_from_host(&data).unwrap();

        let mut vdata = engine.gpu_cache[0].1.copy_to_host().unwrap();
        for i in 0..epb {
            vdata[i] = f16::from_f32((i + 100) as f32);
        }
        engine.gpu_cache[0].1.copy_from_host(&vdata).unwrap();

        engine
            .copy_blocks(&[(BlockId(0), BlockId(2))], &stream)
            .unwrap();

        let key_data = engine.gpu_cache[0].0.copy_to_host().unwrap();
        let val_data = engine.gpu_cache[0].1.copy_to_host().unwrap();
        let dst_off = 2 * epb;
        for i in 0..epb {
            assert_eq!(key_data[dst_off + i], f16::from_f32(i as f32));
            assert_eq!(val_data[dst_off + i], f16::from_f32((i + 100) as f32));
        }
    }

    #[test]
    fn copy_blocks_out_of_range() {
        let mut engine = make_engine(1, 2, 4, 2, 4, 2);
        let stream = GpuStream::new(0).unwrap();
        let result = engine.copy_blocks(&[(BlockId(0), BlockId(10))], &stream);
        assert!(result.is_err());
    }

    #[test]
    fn swap_in_and_out() {
        let mut engine = make_engine(1, 2, 4, 2, 4, 4);
        let epb = engine.elements_per_block();
        let stream = GpuStream::new(0).unwrap();

        // Write pattern into GPU block 1
        let gpu_off = 1 * epb;
        let mut kdata = engine.gpu_cache[0].0.copy_to_host().unwrap();
        let mut vdata = engine.gpu_cache[0].1.copy_to_host().unwrap();
        for i in 0..epb {
            kdata[gpu_off + i] = f16::from_f32(42.0);
            vdata[gpu_off + i] = f16::from_f32(43.0);
        }
        engine.gpu_cache[0].0.copy_from_host(&kdata).unwrap();
        engine.gpu_cache[0].1.copy_from_host(&vdata).unwrap();

        // Swap out GPU block 1 -> CPU block 3
        engine
            .swap_out(&[(BlockId(1), BlockId(3))], &stream)
            .unwrap();

        let cpu_off = 3 * epb;
        assert_eq!(engine.cpu_cache[0].0.as_slice()[cpu_off], f16::from_f32(42.0));
        assert_eq!(engine.cpu_cache[0].1.as_slice()[cpu_off], f16::from_f32(43.0));

        // Zero GPU block 1
        let mut kdata = engine.gpu_cache[0].0.copy_to_host().unwrap();
        let mut vdata = engine.gpu_cache[0].1.copy_to_host().unwrap();
        for i in 0..epb {
            kdata[gpu_off + i] = f16::ZERO;
            vdata[gpu_off + i] = f16::ZERO;
        }
        engine.gpu_cache[0].0.copy_from_host(&kdata).unwrap();
        engine.gpu_cache[0].1.copy_from_host(&vdata).unwrap();

        // Swap in CPU block 3 -> GPU block 1
        engine
            .swap_in(&[(BlockId(3), BlockId(1))], &stream)
            .unwrap();

        let kdata = engine.gpu_cache[0].0.copy_to_host().unwrap();
        let vdata = engine.gpu_cache[0].1.copy_to_host().unwrap();
        assert_eq!(kdata[gpu_off], f16::from_f32(42.0));
        assert_eq!(vdata[gpu_off], f16::from_f32(43.0));
    }

    #[test]
    fn swap_in_out_of_range() {
        let mut engine = make_engine(1, 2, 4, 2, 4, 2);
        let stream = GpuStream::new(0).unwrap();
        assert!(engine.swap_in(&[(BlockId(10), BlockId(0))], &stream).is_err());
        assert!(engine.swap_in(&[(BlockId(0), BlockId(10))], &stream).is_err());
    }

    #[test]
    fn swap_out_out_of_range() {
        let mut engine = make_engine(1, 2, 4, 2, 4, 2);
        let stream = GpuStream::new(0).unwrap();
        assert!(engine.swap_out(&[(BlockId(10), BlockId(0))], &stream).is_err());
        assert!(engine.swap_out(&[(BlockId(0), BlockId(10))], &stream).is_err());
    }

    #[test]
    fn multi_layer_copy() {
        let mut engine = make_engine(3, 2, 4, 2, 4, 2);
        let epb = engine.elements_per_block();
        let stream = GpuStream::new(0).unwrap();

        // Write distinct pattern per layer into block 0
        for layer in 0..3 {
            let val = f16::from_f32((layer * 10) as f32);
            let mut data = engine.gpu_cache[layer].0.copy_to_host().unwrap();
            for i in 0..epb {
                data[i] = val;
            }
            engine.gpu_cache[layer].0.copy_from_host(&data).unwrap();
        }

        engine
            .copy_blocks(&[(BlockId(0), BlockId(1))], &stream)
            .unwrap();

        for layer in 0..3 {
            let expected = f16::from_f32((layer * 10) as f32);
            let data = engine.gpu_cache[layer].0.copy_to_host().unwrap();
            let dst_off = 1 * epb;
            assert_eq!(data[dst_off], expected);
        }
    }
}
