// v2/kv_cache.rs -- CUDA KV cache engine with direct device-to-device copies
//
// Layout: [num_blocks, block_size, num_heads, head_dim] flattened in f16.
// Elements per block = block_size * num_heads * head_dim.
// Bytes per block = elements_per_block * 2 (f16 = 2 bytes).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use half::f16;

use crate::types::BlockId;

pub trait KVCacheEngine {
    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]);
    fn swap_in(&mut self, pairs: &[(BlockId, BlockId)]);
    fn swap_out(&mut self, pairs: &[(BlockId, BlockId)]);
    fn num_gpu_blocks(&self) -> usize;
    fn num_cpu_blocks(&self) -> usize;
}

pub struct CacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub num_cpu_blocks: usize,
}

pub struct CudaKVCache {
    pub gpu_cache: Vec<(CudaSlice<f16>, CudaSlice<f16>)>,
    cpu_cache: Vec<(Vec<f16>, Vec<f16>)>,
    num_layers: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    elements_per_block: usize,
    block_size: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaKVCache {
    pub fn new(
        config: &CacheConfig,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, String> {
        context
            .bind_to_thread()
            .map_err(|e| format!("KV cache CUDA context bind failed: {e}"))?;

        let elements_per_block = config.block_size * config.num_heads * config.head_dim;
        let gpu_total = config.num_gpu_blocks * elements_per_block;
        let cpu_total = config.num_cpu_blocks * elements_per_block;

        let mut gpu_cache = Vec::with_capacity(config.num_layers);
        let mut cpu_cache = Vec::with_capacity(config.num_layers);

        for layer in 0..config.num_layers {
            let key_gpu: CudaSlice<f16> = stream
                .alloc_zeros(gpu_total)
                .map_err(|e| format!("CUDA key cache alloc failed layer {layer}: {e}"))?;
            let val_gpu: CudaSlice<f16> = stream
                .alloc_zeros(gpu_total)
                .map_err(|e| format!("CUDA val cache alloc failed layer {layer}: {e}"))?;
            gpu_cache.push((key_gpu, val_gpu));

            cpu_cache.push((vec![f16::ZERO; cpu_total], vec![f16::ZERO; cpu_total]));
        }

        Ok(Self {
            gpu_cache,
            cpu_cache,
            num_layers: config.num_layers,
            num_gpu_blocks: config.num_gpu_blocks,
            num_cpu_blocks: config.num_cpu_blocks,
            elements_per_block,
            block_size: config.block_size,
            context,
            stream,
        })
    }

    fn bytes_per_block(&self) -> usize {
        self.elements_per_block * std::mem::size_of::<f16>()
    }

    pub fn sync(&self) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("KV cache stream sync failed: {e}"))
    }

    pub fn get_gpu_cache(&self) -> &[(CudaSlice<f16>, CudaSlice<f16>)] {
        &self.gpu_cache
    }

    pub fn elements_per_block(&self) -> usize {
        self.elements_per_block
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn max_blocks_for_memory(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        available_bytes: usize,
    ) -> usize {
        let elements_per_block = block_size * num_heads * head_dim;
        let bytes_per_block = 2 * num_layers * elements_per_block * std::mem::size_of::<f16>();
        if bytes_per_block == 0 {
            return 0;
        }
        available_bytes / bytes_per_block
    }
}

impl KVCacheEngine for CudaKVCache {
    fn copy_blocks(&mut self, pairs: &[(BlockId, BlockId)]) {
        if pairs.is_empty() {
            return;
        }

        self.context
            .bind_to_thread()
            .expect("copy_blocks: KV cache CUDA context bind failed");

        let bpb = self.bytes_per_block();
        let stream = &self.stream;
        let cu_stream = stream.cu_stream();

        for &(src, dst) in pairs {
            let src_byte_offset = src.0 as usize * bpb;
            let dst_byte_offset = dst.0 as usize * bpb;

            for layer_idx in 0..self.num_layers {
                let (ref k_cache, ref v_cache) = self.gpu_cache[layer_idx];
                let (k_ptr, _k_guard) = DevicePtr::device_ptr(k_cache, stream);
                let (v_ptr, _v_guard) = DevicePtr::device_ptr(v_cache, stream);

                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        k_ptr + dst_byte_offset as u64,
                        k_ptr + src_byte_offset as u64,
                        bpb,
                        cu_stream,
                    )
                    .result()
                    .expect("copy_blocks: cuMemcpyDtoDAsync K failed");
                }

                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        v_ptr + dst_byte_offset as u64,
                        v_ptr + src_byte_offset as u64,
                        bpb,
                        cu_stream,
                    )
                    .result()
                    .expect("copy_blocks: cuMemcpyDtoDAsync V failed");
                }
            }
        }
    }

    fn swap_in(&mut self, pairs: &[(BlockId, BlockId)]) {
        if pairs.is_empty() {
            return;
        }

        self.context
            .bind_to_thread()
            .expect("swap_in: KV cache CUDA context bind failed");

        let epb = self.elements_per_block;
        let bpb = self.bytes_per_block();
        let stream = &self.stream;
        let cu_stream = stream.cu_stream();

        for &(cpu_id, gpu_id) in pairs {
            let cpu_offset = cpu_id.0 as usize * epb;
            let gpu_byte_offset = gpu_id.0 as usize * bpb;

            for layer_idx in 0..self.num_layers {
                let (ref k_cpu, ref v_cpu) = self.cpu_cache[layer_idx];
                let (ref k_gpu, ref v_gpu) = self.gpu_cache[layer_idx];
                let (k_gpu_ptr, _kg) = DevicePtr::device_ptr(k_gpu, stream);
                let (v_gpu_ptr, _vg) = DevicePtr::device_ptr(v_gpu, stream);

                let k_src = &k_cpu[cpu_offset..cpu_offset + epb];
                let v_src = &v_cpu[cpu_offset..cpu_offset + epb];

                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        k_gpu_ptr + gpu_byte_offset as u64,
                        k_src,
                        cu_stream,
                    )
                    .expect("swap_in: memcpy_htod_async K failed");
                }

                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        v_gpu_ptr + gpu_byte_offset as u64,
                        v_src,
                        cu_stream,
                    )
                    .expect("swap_in: memcpy_htod_async V failed");
                }
            }
        }
    }

    fn swap_out(&mut self, pairs: &[(BlockId, BlockId)]) {
        if pairs.is_empty() {
            return;
        }

        self.context
            .bind_to_thread()
            .expect("swap_out: KV cache CUDA context bind failed");

        let epb = self.elements_per_block;
        let bpb = self.bytes_per_block();
        let stream = &self.stream;
        let cu_stream = stream.cu_stream();
        let gpu_cache = &self.gpu_cache;
        let cpu_cache = &mut self.cpu_cache;

        for &(gpu_id, cpu_id) in pairs {
            let gpu_byte_offset = gpu_id.0 as usize * bpb;
            let cpu_offset = cpu_id.0 as usize * epb;

            for layer_idx in 0..self.num_layers {
                let (ref k_gpu, ref v_gpu) = gpu_cache[layer_idx];
                let (ref mut k_cpu, ref mut v_cpu) = cpu_cache[layer_idx];
                let (k_gpu_ptr, _kg) = DevicePtr::device_ptr(k_gpu, stream);
                let (v_gpu_ptr, _vg) = DevicePtr::device_ptr(v_gpu, stream);

                let k_dst = &mut k_cpu[cpu_offset..cpu_offset + epb];
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_async(
                        k_dst,
                        k_gpu_ptr + gpu_byte_offset as u64,
                        cu_stream,
                    )
                    .expect("swap_out: memcpy_dtoh_async K failed");
                }

                let v_dst = &mut v_cpu[cpu_offset..cpu_offset + epb];
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_async(
                        v_dst,
                        v_gpu_ptr + gpu_byte_offset as u64,
                        cu_stream,
                    )
                    .expect("swap_out: memcpy_dtoh_async V failed");
                }
            }
        }
    }

    fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    fn num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }
}
