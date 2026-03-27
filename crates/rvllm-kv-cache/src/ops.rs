//! Cache operations for writing into paged KV buffers.

use half::f16;
use tracing::debug;
use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::prelude::GpuBuffer;

/// Reshape and cache key/value tensors into their paged GPU buffers.
///
/// `key` and `value` are flat token-level tensors of shape [num_tokens, num_heads * head_dim].
/// `slot_mapping` maps each token position to a flat slot index in the cache buffer.
/// The slot index encodes both the block and the position within the block:
///   block_idx = slot / block_size
///   block_offset = slot % block_size
///
/// The cache buffers are shaped [num_blocks, block_size, num_heads, head_dim] (flattened).
pub fn reshape_and_cache(
    key: &[f16],
    value: &[f16],
    key_cache: &mut GpuBuffer<f16>,
    value_cache: &mut GpuBuffer<f16>,
    slot_mapping: &[i32],
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let head_stride = num_heads * head_dim;
    let num_tokens = slot_mapping.len();

    if key.len() != num_tokens * head_stride {
        return Err(LLMError::MemoryError(format!(
            "reshape_and_cache: key length {} != num_tokens({}) * head_stride({})",
            key.len(),
            num_tokens,
            head_stride
        )));
    }
    if value.len() != num_tokens * head_stride {
        return Err(LLMError::MemoryError(format!(
            "reshape_and_cache: value length {} != num_tokens({}) * head_stride({})",
            value.len(),
            num_tokens,
            head_stride
        )));
    }

    let block_stride = block_size * head_stride;

    let mut key_data = key_cache.copy_to_host()?;
    let mut val_data = value_cache.copy_to_host()?;

    for (token_idx, &slot) in slot_mapping.iter().enumerate() {
        if slot < 0 {
            // Padding token, skip
            continue;
        }
        let slot = slot as usize;
        let block_idx = slot / block_size;
        let block_offset = slot % block_size;

        let cache_offset = block_idx * block_stride + block_offset * head_stride;
        let src_offset = token_idx * head_stride;

        if cache_offset + head_stride > key_data.len() {
            return Err(LLMError::MemoryError(format!(
                "reshape_and_cache: cache offset {cache_offset} + {head_stride} exceeds buffer len {}",
                key_data.len()
            )));
        }

        key_data[cache_offset..cache_offset + head_stride]
            .copy_from_slice(&key[src_offset..src_offset + head_stride]);
        val_data[cache_offset..cache_offset + head_stride]
            .copy_from_slice(&value[src_offset..src_offset + head_stride]);
    }

    key_cache.copy_from_host(&key_data)?;
    value_cache.copy_from_host(&val_data)?;

    debug!(num_tokens, "reshape_and_cache complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_gpu::prelude::{GpuAllocator, MockGpuAllocator};

    fn make_buffer(
        alloc: &MockGpuAllocator,
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> GpuBuffer<f16> {
        let total = num_blocks * block_size * num_heads * head_dim;
        alloc.alloc::<f16>(total).unwrap()
    }

    #[test]
    fn reshape_single_token() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let num_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let num_blocks = 2;

        let mut key_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);
        let mut val_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);

        let hs = num_heads * head_dim;
        let key: Vec<f16> = (0..hs).map(|i| f16::from_f32(i as f32)).collect();
        let value: Vec<f16> = (0..hs).map(|i| f16::from_f32((i + 100) as f32)).collect();

        // Slot 5 => block 1, offset 1 (block_size=4)
        let slot_mapping = vec![5_i32];

        reshape_and_cache(
            &key,
            &value,
            &mut key_cache,
            &mut val_cache,
            &slot_mapping,
            num_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        let key_data = key_cache.copy_to_host().unwrap();
        let val_data = val_cache.copy_to_host().unwrap();

        let block_stride = block_size * hs;
        let cache_off = 1 * block_stride + 1 * hs;
        for i in 0..hs {
            assert_eq!(key_data[cache_off + i], f16::from_f32(i as f32));
            assert_eq!(val_data[cache_off + i], f16::from_f32((i + 100) as f32));
        }
    }

    #[test]
    fn reshape_multiple_tokens() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let num_heads = 1;
        let head_dim = 2;
        let block_size = 2;
        let num_blocks = 4;
        let hs = num_heads * head_dim;

        let mut key_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);
        let mut val_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);

        // 3 tokens
        let key: Vec<f16> = (0..3 * hs).map(|i| f16::from_f32(i as f32)).collect();
        let value: Vec<f16> = (0..3 * hs).map(|i| f16::from_f32((i + 50) as f32)).collect();
        let slot_mapping = vec![0_i32, 3, 6];

        reshape_and_cache(
            &key,
            &value,
            &mut key_cache,
            &mut val_cache,
            &slot_mapping,
            num_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        let key_data = key_cache.copy_to_host().unwrap();

        // Token 0 -> slot 0 -> block 0, offset 0
        assert_eq!(key_data[0], f16::from_f32(0.0));
        assert_eq!(key_data[1], f16::from_f32(1.0));

        // Token 1 -> slot 3 -> block 1, offset 1
        let block_stride = block_size * hs;
        let off = 1 * block_stride + 1 * hs;
        assert_eq!(key_data[off], f16::from_f32(2.0));
        assert_eq!(key_data[off + 1], f16::from_f32(3.0));
    }

    #[test]
    fn negative_slot_skipped() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let num_heads = 1;
        let head_dim = 2;
        let block_size = 2;
        let num_blocks = 2;
        let hs = num_heads * head_dim;

        let mut key_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);
        let mut val_cache = make_buffer(&alloc, num_blocks, block_size, num_heads, head_dim);

        let key: Vec<f16> = vec![f16::from_f32(99.0); 2 * hs];
        let value: Vec<f16> = vec![f16::from_f32(99.0); 2 * hs];
        let slot_mapping = vec![-1_i32, 0];

        reshape_and_cache(
            &key,
            &value,
            &mut key_cache,
            &mut val_cache,
            &slot_mapping,
            num_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        let key_data = key_cache.copy_to_host().unwrap();
        // Slot -1 should leave cache at zero everywhere except slot 0
        // Token 1 -> slot 0
        assert_eq!(key_data[0], f16::from_f32(99.0));
    }

    #[test]
    fn mismatched_key_length_errors() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let mut key_cache = make_buffer(&alloc, 2, 2, 1, 2);
        let mut val_cache = make_buffer(&alloc, 2, 2, 1, 2);

        let result = reshape_and_cache(
            &[f16::ZERO; 3], // wrong length
            &[f16::ZERO; 2],
            &mut key_cache,
            &mut val_cache,
            &[0],
            1,
            2,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn out_of_bounds_slot_errors() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let mut key_cache = make_buffer(&alloc, 1, 2, 1, 2);
        let mut val_cache = make_buffer(&alloc, 1, 2, 1, 2);
        let hs = 2;

        let key: Vec<f16> = vec![f16::ZERO; hs];
        let value: Vec<f16> = vec![f16::ZERO; hs];
        // Slot 10 would be block 5 which doesn't exist (only 1 block)
        let result = reshape_and_cache(
            &key,
            &value,
            &mut key_cache,
            &mut val_cache,
            &[10],
            1,
            2,
            2,
        );
        assert!(result.is_err());
    }
}
