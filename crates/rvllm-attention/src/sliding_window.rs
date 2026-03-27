//! Sliding window attention backend for models like Mistral, Gemma 2, etc.
//!
//! Restricts each query token to attend only to the most recent `window_size`
//! KV positions. Integrates with the paged KV cache by skipping blocks that
//! fall entirely outside the window and clamping partially-covered blocks.

use half::f16;
use rvllm_core::prelude::{LLMError, Result};

use crate::backend::AttentionBackend;
use crate::buffer::GpuBuffer;

/// Configuration for sliding window attention.
///
/// Models may use sliding window on all layers (Mistral) or only on
/// alternating layers (Gemma 2). The caller decides per-layer whether to
/// use `SlidingWindowAttention` or a standard attention backend.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of recent tokens each query can attend to.
    pub window_size: usize,
    /// Per-layer override. If `Some`, only layers in the set use sliding window;
    /// all others fall through to the inner backend. If `None`, all layers use it.
    pub layer_mask: Option<Vec<bool>>,
}

impl SlidingWindowConfig {
    /// Create a global sliding window config (all layers).
    pub fn global(window_size: usize) -> Self {
        Self {
            window_size,
            layer_mask: None,
        }
    }

    /// Create a per-layer sliding window config.
    pub fn per_layer(window_size: usize, layer_mask: Vec<bool>) -> Self {
        Self {
            window_size,
            layer_mask: Some(layer_mask),
        }
    }

    /// Check whether a given layer index should use sliding window.
    pub fn applies_to_layer(&self, layer_idx: usize) -> bool {
        match &self.layer_mask {
            None => true,
            Some(mask) => mask.get(layer_idx).copied().unwrap_or(false),
        }
    }
}

/// Sliding window attention backend.
///
/// Wraps an inner `AttentionBackend` and restricts the context window to
/// the last `window_size` tokens for each sequence. Blocks outside the
/// window are skipped in the paged attention computation, and the block
/// manager can evict them.
pub struct SlidingWindowAttention {
    config: SlidingWindowConfig,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention backend.
    pub fn new(config: SlidingWindowConfig) -> Self {
        tracing::info!(
            window_size = config.window_size,
            per_layer = config.layer_mask.is_some(),
            "initializing SlidingWindowAttention backend"
        );
        Self { config }
    }

    /// Return the window size.
    pub fn window_size(&self) -> usize {
        self.config.window_size
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }

    /// Compute the effective start position for a given context length.
    ///
    /// Positions before this are outside the window and should not be attended to.
    pub fn window_start(&self, context_len: usize) -> usize {
        context_len.saturating_sub(self.config.window_size)
    }

    /// Given a context length and block size, return the range of block indices
    /// that fall within the sliding window.
    ///
    /// Returns `(first_block_idx, last_block_idx_exclusive)`.
    pub fn active_block_range(&self, context_len: usize, block_size: usize) -> (usize, usize) {
        if context_len == 0 || block_size == 0 {
            return (0, 0);
        }
        let start_pos = self.window_start(context_len);
        let first_block = start_pos / block_size;
        let last_block = (context_len + block_size - 1) / block_size;
        (first_block, last_block)
    }

    /// Compute which block indices are evictable for a sequence, given its
    /// current context length and block size.
    ///
    /// Blocks whose tokens are entirely before `window_start` can be freed.
    pub fn evictable_blocks(&self, context_len: usize, block_size: usize) -> Vec<usize> {
        if context_len == 0 || block_size == 0 {
            return Vec::new();
        }
        let start_pos = self.window_start(context_len);
        let evictable_end = start_pos / block_size;
        (0..evictable_end).collect()
    }
}

impl AttentionBackend for SlidingWindowAttention {
    /// Paged attention forward pass with sliding window masking.
    ///
    /// Only KV positions within `[context_len - window_size, context_len)` are
    /// attended to for each sequence. Positions outside the window receive
    /// `-inf` score (equivalently, they are skipped).
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D, got {} dims",
                query.shape.len()
            )));
        }
        let num_tokens = query.shape[0];
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];

        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache must be 4-D, got {} dims",
                key_cache.shape.len()
            )));
        }
        let block_size = key_cache.shape[1];

        let num_seqs = context_lens.data.len();
        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_heads, head_dim],
            });
        }

        let max_blocks_per_seq = block_tables.shape.get(1).copied().unwrap_or(0);
        let mut output = vec![f16::ZERO; num_tokens * num_heads * head_dim];

        let mut token_offset = 0usize;
        for seq_idx in 0..num_seqs {
            let ctx_len = (context_lens.data[seq_idx] as usize).min(max_context_len);
            let win_start = self.window_start(ctx_len);

            let seq_tokens = if seq_idx + 1 < num_seqs {
                1
            } else {
                (num_tokens - token_offset).max(1)
            };

            for t in 0..seq_tokens {
                let q_base = (token_offset + t) * num_heads * head_dim;
                if q_base + num_heads * head_dim > query.data.len() {
                    break;
                }

                for h in 0..num_heads {
                    let q_start = q_base + h * head_dim;
                    let q_vec: Vec<f32> = (0..head_dim)
                        .map(|d| query.data[q_start + d].to_f32())
                        .collect();

                    // Only attend to positions within the sliding window
                    let mut scores = Vec::with_capacity(ctx_len - win_start);
                    let mut attended_positions = Vec::with_capacity(ctx_len - win_start);

                    for pos in win_start..ctx_len {
                        let block_idx = pos / block_size;
                        let block_off = pos % block_size;
                        if block_idx >= max_blocks_per_seq {
                            break;
                        }
                        let phys_block =
                            block_tables.data[seq_idx * max_blocks_per_seq + block_idx] as usize;
                        let k_base =
                            ((phys_block * block_size + block_off) * num_heads + h) * head_dim;

                        if k_base + head_dim > key_cache.data.len() {
                            break;
                        }

                        let dot: f32 = (0..head_dim)
                            .map(|d| q_vec[d] * key_cache.data[k_base + d].to_f32())
                            .sum();
                        scores.push(dot * scale);
                        attended_positions.push(pos);
                    }

                    if scores.is_empty() {
                        continue;
                    }

                    // Softmax over windowed scores
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Weighted sum of values within the window
                    let mut out_vec = vec![0.0f32; head_dim];
                    for (i, &pos) in attended_positions.iter().enumerate() {
                        let block_idx = pos / block_size;
                        let block_off = pos % block_size;
                        let phys_block =
                            block_tables.data[seq_idx * max_blocks_per_seq + block_idx] as usize;
                        let v_base =
                            ((phys_block * block_size + block_off) * num_heads + h) * head_dim;
                        let weight = exp_scores[i] / sum_exp;
                        for d in 0..head_dim {
                            out_vec[d] += weight * value_cache.data[v_base + d].to_f32();
                        }
                    }

                    let o_start = (token_offset + t) * num_heads * head_dim + h * head_dim;
                    for d in 0..head_dim {
                        output[o_start + d] = f16::from_f32(out_vec[d]);
                    }
                }
            }
            token_offset += seq_tokens;
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_tokens, num_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "SlidingWindowAttention"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f16_buf(data: Vec<f32>, shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer {
            data: data.into_iter().map(f16::from_f32).collect(),
            shape,
        }
    }

    fn make_i32_buf(data: Vec<i32>, shape: Vec<usize>) -> GpuBuffer<i32> {
        GpuBuffer { data, shape }
    }

    #[test]
    fn sliding_window_name() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        assert_eq!(sw.name(), "SlidingWindowAttention");
    }

    #[test]
    fn window_start_calculation() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        assert_eq!(sw.window_start(10), 6);
        assert_eq!(sw.window_start(4), 0);
        assert_eq!(sw.window_start(2), 0);
        assert_eq!(sw.window_start(0), 0);
    }

    #[test]
    fn active_block_range_calculation() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        // context_len=10, block_size=4 => window_start=6, first_block=1, last_block=3
        let (first, last) = sw.active_block_range(10, 4);
        assert_eq!(first, 1);
        assert_eq!(last, 3);
    }

    #[test]
    fn active_block_range_small_context() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(8));
        // context_len=4, block_size=4 => window_start=0, first_block=0, last_block=1
        let (first, last) = sw.active_block_range(4, 4);
        assert_eq!(first, 0);
        assert_eq!(last, 1);
    }

    #[test]
    fn active_block_range_empty() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        assert_eq!(sw.active_block_range(0, 4), (0, 0));
        assert_eq!(sw.active_block_range(4, 0), (0, 0));
    }

    #[test]
    fn evictable_blocks_returns_old_blocks() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        // context_len=12, block_size=4 => window_start=8, evictable_end=2
        let evictable = sw.evictable_blocks(12, 4);
        assert_eq!(evictable, vec![0, 1]);
    }

    #[test]
    fn evictable_blocks_none_when_context_fits_window() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(8));
        let evictable = sw.evictable_blocks(6, 4);
        assert!(evictable.is_empty());
    }

    #[test]
    fn config_global_applies_to_all_layers() {
        let cfg = SlidingWindowConfig::global(128);
        assert!(cfg.applies_to_layer(0));
        assert!(cfg.applies_to_layer(31));
        assert!(cfg.applies_to_layer(999));
    }

    #[test]
    fn config_per_layer_mask() {
        // Gemma 2 style: sliding window on even layers only
        let mask = vec![true, false, true, false, true, false];
        let cfg = SlidingWindowConfig::per_layer(4096, mask);
        assert!(cfg.applies_to_layer(0));
        assert!(!cfg.applies_to_layer(1));
        assert!(cfg.applies_to_layer(2));
        assert!(!cfg.applies_to_layer(3));
        // Out-of-range defaults to false
        assert!(!cfg.applies_to_layer(100));
    }

    #[test]
    fn forward_empty_batch() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        let query = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 2, 4],
        };
        let kc = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 4, 2, 4],
        };
        let vc = GpuBuffer {
            data: Vec::new(),
            shape: vec![0, 4, 2, 4],
        };
        let bt = make_i32_buf(Vec::new(), vec![0, 0]);
        let cl = make_i32_buf(Vec::new(), vec![0]);
        let out = sw.forward(&query, &kc, &vc, &bt, &cl, 0, 1.0).unwrap();
        assert!(out.data.is_empty());
    }

    #[test]
    fn forward_rejects_wrong_query_dims() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        let query = GpuBuffer {
            data: vec![f16::ZERO; 16],
            shape: vec![4, 4],
        };
        let kc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 1],
        };
        let vc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1, 1, 1],
        };
        let bt = make_i32_buf(vec![0], vec![1, 1]);
        let cl = make_i32_buf(vec![1], vec![1]);
        assert!(sw.forward(&query, &kc, &vc, &bt, &cl, 1, 1.0).is_err());
    }

    #[test]
    fn forward_rejects_wrong_cache_dims() {
        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(4));
        let query = GpuBuffer {
            data: vec![f16::ZERO; 8],
            shape: vec![1, 2, 4],
        };
        let kc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1], // 2-D instead of 4-D
        };
        let vc = GpuBuffer {
            data: Vec::new(),
            shape: vec![1, 1],
        };
        let bt = make_i32_buf(vec![0], vec![1, 1]);
        let cl = make_i32_buf(vec![1], vec![1]);
        assert!(sw.forward(&query, &kc, &vc, &bt, &cl, 1, 1.0).is_err());
    }

    #[test]
    fn forward_sliding_window_masks_old_positions() {
        // Setup: 1 sequence, 1 head, head_dim=2, block_size=2, context_len=6, window=4
        // This means we have 3 blocks (0,1,2). Positions 0-1 are in block 0,
        // positions 2-3 in block 1, positions 4-5 in block 2.
        // Window start = 6 - 4 = 2, so only positions 2-5 are attended.
        let num_heads = 1;
        let head_dim = 2;
        let block_size = 2;
        let ctx_len = 6;
        let window_size = 4;

        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(window_size));

        // Query: [1, 1, 2] -- one token
        let query = make_f16_buf(vec![1.0, 0.0], vec![1, num_heads, head_dim]);

        // Key cache: 3 blocks, block_size=2, 1 head, head_dim=2
        // Block 0 (positions 0,1): keys = [10.0, 0.0], [10.0, 0.0]  -- should be masked
        // Block 1 (positions 2,3): keys = [1.0, 0.0], [1.0, 0.0]
        // Block 2 (positions 4,5): keys = [1.0, 0.0], [1.0, 0.0]
        let key_data = vec![
            10.0, 0.0, 10.0, 0.0, // block 0 (outside window)
            1.0, 0.0, 1.0, 0.0, // block 1 (inside window)
            1.0, 0.0, 1.0, 0.0, // block 2 (inside window)
        ];
        let key_cache = make_f16_buf(key_data, vec![3, block_size, num_heads, head_dim]);

        // Value cache: same layout
        // Block 0: values = [100.0, 100.0], [100.0, 100.0]  -- should be masked
        // Block 1: values = [1.0, 1.0], [1.0, 1.0]
        // Block 2: values = [1.0, 1.0], [1.0, 1.0]
        let val_data = vec![
            100.0, 100.0, 100.0, 100.0, // block 0 (outside window)
            1.0, 1.0, 1.0, 1.0, // block 1 (inside window)
            1.0, 1.0, 1.0, 1.0, // block 2 (inside window)
        ];
        let value_cache = make_f16_buf(val_data, vec![3, block_size, num_heads, head_dim]);

        // Block tables: seq 0 uses blocks [0, 1, 2]
        let block_tables = make_i32_buf(vec![0, 1, 2], vec![1, 3]);
        let context_lens = make_i32_buf(vec![ctx_len as i32], vec![1]);

        let out = sw
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                ctx_len,
                1.0,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, 2]);
        // If sliding window works, block 0 is masked out and output should be
        // ~[1.0, 1.0] (uniform attention over 4 identical value vectors).
        // If it fails and attends to block 0, output would be skewed toward
        // [100.0, 100.0].
        for &v in &out.data {
            let f = v.to_f32();
            assert!(
                (f - 1.0).abs() < 0.1,
                "expected ~1.0 (window masked old blocks), got {f}"
            );
        }
    }

    #[test]
    fn forward_full_context_within_window() {
        // When context_len <= window_size, all positions are attended.
        let num_heads = 1;
        let head_dim = 2;
        let block_size = 2;
        let ctx_len = 4;
        let window_size = 8; // bigger than context

        let sw = SlidingWindowAttention::new(SlidingWindowConfig::global(window_size));

        let query = make_f16_buf(vec![1.0, 0.0], vec![1, num_heads, head_dim]);

        let key_data = vec![
            1.0, 0.0, 1.0, 0.0, // block 0
            1.0, 0.0, 1.0, 0.0, // block 1
        ];
        let key_cache = make_f16_buf(key_data, vec![2, block_size, num_heads, head_dim]);

        let val_data = vec![
            2.0, 3.0, 2.0, 3.0, // block 0
            2.0, 3.0, 2.0, 3.0, // block 1
        ];
        let value_cache = make_f16_buf(val_data, vec![2, block_size, num_heads, head_dim]);

        let block_tables = make_i32_buf(vec![0, 1], vec![1, 2]);
        let context_lens = make_i32_buf(vec![ctx_len as i32], vec![1]);

        let out = sw
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                ctx_len,
                1.0,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, 2]);
        // All values are [2.0, 3.0], uniform attention => output ~[2.0, 3.0]
        let v0 = out.data[0].to_f32();
        let v1 = out.data[1].to_f32();
        assert!((v0 - 2.0).abs() < 0.1, "expected ~2.0, got {v0}");
        assert!((v1 - 3.0).abs() < 0.1, "expected ~3.0, got {v1}");
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SlidingWindowAttention>();
    }
}
