//! Prefix caching for KV cache block reuse across requests.
//!
//! Hashes prompt token prefixes at block boundaries to detect shared prefixes.
//! When two requests share a prefix, the second reuses the already-computed
//! KV cache blocks instead of recomputing attention. Eviction follows LRU
//! policy on blocks that have no active references.
//!
//! Integrates with [`BlockManager::mark_shared`] and the existing ref-counting
//! infrastructure for copy-on-write correctness.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use rvllm_core::prelude::{BlockId, TokenId};

/// Hash of a token prefix aligned to block boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixHash(u64);

/// A cached prefix block entry, tracking its physical block and LRU order.
#[derive(Debug, Clone)]
struct CachedBlock {
    block_id: BlockId,
    /// Monotonically increasing access counter for LRU eviction.
    last_access: u64,
    /// Number of active sequences referencing this cached block.
    ref_count: usize,
}

/// Prefix cache that maps token-prefix hashes to physical GPU block ids.
///
/// Each entry covers exactly one block's worth of tokens (block_size tokens).
/// The hash for block N includes all tokens from position 0 through
/// (N+1)*block_size - 1, ensuring that two sequences only share a block
/// if their entire prefix up to that point is identical.
pub struct PrefixCache {
    /// block_size tokens per block.
    block_size: usize,
    /// Maximum number of blocks to keep in the eviction pool.
    max_cached_blocks: usize,
    /// Hash -> cached block mapping.
    cache: HashMap<PrefixHash, CachedBlock>,
    /// Reverse mapping: block_id -> hash, for eviction cleanup.
    block_to_hash: HashMap<BlockId, PrefixHash>,
    /// Monotonic counter for LRU ordering.
    access_counter: u64,
}

impl PrefixCache {
    /// Create a new prefix cache.
    ///
    /// `max_cached_blocks` limits how many evictable blocks are retained.
    /// Active blocks (ref_count > 0) are never evicted regardless of limit.
    pub fn new(block_size: usize, max_cached_blocks: usize) -> Self {
        Self {
            block_size,
            max_cached_blocks,
            cache: HashMap::new(),
            block_to_hash: HashMap::new(),
            access_counter: 0,
        }
    }

    /// Compute the prefix hash for block index `block_idx` given the full
    /// token sequence. The hash covers tokens [0, (block_idx+1)*block_size).
    pub fn hash_prefix(&self, tokens: &[TokenId], block_idx: usize) -> Option<PrefixHash> {
        let end = (block_idx + 1) * self.block_size;
        if tokens.len() < end {
            return None;
        }
        let prefix = &tokens[..end];
        let hash = Self::compute_hash(prefix);
        Some(hash)
    }

    /// Look up cached blocks for a token sequence. Returns a vec of
    /// (block_index, block_id) pairs for all prefix blocks that hit the cache.
    /// Blocks are returned in order from block 0 onward; the first miss
    /// terminates the search (prefixes must be contiguous).
    pub fn lookup(&mut self, tokens: &[TokenId]) -> Vec<(usize, BlockId)> {
        let num_full_blocks = tokens.len() / self.block_size;
        let mut hits = Vec::new();

        for block_idx in 0..num_full_blocks {
            let hash = match self.hash_prefix(tokens, block_idx) {
                Some(h) => h,
                None => break,
            };
            match self.cache.get_mut(&hash) {
                Some(entry) => {
                    self.access_counter += 1;
                    entry.last_access = self.access_counter;
                    entry.ref_count += 1;
                    hits.push((block_idx, entry.block_id));
                }
                None => break,
            }
        }

        hits
    }

    /// Insert a computed prefix block into the cache.
    ///
    /// `tokens` is the full prompt, `block_idx` is which block was computed,
    /// and `block_id` is the physical GPU block holding the KV data.
    /// Returns true if newly inserted, false if already present.
    pub fn insert(
        &mut self,
        tokens: &[TokenId],
        block_idx: usize,
        block_id: BlockId,
    ) -> bool {
        let hash = match self.hash_prefix(tokens, block_idx) {
            Some(h) => h,
            None => return false,
        };

        if self.cache.contains_key(&hash) {
            // Already cached; just bump access time.
            if let Some(entry) = self.cache.get_mut(&hash) {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
            }
            return false;
        }

        // Evict if over capacity.
        self.maybe_evict();

        self.access_counter += 1;
        self.cache.insert(
            hash,
            CachedBlock {
                block_id,
                last_access: self.access_counter,
                ref_count: 0,
            },
        );
        self.block_to_hash.insert(block_id, hash);
        true
    }

    /// Decrement the reference count on cached prefix blocks for a sequence.
    /// Called when a sequence finishes or is freed. `block_ids` are the
    /// physical block ids that were obtained via `lookup`.
    pub fn release(&mut self, block_ids: &[BlockId]) {
        for &bid in block_ids {
            if let Some(hash) = self.block_to_hash.get(&bid) {
                if let Some(entry) = self.cache.get_mut(hash) {
                    entry.ref_count = entry.ref_count.saturating_sub(1);
                }
            }
        }
    }

    /// Evict the least-recently-used block that has no active references.
    /// Returns the evicted block id if one was evicted.
    pub fn evict_one(&mut self) -> Option<BlockId> {
        let evictable = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.ref_count == 0)
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(hash, entry)| (*hash, entry.block_id));

        if let Some((hash, block_id)) = evictable {
            self.cache.remove(&hash);
            self.block_to_hash.remove(&block_id);
            Some(block_id)
        } else {
            None
        }
    }

    /// Count how many contiguous prefix blocks would hit the cache,
    /// without modifying any state. Used by `can_allocate` which takes `&self`.
    pub fn count_hits(&self, tokens: &[TokenId]) -> usize {
        let num_full_blocks = tokens.len() / self.block_size;
        let mut count = 0;
        for block_idx in 0..num_full_blocks {
            let hash = match self.hash_prefix(tokens, block_idx) {
                Some(h) => h,
                None => break,
            };
            if self.cache.contains_key(&hash) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Number of blocks currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// True if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Number of evictable blocks (ref_count == 0).
    pub fn num_evictable(&self) -> usize {
        self.cache.values().filter(|e| e.ref_count == 0).count()
    }

    /// Number of blocks actively referenced.
    pub fn num_active(&self) -> usize {
        self.cache.values().filter(|e| e.ref_count > 0).count()
    }

    /// Check if a block id is managed by this prefix cache.
    pub fn contains_block(&self, block_id: BlockId) -> bool {
        self.block_to_hash.contains_key(&block_id)
    }

    /// Get all block ids in the cache (for diagnostics).
    pub fn cached_block_ids(&self) -> Vec<BlockId> {
        self.cache.values().map(|e| e.block_id).collect()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn compute_hash(tokens: &[TokenId]) -> PrefixHash {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        PrefixHash(hasher.finish())
    }

    /// Evict LRU unreferenced entries until we are at or below capacity.
    fn maybe_evict(&mut self) {
        while self.cache.len() >= self.max_cached_blocks {
            if self.evict_one().is_none() {
                // All entries are actively referenced; can't evict.
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integration helpers for BlockManager
// ---------------------------------------------------------------------------

/// Compute the number of prefix blocks that can be reused for a sequence.
///
/// Given a prompt's token ids and a `PrefixCache`, returns how many leading
/// blocks hit the cache. The caller should skip allocating and computing
/// KV for those blocks.
pub fn prefix_hit_count(cache: &mut PrefixCache, tokens: &[TokenId]) -> usize {
    cache.lookup(tokens).len()
}

/// After a prefill, register all fully-computed prefix blocks in the cache.
/// `tokens` is the full prompt, `block_ids` are the physical blocks in order.
/// Only full blocks (not the last partial block) are registered.
pub fn register_prefix_blocks(
    cache: &mut PrefixCache,
    tokens: &[TokenId],
    block_ids: &[BlockId],
    block_size: usize,
) -> Vec<BlockId> {
    let num_full = tokens.len() / block_size;
    let mut newly_cached = Vec::new();
    for block_idx in 0..num_full.min(block_ids.len()) {
        if cache.insert(tokens, block_idx, block_ids[block_idx]) {
            newly_cached.push(block_ids[block_idx]);
        }
    }
    newly_cached
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(n: usize) -> Vec<TokenId> {
        (0..n as u32).collect()
    }

    #[test]
    fn hash_prefix_basic() {
        let cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(16);

        let h0 = cache.hash_prefix(&tokens, 0);
        assert!(h0.is_some());

        let h1 = cache.hash_prefix(&tokens, 1);
        assert!(h1.is_some());
        assert_ne!(h0, h1);

        // Block 4 needs 20 tokens, we only have 16.
        let h4 = cache.hash_prefix(&tokens, 4);
        assert!(h4.is_none());
    }

    #[test]
    fn hash_deterministic() {
        let cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(8);
        let h1 = cache.hash_prefix(&tokens, 0).unwrap();
        let h2 = cache.hash_prefix(&tokens, 0).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn shared_prefix_same_hash() {
        let cache = PrefixCache::new(4, 100);
        let tokens_a = make_tokens(12);
        let mut tokens_b = make_tokens(12);
        // Same first 8 tokens, different last 4.
        tokens_b[8] = 999;
        tokens_b[9] = 998;

        // Block 0 covers [0..4), same in both.
        let ha = cache.hash_prefix(&tokens_a, 0).unwrap();
        let hb = cache.hash_prefix(&tokens_b, 0).unwrap();
        assert_eq!(ha, hb);

        // Block 1 covers [0..8), same in both.
        let ha1 = cache.hash_prefix(&tokens_a, 1).unwrap();
        let hb1 = cache.hash_prefix(&tokens_b, 1).unwrap();
        assert_eq!(ha1, hb1);

        // Block 2 covers [0..12), different.
        let ha2 = cache.hash_prefix(&tokens_a, 2).unwrap();
        let hb2 = cache.hash_prefix(&tokens_b, 2).unwrap();
        assert_ne!(ha2, hb2);
    }

    #[test]
    fn insert_and_lookup() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(12);

        // Insert blocks 0, 1, 2.
        cache.insert(&tokens, 0, BlockId(10));
        cache.insert(&tokens, 1, BlockId(11));
        cache.insert(&tokens, 2, BlockId(12));
        assert_eq!(cache.len(), 3);

        // Lookup with same prefix.
        let hits = cache.lookup(&tokens);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0], (0, BlockId(10)));
        assert_eq!(hits[1], (1, BlockId(11)));
        assert_eq!(hits[2], (2, BlockId(12)));
    }

    #[test]
    fn lookup_partial_match() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens_a = make_tokens(12);

        // Insert blocks for tokens_a.
        cache.insert(&tokens_a, 0, BlockId(10));
        cache.insert(&tokens_a, 1, BlockId(11));
        cache.insert(&tokens_a, 2, BlockId(12));

        // Lookup with different suffix -- first 8 tokens same, last 4 differ.
        let mut tokens_b = make_tokens(12);
        tokens_b[8] = 999;
        let hits = cache.lookup(&tokens_b);
        // Blocks 0 and 1 match (cover [0..8)), block 2 doesn't.
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].1, BlockId(10));
        assert_eq!(hits[1].1, BlockId(11));
    }

    #[test]
    fn lookup_stops_at_first_miss() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(12);

        // Only insert block 0 and block 2 (skip block 1).
        cache.insert(&tokens, 0, BlockId(10));
        cache.insert(&tokens, 2, BlockId(12));

        let hits = cache.lookup(&tokens);
        // Should stop after block 0 since block 1 is missing.
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1, BlockId(10));
    }

    #[test]
    fn evict_lru() {
        let mut cache = PrefixCache::new(4, 3);
        let tokens = make_tokens(16);

        cache.insert(&tokens, 0, BlockId(10));
        cache.insert(&tokens, 1, BlockId(11));
        cache.insert(&tokens, 2, BlockId(12));

        // After insert, ref_count=0 so they're already evictable.
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.num_evictable(), 3);

        // Insert a 4th block triggers eviction of the LRU (block 10).
        cache.insert(&tokens, 3, BlockId(13));
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_block(BlockId(10)));
        assert!(cache.contains_block(BlockId(13)));
    }

    #[test]
    fn evict_skips_active_refs() {
        let mut cache = PrefixCache::new(4, 2);
        let tokens = make_tokens(12);

        cache.insert(&tokens, 0, BlockId(10));
        cache.insert(&tokens, 1, BlockId(11));

        // Simulate active use via lookup (bumps ref_count).
        let _hits = cache.lookup(&tokens);
        assert_eq!(cache.num_evictable(), 0);

        // Trying to insert a 3rd block can't evict, so cache grows.
        cache.insert(&tokens, 2, BlockId(12));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn release_decrements_refcount() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(8);

        cache.insert(&tokens, 0, BlockId(10));
        cache.insert(&tokens, 1, BlockId(11));
        // After insert, ref_count is 0 (evictable). Simulate active use.
        let _hits = cache.lookup(&tokens);
        assert_eq!(cache.num_active(), 2);
        assert_eq!(cache.num_evictable(), 0);

        cache.release(&[BlockId(10)]);
        assert_eq!(cache.num_active(), 1);
        assert_eq!(cache.num_evictable(), 1);

        cache.release(&[BlockId(11)]);
        assert_eq!(cache.num_evictable(), 2);
    }

    #[test]
    fn duplicate_insert_no_double_entry() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(8);

        assert!(cache.insert(&tokens, 0, BlockId(10)));
        assert!(!cache.insert(&tokens, 0, BlockId(10)));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn register_prefix_blocks_helper() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(10); // 2 full blocks of 4, plus 2 leftover.
        let block_ids = vec![BlockId(1), BlockId(2), BlockId(3)];

        let newly = register_prefix_blocks(&mut cache, &tokens, &block_ids, 4);
        assert_eq!(newly.len(), 2); // Only 2 full blocks.
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn prefix_hit_count_helper() {
        let mut cache = PrefixCache::new(4, 100);
        let tokens = make_tokens(12);

        cache.insert(&tokens, 0, BlockId(1));
        cache.insert(&tokens, 1, BlockId(2));
        // Release so lookup re-increments.
        cache.release(&[BlockId(1), BlockId(2)]);

        let count = prefix_hit_count(&mut cache, &tokens);
        assert_eq!(count, 2);
    }

    #[test]
    fn empty_cache() {
        let mut cache = PrefixCache::new(4, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.evict_one(), None);

        let hits = cache.lookup(&make_tokens(8));
        assert!(hits.is_empty());
    }
}
