use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::types::{BlockId, SequenceId, TokenId};

pub struct BlockManagerConfig {
    pub num_gpu_blocks: usize,
    pub num_cpu_blocks: usize,
    pub block_size: usize,
    pub watermark: f32,
    pub prefix_cache_blocks: usize,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            num_gpu_blocks: 4096,
            num_cpu_blocks: 512,
            block_size: 16,
            watermark: 0.04,
            prefix_cache_blocks: 0,
        }
    }
}

struct FreeListPool {
    free: VecDeque<BlockId>,
    total: usize,
}

impl FreeListPool {
    fn new(total: usize) -> Self {
        let mut free = VecDeque::with_capacity(total);
        for i in 0..total {
            free.push_back(BlockId(i as u32));
        }
        Self { free, total }
    }

    fn allocate(&mut self) -> Option<BlockId> {
        self.free.pop_front()
    }

    fn free(&mut self, block_id: BlockId) {
        self.free.push_back(block_id);
    }

    fn free_blocks(&self) -> usize {
        self.free.len()
    }

    fn total_blocks(&self) -> usize {
        self.total
    }
}

struct RefCounter {
    counts: Vec<u32>,
}

impl RefCounter {
    fn new(capacity: usize) -> Self {
        Self {
            counts: vec![0; capacity],
        }
    }

    #[inline]
    fn increment(&mut self, block_id: BlockId) {
        self.counts[block_id.0 as usize] += 1;
    }

    #[inline]
    fn decrement(&mut self, block_id: BlockId) -> u32 {
        let c = &mut self.counts[block_id.0 as usize];
        *c = c.saturating_sub(1);
        *c
    }

    #[inline]
    fn get(&self, block_id: BlockId) -> u32 {
        self.counts[block_id.0 as usize]
    }

    #[inline]
    fn clear(&mut self, block_id: BlockId) {
        self.counts[block_id.0 as usize] = 0;
    }
}

struct SeqSlots {
    id_to_slot: HashMap<SequenceId, u32>,
    free_list: VecDeque<u32>,
    next_slot: u32,
}

impl SeqSlots {
    fn new() -> Self {
        Self {
            id_to_slot: HashMap::new(),
            free_list: VecDeque::new(),
            next_slot: 0,
        }
    }

    fn get_or_create(&mut self, seq_id: SequenceId) -> u32 {
        if let Some(&slot) = self.id_to_slot.get(&seq_id) {
            return slot;
        }
        let slot = self.free_list.pop_front().unwrap_or_else(|| {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        });
        self.id_to_slot.insert(seq_id, slot);
        slot
    }

    #[inline]
    fn get(&self, seq_id: SequenceId) -> Option<u32> {
        self.id_to_slot.get(&seq_id).copied()
    }

    fn remove(&mut self, seq_id: SequenceId) -> Option<u32> {
        if let Some(slot) = self.id_to_slot.remove(&seq_id) {
            self.free_list.push_back(slot);
            Some(slot)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixHash(u64);

struct CachedBlock {
    block_id: BlockId,
    last_access: u64,
    ref_count: u32,
}

struct PrefixCache {
    block_size: usize,
    max_cached_blocks: usize,
    cache: HashMap<PrefixHash, CachedBlock>,
    block_to_hash: HashMap<BlockId, PrefixHash>,
    access_counter: u64,
}

impl PrefixCache {
    fn new(block_size: usize, max_cached_blocks: usize) -> Self {
        Self {
            block_size,
            max_cached_blocks,
            cache: HashMap::new(),
            block_to_hash: HashMap::new(),
            access_counter: 0,
        }
    }

    fn is_disabled(&self) -> bool {
        self.max_cached_blocks == 0
    }

    fn hash_prefix(&self, tokens: &[TokenId], block_idx: usize) -> Option<PrefixHash> {
        let end = (block_idx + 1) * self.block_size;
        if tokens.len() < end {
            return None;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens[..end].hash(&mut hasher);
        Some(PrefixHash(hasher.finish()))
    }

    fn lookup(&mut self, tokens: &[TokenId]) -> Vec<(usize, BlockId)> {
        if self.is_disabled() {
            return Vec::new();
        }
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

    fn count_hits(&self, tokens: &[TokenId]) -> usize {
        if self.is_disabled() {
            return 0;
        }
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

    fn insert(&mut self, tokens: &[TokenId], block_idx: usize, block_id: BlockId) -> bool {
        if self.is_disabled() {
            return false;
        }
        let hash = match self.hash_prefix(tokens, block_idx) {
            Some(h) => h,
            None => return false,
        };
        if self.cache.contains_key(&hash) {
            if let Some(entry) = self.cache.get_mut(&hash) {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
            }
            return false;
        }
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

    fn release(&mut self, block_ids: &[BlockId]) {
        for &bid in block_ids {
            if let Some(hash) = self.block_to_hash.get(&bid) {
                if let Some(entry) = self.cache.get_mut(hash) {
                    entry.ref_count = entry.ref_count.saturating_sub(1);
                }
            }
        }
    }

    fn evict_one(&mut self) -> Option<BlockId> {
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

    fn maybe_evict(&mut self) {
        while self.cache.len() >= self.max_cached_blocks {
            if self.evict_one().is_none() {
                break;
            }
        }
    }
}

#[inline]
fn ensure_slot<T>(vec: &mut Vec<Option<T>>, slot: u32) {
    let idx = slot as usize;
    if idx >= vec.len() {
        vec.resize_with(idx + 1, || None);
    }
}

pub struct BlockManager {
    gpu_pool: FreeListPool,
    cpu_pool: FreeListPool,
    block_size: usize,
    watermark: f32,
    seq_slots: SeqSlots,
    gpu_tables: Vec<Option<Vec<BlockId>>>,
    cpu_tables: Vec<Option<Vec<BlockId>>>,
    gpu_refs: RefCounter,
    cpu_refs: RefCounter,
    table_sent_len: Vec<Option<usize>>,
    prefix_cache: PrefixCache,
    prefix_blocks: Vec<Option<Vec<BlockId>>>,
}

impl BlockManager {
    pub fn new(config: BlockManagerConfig) -> Self {
        let gpu_cap = config.num_gpu_blocks;
        let cpu_cap = config.num_cpu_blocks;
        Self {
            gpu_pool: FreeListPool::new(gpu_cap),
            cpu_pool: FreeListPool::new(cpu_cap),
            block_size: config.block_size,
            watermark: config.watermark,
            seq_slots: SeqSlots::new(),
            gpu_tables: Vec::new(),
            cpu_tables: Vec::new(),
            gpu_refs: RefCounter::new(gpu_cap),
            cpu_refs: RefCounter::new(cpu_cap),
            table_sent_len: Vec::new(),
            prefix_cache: PrefixCache::new(config.block_size, config.prefix_cache_blocks),
            prefix_blocks: Vec::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    fn blocks_needed(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    pub fn usable_gpu_blocks(&self) -> usize {
        let total = self.gpu_pool.total_blocks();
        let reserved = ((total as f32) * self.watermark).ceil() as usize;
        self.gpu_pool.free_blocks().saturating_sub(reserved)
    }

    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        num_blocks <= self.usable_gpu_blocks()
    }

    pub fn can_allocate_tokens(
        &self,
        seq_id: SequenceId,
        total_tokens: usize,
        prompt_tokens: &[TokenId],
    ) -> bool {
        let needed = self.blocks_needed(total_tokens);
        let existing = self
            .seq_slots
            .get(seq_id)
            .and_then(|s| self.gpu_tables.get(s as usize))
            .and_then(|o| o.as_ref())
            .map(|t| t.len())
            .unwrap_or(0);
        let mut additional = needed.saturating_sub(existing);
        if existing == 0 {
            additional = additional.saturating_sub(self.prefix_cache.count_hits(prompt_tokens));
        }
        additional <= self.usable_gpu_blocks()
    }

    pub fn allocate(&mut self, seq_id: SequenceId, num_tokens: usize) -> Vec<BlockId> {
        self.allocate_with_prefix(seq_id, num_tokens, &[])
    }

    pub fn allocate_with_prefix(
        &mut self,
        seq_id: SequenceId,
        num_tokens: usize,
        prompt_tokens: &[TokenId],
    ) -> Vec<BlockId> {
        let needed = self.blocks_needed(num_tokens);
        let slot = self.seq_slots.get_or_create(seq_id);
        ensure_slot(&mut self.gpu_tables, slot);
        ensure_slot(&mut self.table_sent_len, slot);

        let table = self.gpu_tables[slot as usize].get_or_insert_with(Vec::new);
        let existing = table.len();

        if existing >= needed {
            return table.clone();
        }

        let mut cached_hits = Vec::new();
        if existing == 0 && !prompt_tokens.is_empty() {
            cached_hits = self.prefix_cache.lookup(prompt_tokens);
        }

        let mut prefix_block_ids = Vec::new();

        for block_idx in existing..needed {
            let cached = cached_hits
                .iter()
                .find(|(idx, _)| *idx == block_idx)
                .map(|(_, bid)| *bid);

            if let Some(block_id) = cached {
                table.push(block_id);
                self.gpu_refs.increment(block_id);
                prefix_block_ids.push(block_id);
            } else if let Some(block_id) = self.gpu_pool.allocate() {
                table.push(block_id);
                self.gpu_refs.increment(block_id);
            } else {
                break;
            }
        }

        if !prefix_block_ids.is_empty() {
            ensure_slot(&mut self.prefix_blocks, slot);
            self.prefix_blocks[slot as usize] = Some(prefix_block_ids);
        }

        table.clone()
    }

    pub fn allocate_incremental(&mut self, seq_id: SequenceId, new_tokens: usize) -> Vec<BlockId> {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        ensure_slot(&mut self.gpu_tables, slot);
        let table = self.gpu_tables[slot as usize].get_or_insert_with(Vec::new);
        let existing = table.len();

        let current_capacity = existing * self.block_size;
        let total_needed_tokens = current_capacity + new_tokens;
        let needed = (total_needed_tokens + self.block_size - 1) / self.block_size;

        if needed <= existing {
            return Vec::new();
        }

        let mut new_blocks = Vec::with_capacity(needed - existing);
        for _ in existing..needed {
            match self.gpu_pool.allocate() {
                Some(block_id) => {
                    self.gpu_refs.increment(block_id);
                    table.push(block_id);
                    new_blocks.push(block_id);
                }
                None => break,
            }
        }
        new_blocks
    }

    pub fn register_prefix(&mut self, seq_id: SequenceId, prompt_tokens: &[TokenId]) {
        if self.prefix_cache.is_disabled() {
            return;
        }
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return,
        };
        let table = match self.gpu_tables.get(slot as usize).and_then(|o| o.as_ref()) {
            Some(t) => t,
            None => return,
        };
        let num_full = prompt_tokens.len() / self.block_size;
        for block_idx in 0..num_full.min(table.len()) {
            if self
                .prefix_cache
                .insert(prompt_tokens, block_idx, table[block_idx])
            {
                self.gpu_refs.increment(table[block_idx]);
            }
        }
    }

    pub fn free(&mut self, seq_id: SequenceId) {
        let slot = match self.seq_slots.remove(seq_id) {
            Some(s) => s,
            None => return,
        };

        if let Some(prefix_bids) = self
            .prefix_blocks
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            self.prefix_cache.release(&prefix_bids);
        }

        if let Some(table) = self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            for &block_id in &table {
                let remaining = self.gpu_refs.decrement(block_id);
                if remaining == 0 {
                    self.gpu_pool.free(block_id);
                    self.gpu_refs.clear(block_id);
                }
            }
        }

        if let Some(table) = self
            .cpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            for &block_id in &table {
                let remaining = self.cpu_refs.decrement(block_id);
                if remaining == 0 {
                    self.cpu_pool.free(block_id);
                    self.cpu_refs.clear(block_id);
                }
            }
        }

        if let Some(sent) = self.table_sent_len.get_mut(slot as usize) {
            *sent = None;
        }
    }

    pub fn get_block_table(&self, seq_id: SequenceId) -> &[BlockId] {
        self.seq_slots
            .get(seq_id)
            .and_then(|s| self.gpu_tables.get(s as usize))
            .and_then(|o| o.as_ref())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn get_block_table_update(&self, seq_id: SequenceId) -> Option<Vec<BlockId>> {
        let slot = self.seq_slots.get(seq_id)?;
        let table = self.gpu_tables.get(slot as usize)?.as_ref()?;
        let sent_len = self.table_sent_len.get(slot as usize).and_then(|o| *o);

        match sent_len {
            Some(len) if len == table.len() => None,
            _ => Some(table.clone()),
        }
    }

    pub fn mark_table_sent(&mut self, seq_id: SequenceId) {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return,
        };
        ensure_slot(&mut self.table_sent_len, slot);
        let table_len = self
            .gpu_tables
            .get(slot as usize)
            .and_then(|o| o.as_ref())
            .map(|t| t.len())
            .unwrap_or(0);
        self.table_sent_len[slot as usize] = Some(table_len);
    }

    pub fn cow_if_needed(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let table = match self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.as_mut())
        {
            Some(t) => t,
            None => return Vec::new(),
        };
        let last_idx = match table.len().checked_sub(1) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let old_block_id = table[last_idx];
        if self.gpu_refs.get(old_block_id) <= 1 {
            return Vec::new();
        }

        match self.gpu_pool.allocate() {
            Some(new_block_id) => {
                self.gpu_refs.decrement(old_block_id);
                self.gpu_refs.increment(new_block_id);
                table[last_idx] = new_block_id;
                vec![(old_block_id, new_block_id)]
            }
            None => Vec::new(),
        }
    }

    pub fn swap_out(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let gpu_table = match self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut mapping = Vec::with_capacity(gpu_table.len());
        let mut cpu_table = Vec::with_capacity(gpu_table.len());

        for &gpu_block in &gpu_table {
            match self.cpu_pool.allocate() {
                Some(cpu_block) => {
                    mapping.push((gpu_block, cpu_block));
                    cpu_table.push(cpu_block);
                    self.cpu_refs.increment(cpu_block);

                    let remaining = self.gpu_refs.decrement(gpu_block);
                    if remaining == 0 {
                        self.gpu_pool.free(gpu_block);
                        self.gpu_refs.clear(gpu_block);
                    }
                }
                None => break,
            }
        }

        ensure_slot(&mut self.cpu_tables, slot);
        self.cpu_tables[slot as usize] = Some(cpu_table);

        if let Some(sent) = self.table_sent_len.get_mut(slot as usize) {
            *sent = None;
        }

        mapping
    }

    pub fn swap_in(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let cpu_table = match self
            .cpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut mapping = Vec::with_capacity(cpu_table.len());
        let mut gpu_table = Vec::with_capacity(cpu_table.len());

        for &cpu_block in &cpu_table {
            match self.gpu_pool.allocate() {
                Some(gpu_block) => {
                    mapping.push((cpu_block, gpu_block));
                    gpu_table.push(gpu_block);
                    self.gpu_refs.increment(gpu_block);

                    let remaining = self.cpu_refs.decrement(cpu_block);
                    if remaining == 0 {
                        self.cpu_pool.free(cpu_block);
                        self.cpu_refs.clear(cpu_block);
                    }
                }
                None => break,
            }
        }

        ensure_slot(&mut self.gpu_tables, slot);
        self.gpu_tables[slot as usize] = Some(gpu_table);

        mapping
    }

    pub fn can_swap_out(&self, seq_id: SequenceId) -> bool {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return false,
        };
        match self.gpu_tables.get(slot as usize).and_then(|o| o.as_ref()) {
            Some(t) => t.len() <= self.cpu_pool.free_blocks(),
            None => false,
        }
    }

    pub fn can_swap_in(&self, seq_id: SequenceId) -> bool {
        let slot = match self.seq_slots.get(seq_id) {
            Some(s) => s,
            None => return false,
        };
        match self.cpu_tables.get(slot as usize).and_then(|o| o.as_ref()) {
            Some(t) => t.len() <= self.usable_gpu_blocks(),
            None => false,
        }
    }

    pub fn above_watermark(&self) -> bool {
        let total = self.gpu_pool.total_blocks();
        let reserved = ((total as f32) * self.watermark).ceil() as usize;
        self.gpu_pool.free_blocks() > reserved
    }

    pub fn evict_prefix_block(&mut self) -> Option<BlockId> {
        let block_id = self.prefix_cache.evict_one()?;
        let remaining = self.gpu_refs.decrement(block_id);
        if remaining == 0 {
            self.gpu_pool.free(block_id);
            self.gpu_refs.clear(block_id);
        }
        Some(block_id)
    }

    pub fn num_free_gpu_blocks(&self) -> usize {
        self.gpu_pool.free_blocks()
    }

    pub fn num_free_cpu_blocks(&self) -> usize {
        self.cpu_pool.free_blocks()
    }

    pub fn num_total_gpu_blocks(&self) -> usize {
        self.gpu_pool.total_blocks()
    }

    pub fn num_total_cpu_blocks(&self) -> usize {
        self.cpu_pool.total_blocks()
    }

    pub fn has_seq(&self, seq_id: SequenceId) -> bool {
        self.seq_slots.get(seq_id).is_some()
    }
}

impl crate::scheduler::BlockManagerOps for BlockManager {
    fn allocate(&mut self, seq_id: SequenceId, num_tokens: usize) -> Vec<BlockId> {
        BlockManager::allocate(self, seq_id, num_tokens)
    }

    fn allocate_incremental(&mut self, seq_id: SequenceId, new_tokens: usize) -> Vec<BlockId> {
        BlockManager::allocate_incremental(self, seq_id, new_tokens)
    }

    fn free(&mut self, seq_id: SequenceId) {
        BlockManager::free(self, seq_id)
    }

    fn get_block_table(&self, seq_id: SequenceId) -> Option<&[BlockId]> {
        let table = BlockManager::get_block_table(self, seq_id);
        if table.is_empty() {
            None
        } else {
            Some(table)
        }
    }

    fn get_block_table_update(&self, seq_id: SequenceId) -> Option<Vec<BlockId>> {
        BlockManager::get_block_table_update(self, seq_id)
    }

    fn mark_table_sent(&mut self, seq_id: SequenceId) {
        BlockManager::mark_table_sent(self, seq_id)
    }

    fn cow_if_needed(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        BlockManager::cow_if_needed(self, seq_id)
    }

    fn swap_out(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        BlockManager::swap_out(self, seq_id)
    }

    fn swap_in(&mut self, seq_id: SequenceId) -> Vec<(BlockId, BlockId)> {
        BlockManager::swap_in(self, seq_id)
    }

    fn can_allocate(&self, num_blocks: usize) -> bool {
        BlockManager::can_allocate(self, num_blocks)
    }

    fn above_watermark(&self) -> bool {
        BlockManager::above_watermark(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(gpu: usize, cpu: usize) -> BlockManagerConfig {
        BlockManagerConfig {
            num_gpu_blocks: gpu,
            num_cpu_blocks: cpu,
            block_size: 16,
            watermark: 0.0,
            prefix_cache_blocks: 0,
        }
    }

    fn make_config_with_prefix(gpu: usize, cpu: usize, prefix: usize) -> BlockManagerConfig {
        BlockManagerConfig {
            num_gpu_blocks: gpu,
            num_cpu_blocks: cpu,
            block_size: 4,
            watermark: 0.0,
            prefix_cache_blocks: prefix,
        }
    }

    #[test]
    fn allocate_and_free() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let sid = SequenceId(1);

        let table = mgr.allocate(sid, 32);
        assert_eq!(table.len(), 2);
        assert_eq!(mgr.get_block_table(sid).len(), 2);

        mgr.free(sid);
        assert!(mgr.get_block_table(sid).is_empty());
        assert_eq!(mgr.num_free_gpu_blocks(), 10);
    }

    #[test]
    fn allocate_incremental() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let sid = SequenceId(1);

        // Allocate 16 tokens -> 1 block (block_size=16)
        let table = mgr.allocate(sid, 16);
        assert_eq!(table.len(), 1);

        // allocate_incremental adds new_tokens to existing capacity.
        // capacity=16, +1 -> total_needed=17 -> needs 2 blocks -> 1 new block
        let new = mgr.allocate_incremental(sid, 1);
        assert_eq!(new.len(), 1);
        assert_eq!(mgr.get_block_table(sid).len(), 2);

        // Now capacity=32. Adding 16 -> total_needed=48 -> needs 3 blocks -> 1 new
        let new = mgr.allocate_incremental(sid, 16);
        assert_eq!(new.len(), 1);
        assert_eq!(mgr.get_block_table(sid).len(), 3);
    }

    #[test]
    fn can_allocate() {
        let mgr = BlockManager::new(make_config(2, 2));
        assert!(mgr.can_allocate(2));
        assert!(!mgr.can_allocate(3));
    }

    #[test]
    fn block_table_update_tracking() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let sid = SequenceId(1);

        mgr.allocate(sid, 16);
        assert!(mgr.get_block_table_update(sid).is_some());

        mgr.mark_table_sent(sid);
        assert!(mgr.get_block_table_update(sid).is_none());

        mgr.allocate_incremental(sid, 16);
        assert!(mgr.get_block_table_update(sid).is_some());

        mgr.mark_table_sent(sid);
        assert!(mgr.get_block_table_update(sid).is_none());
    }

    #[test]
    fn swap_out_and_in() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let sid = SequenceId(1);

        mgr.allocate(sid, 32);
        assert_eq!(mgr.get_block_table(sid).len(), 2);

        let out_mapping = mgr.swap_out(sid);
        assert_eq!(out_mapping.len(), 2);
        assert!(mgr.get_block_table(sid).is_empty());

        let in_mapping = mgr.swap_in(sid);
        assert_eq!(in_mapping.len(), 2);
        assert_eq!(mgr.get_block_table(sid).len(), 2);
    }

    #[test]
    fn cow_when_shared() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let s1 = SequenceId(1);
        let s2 = SequenceId(2);

        mgr.allocate(s1, 16);
        let table1 = mgr.get_block_table(s1).to_vec();
        let block_id = table1[0];

        mgr.gpu_refs.increment(block_id);

        let slot2 = mgr.seq_slots.get_or_create(s2);
        ensure_slot(&mut mgr.gpu_tables, slot2);
        mgr.gpu_tables[slot2 as usize] = Some(vec![block_id]);

        let copies = mgr.cow_if_needed(s2);
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].0, block_id);
        assert_ne!(copies[0].1, block_id);
    }

    #[test]
    fn cow_not_needed_when_unshared() {
        let mut mgr = BlockManager::new(make_config(10, 10));
        let sid = SequenceId(1);
        mgr.allocate(sid, 16);

        let copies = mgr.cow_if_needed(sid);
        assert!(copies.is_empty());
    }

    #[test]
    fn watermark() {
        let config = BlockManagerConfig {
            num_gpu_blocks: 100,
            num_cpu_blocks: 10,
            block_size: 16,
            watermark: 0.04,
            prefix_cache_blocks: 0,
        };
        let mgr = BlockManager::new(config);
        assert_eq!(mgr.usable_gpu_blocks(), 96);
        assert!(mgr.above_watermark());
    }

    #[test]
    fn prefix_cache_hit() {
        let mut mgr = BlockManager::new(make_config_with_prefix(20, 10, 100));
        let s1 = SequenceId(1);
        let tokens: Vec<TokenId> = (0..12).collect();

        mgr.allocate_with_prefix(s1, 12, &tokens);
        assert_eq!(mgr.get_block_table(s1).len(), 3);

        mgr.register_prefix(s1, &tokens);
        mgr.free(s1);

        let s2 = SequenceId(2);
        let table = mgr.allocate_with_prefix(s2, 12, &tokens);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn slot_recycling() {
        let mut mgr = BlockManager::new(make_config(100, 10));

        for i in 0..5u64 {
            mgr.allocate(SequenceId(i), 16);
        }
        mgr.free(SequenceId(2));
        mgr.free(SequenceId(3));

        mgr.allocate(SequenceId(100), 16);
        mgr.allocate(SequenceId(101), 16);

        assert!(mgr.has_seq(SequenceId(100)));
        assert!(mgr.has_seq(SequenceId(101)));
    }
}
