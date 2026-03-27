use crate::PhysicalBlock;

/// Per-sequence mapping of logical block index to physical block.
#[derive(Debug, Clone)]
pub struct BlockTable {
    blocks: Vec<PhysicalBlock>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(cap),
        }
    }

    pub fn get(&self, logical_idx: usize) -> Option<&PhysicalBlock> {
        self.blocks.get(logical_idx)
    }

    pub fn get_mut(&mut self, logical_idx: usize) -> Option<&mut PhysicalBlock> {
        self.blocks.get_mut(logical_idx)
    }

    pub fn push(&mut self, block: PhysicalBlock) {
        self.blocks.push(block);
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PhysicalBlock> {
        self.blocks.iter()
    }

    pub fn last(&self) -> Option<&PhysicalBlock> {
        self.blocks.last()
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
    }
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use rvllm_core::prelude::BlockId;

    #[test]
    fn empty_table() {
        let table = BlockTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.get(0).is_none());
    }

    #[test]
    fn push_and_get() {
        let mut table = BlockTable::new();
        let block = PhysicalBlock::new(BlockId(0), 16, Device::Gpu);
        table.push(block);
        assert_eq!(table.len(), 1);
        let b = table.get(0).unwrap();
        assert_eq!(b.block_id, BlockId(0));
    }

    #[test]
    fn iteration() {
        let mut table = BlockTable::new();
        for i in 0..5 {
            table.push(PhysicalBlock::new(BlockId(i), 16, Device::Gpu));
        }
        let ids: Vec<_> = table.iter().map(|b| b.block_id).collect();
        assert_eq!(
            ids,
            vec![BlockId(0), BlockId(1), BlockId(2), BlockId(3), BlockId(4)]
        );
    }
}
