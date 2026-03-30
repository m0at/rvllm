use crate::{LLMError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpBackend {
    Nccl,
}

#[derive(Debug, Clone)]
pub struct TpConfig {
    pub world_size: usize,
    pub rank: usize,
    pub backend: TpBackend,
}

impl TpConfig {
    pub fn new(world_size: usize, rank: usize, backend: TpBackend) -> Result<Self> {
        if world_size == 0 || !world_size.is_power_of_two() {
            return Err(LLMError::ConfigError(format!(
                "TP world_size must be a power of 2, got {}",
                world_size
            )));
        }
        if rank >= world_size {
            return Err(LLMError::ConfigError(format!(
                "TP rank {} >= world_size {}",
                rank, world_size
            )));
        }
        Ok(Self {
            world_size,
            rank,
            backend,
        })
    }

    pub fn single_gpu() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            backend: TpBackend::Nccl,
        }
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_configs() {
        for ws in [1, 2, 4, 8] {
            for r in 0..ws {
                assert!(TpConfig::new(ws, r, TpBackend::Nccl).is_ok());
            }
        }
    }

    #[test]
    fn invalid_world_size() {
        assert!(TpConfig::new(0, 0, TpBackend::Nccl).is_err());
        assert!(TpConfig::new(3, 0, TpBackend::Nccl).is_err());
        assert!(TpConfig::new(6, 0, TpBackend::Nccl).is_err());
    }

    #[test]
    fn rank_out_of_bounds() {
        assert!(TpConfig::new(4, 4, TpBackend::Nccl).is_err());
        assert!(TpConfig::new(4, 10, TpBackend::Nccl).is_err());
    }

    #[test]
    fn single_gpu_not_distributed() {
        assert!(!TpConfig::single_gpu().is_distributed());
    }
}
