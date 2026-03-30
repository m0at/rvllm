pub mod comm;
pub mod config;
pub mod nccl;
pub mod parallel_layer;
pub mod shard;

pub use comm::TpCommunicator;
pub use config::{TpBackend, TpConfig};
pub use nccl::NcclTpCommunicator;
pub use parallel_layer::TpLinearLayer;
pub use shard::{shard_weight, ShardStrategy};

pub use rvllm_core::prelude::{LLMError, Result};
