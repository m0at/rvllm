#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod clients;
pub mod config;
pub mod core;
pub mod environments;
pub mod error;
pub mod logger;
pub mod serve;
pub mod types;
pub mod utils;

pub use clients::{LanguageModel, StubLanguageModel};
#[cfg(feature = "cuda")]
pub use clients::{RvllmCudaClient, RvllmCudaConfig};
pub use config::RlmConfig;
pub use core::{Rlm, RlmBuilder, RLM};
pub use environments::{
    Environment, ExecutionCallbacks, FnTool, HostCall, LocalEnvironment, Tool, ToolRegistry,
};
pub use error::{Result, RlmError};
pub use logger::{TrajectoryLogger, TrajectorySnapshot};
pub use serve::{ServeRequest, ServeResponse, ServeService};
pub use types::{
    BackendKind, ChatCompletion, CodeBlock, EnvironmentKind, ModelUsageSummary, PerplexitySummary,
    Prompt, QueryMetadata, ReplResult, RlmChatCompletion, RlmIteration, RlmMetadata, UsageSummary,
};
