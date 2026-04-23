//! rvllm-core: zero rvllm-* deps. Error model, ids, dtype, shape, config.
//!
//! Every other crate re-exports `RvllmError` and `Result` from here.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
// `panic!` is allowed only in tests and in builder-validation paths that
// are explicitly documented as invariant-violating. Everywhere else,
// errors flow through `Result<T, RvllmError>`.

pub mod arch;
pub mod config;
pub mod dtype;
pub mod env;
pub mod error;
pub mod ids;
pub mod shape;

pub use arch::CompileTarget;
pub use config::{
    GraphMode, LogLevel, ModelArch, ModelConfig, PreemptionMode, RuntimeConfig,
    RuntimeConfigBuilder,
};
pub use dtype::DType;
pub use error::{
    AttentionError, AttnCtx, ConfigError, CudaCtx, CudaErrorKind, CutlassCtx, CutlassError,
    GraphError, IoError, Launch, LoaderCtx, LoaderError, MetaLayoutHash, Result, RvllmError,
    SampleCtx, SamplingError, ScheduleId, SchedulerError,
};
pub use ids::{BlockId, ReqId, SeqId, TokenId};
pub use shape::{Shape, MAX_RANK};
