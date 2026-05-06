//! rvllm-core: zero rvllm-* deps. Error model, ids, dtype, shape, config.
//!
//! Every other crate re-exports `RvllmError` and `Result` from here.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
// Round-19 P2: `RvllmError` carries a `std::backtrace::Backtrace` per
// variant (≥130 B by itself) plus the `Loader { err, ctx, bt }` bundle
// (~168 B), so clippy flags every `Result<_, RvllmError>` as
// `result_large_err`. The right fix is to box large variants, but
// every crate downstream already pattern-matches on `RvllmError` so
// that's a workspace-wide refactor — not in scope here. Suppress at
// the crate level with this note so future readers know the lint is
// intentionally muted, not forgotten. `cargo clippy -- -D warnings`
// now succeeds for `rvllm-core` and consumers.
#![allow(clippy::result_large_err)]
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
