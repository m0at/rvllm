// loc-budget-override: error taxonomy for every subsystem (Cuda/Cutlass/
// Attention/Loader/Config/Scheduler/Graph/Sampling/Io) lives here so every
// crate can share one error type without a dependency cycle. Splitting
// would force the enum across files and hide the taxonomy.
//! Typed, structured errors per `v3/specs/03-errors.md`.
//!
//! Rules enforced here:
//! - No `String` variants. No `Box<dyn Error>`. No `From<String>`.
//! - Every variant carries enough context to diagnose without re-reading logs.
//! - CUDA driver errors in compute paths panic with `CudaCtx`; CUDA errors
//!   during setup or external I/O return `RvllmError::Cuda`.

use std::backtrace::Backtrace;
use std::io;
use std::path::PathBuf;

use crate::dtype::DType;
use crate::ids::ReqId;

// ---------------------------------------------------------------------------
// Placeholder newtypes for identifiers owned by downstream crates.
// Real constructors live in the owning crate; these are used by the core
// error enum so every crate can share one error type without cycles.
// ---------------------------------------------------------------------------

/// sha-256 of a metadata-layout descriptor. Owned and computed by
/// `rvllm-metadata`; used here so `GraphError::CaptureMetadataMismatch`
/// can carry both the captured and replayed hashes without a crate cycle.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct MetaLayoutHash(pub [u8; 32]);

impl std::fmt::Debug for MetaLayoutHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetaLayoutHash(")?;
        for b in &self.0[..8] {
            write!(f, "{b:02x}")?;
        }
        write!(f, "…)")
    }
}

/// Identifier for a CUTLASS kernel schedule (mainloop or epilogue).
/// Owned by `rvllm-cutlass`; used here so the `EpilogueScheduleMismatch`
/// variant can name both sides of a pairing bug.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ScheduleId(pub u32);

// ---------------------------------------------------------------------------
// CUDA context carried by every CUDA-backed error variant.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub enum CudaErrorKind {
    AllocFailed,
    LaunchFailed,
    MemcpyFailed,
    StreamFailed,
    EventFailed,
    GraphFailed,
    ModuleLoadFailed,
    Other,
}

#[derive(Copy, Clone, Debug)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}

#[derive(Clone, Debug)]
pub struct CudaCtx {
    pub stream: u64,
    pub kernel: &'static str,
    pub launch: Option<Launch>,
    pub device: i32,
}

impl CudaCtx {
    pub const fn setup() -> Self {
        Self {
            stream: 0,
            kernel: "",
            launch: None,
            device: -1,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-subsystem context structs.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CutlassCtx {
    pub kernel: &'static str,
    pub stream: u64,
}

#[derive(Clone, Debug)]
pub struct AttnCtx {
    pub op: &'static str,
    pub stream: u64,
    pub num_seqs: u32,
    pub head_dim: u32,
}

#[derive(Clone, Debug)]
pub struct LoaderCtx {
    pub path: PathBuf,
    pub tensor: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SampleCtx {
    pub op: &'static str,
    pub stream: u64,
}

// ---------------------------------------------------------------------------
// Per-subsystem error enums.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum CutlassError {
    WorkspaceTooSmall {
        variant: u32,
        m: usize,
        n: usize,
        k: usize,
        needed: usize,
        given: usize,
    },
    EpilogueScheduleMismatch {
        variant: u32,
        mainloop: ScheduleId,
        epilogue: ScheduleId,
    },
    AutotuneCacheMiss {
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
    },
    KernelLaunchFailed {
        variant: u32,
        cuda: CudaErrorKind,
    },
}

#[derive(Debug)]
pub enum AttentionError {
    Fa3SoMissing { path: PathBuf },
    UnsupportedHeadDim { got: u32, required: u32 },
    GqaRatioInvalid { num_heads: u32, num_kv_heads: u32 },
    ContextExceedsBucket { context: u32, max: u32 },
}

#[derive(Debug)]
pub enum LoaderError {
    MissingTensor { name: String },
    ShapeMismatch { tensor: String, expected: Vec<usize>, got: Vec<usize> },
    DtypeMismatch { tensor: String, expected: DType, got: DType },
    Fp8MisScaled { tensor: String, clamp_ppm: f32 },
    Corrupt { detail: String },
}

#[derive(Debug)]
pub enum ConfigError {
    MissingHfField { name: &'static str, file: PathBuf },
    HfTypeMismatch { name: &'static str, expected: &'static str },
    MissingField { name: &'static str },
    InvalidField { name: &'static str, reason: String },
    UnknownEnvVar { name: String },
    Inconsistent { reasons: Vec<String> },
}

#[derive(Debug)]
pub enum SchedulerError {
    KvExhausted { needed_blocks: u32, free_blocks: u32 },
    BucketNotCaptured { num_seqs: u32 },
    QueueFull,
}

#[derive(Debug)]
pub enum GraphError {
    CaptureMetadataMismatch {
        captured: MetaLayoutHash,
        replay: MetaLayoutHash,
    },
    ReallocInsideCapture {
        allocator: &'static str,
        bytes: usize,
    },
    BucketMissing {
        padded_batch: u32,
    },
    ReplayFailed {
        cuda: CudaErrorKind,
        kernel_at_fault: Option<&'static str>,
    },
    CaptureFailed,
    InstantiateFailed,
}

#[derive(Debug)]
pub enum SamplingError {
    InvalidParams { reason: String },
    CudaFailed { cuda: CudaErrorKind },
}

#[derive(Debug)]
pub enum IoError {
    NotFound,
    PermissionDenied,
    Other,
}

impl From<&io::Error> for IoError {
    fn from(e: &io::Error) -> Self {
        match e.kind() {
            io::ErrorKind::NotFound => IoError::NotFound,
            io::ErrorKind::PermissionDenied => IoError::PermissionDenied,
            _ => IoError::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level error.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RvllmError {
    Cuda {
        kind: CudaErrorKind,
        op: &'static str,
        ctx: CudaCtx,
        bt: Backtrace,
    },
    Cutlass {
        err: CutlassError,
        ctx: CutlassCtx,
        bt: Backtrace,
    },
    Attention {
        err: AttentionError,
        ctx: AttnCtx,
        bt: Backtrace,
    },
    Loader {
        err: LoaderError,
        ctx: LoaderCtx,
        bt: Backtrace,
    },
    Config {
        err: ConfigError,
        field: &'static str,
    },
    Scheduler {
        err: SchedulerError,
        req_id: Option<ReqId>,
    },
    Graph {
        err: GraphError,
        bucket: u32,
        bt: Backtrace,
    },
    Sampling {
        err: SamplingError,
        ctx: SampleCtx,
    },
    Io {
        err: IoError,
        path: PathBuf,
        source: io::Error,
    },
}

/// Crate-wide result alias. Downstream crates reuse this.
pub type Result<T> = core::result::Result<T, RvllmError>;

// ---------------------------------------------------------------------------
// Display: walk subsystem → op → kernel → stream → launch.
// No `From<String>` / `From<&str>` — errors must be constructed with
// structured context.
// ---------------------------------------------------------------------------

impl std::fmt::Display for RvllmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use RvllmError::*;
        match self {
            Cuda { kind, op, ctx, .. } => {
                write!(
                    f,
                    "cuda: {op} ({kind:?}) kernel={:?} stream=0x{:x} device={}",
                    ctx.kernel, ctx.stream, ctx.device
                )?;
                if let Some(l) = ctx.launch {
                    write!(
                        f,
                        " grid={:?} block={:?} smem={}",
                        l.grid, l.block, l.smem
                    )?;
                }
                Ok(())
            }
            Cutlass { err, ctx, .. } => write!(
                f,
                "cutlass: {err:?} kernel={:?} stream=0x{:x}",
                ctx.kernel, ctx.stream
            ),
            Attention { err, ctx, .. } => write!(
                f,
                "attention: {err:?} op={:?} stream=0x{:x} num_seqs={} head_dim={}",
                ctx.op, ctx.stream, ctx.num_seqs, ctx.head_dim
            ),
            Loader { err, ctx, .. } => write!(
                f,
                "loader: {err:?} path={:?} tensor={:?}",
                ctx.path, ctx.tensor
            ),
            Config { err, field } => write!(f, "config: {err:?} field={field:?}"),
            Scheduler { err, req_id } => write!(f, "scheduler: {err:?} req={req_id:?}"),
            Graph { err, bucket, .. } => write!(f, "graph: {err:?} bucket={bucket}"),
            Sampling { err, ctx } => write!(
                f,
                "sampling: {err:?} op={:?} stream=0x{:x}",
                ctx.op, ctx.stream
            ),
            Io { err, path, source } => {
                write!(f, "io: {err:?} path={path:?} source={source}")
            }
        }
    }
}

impl std::error::Error for RvllmError {}

// ---------------------------------------------------------------------------
// Small constructors to keep call sites readable without hiding context.
// ---------------------------------------------------------------------------

impl RvllmError {
    pub fn cuda(op: &'static str, kind: CudaErrorKind, ctx: CudaCtx) -> Self {
        RvllmError::Cuda {
            kind,
            op,
            ctx,
            bt: Backtrace::capture(),
        }
    }

    pub fn cutlass(err: CutlassError, ctx: CutlassCtx) -> Self {
        RvllmError::Cutlass {
            err,
            ctx,
            bt: Backtrace::capture(),
        }
    }

    pub fn graph(err: GraphError, bucket: u32) -> Self {
        RvllmError::Graph {
            err,
            bucket,
            bt: Backtrace::capture(),
        }
    }

    pub fn config(err: ConfigError, field: &'static str) -> Self {
        RvllmError::Config { err, field }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_too_small_round_trips_display() {
        let e = RvllmError::cutlass(
            CutlassError::WorkspaceTooSmall {
                variant: 7,
                m: 1024,
                n: 4096,
                k: 8192,
                needed: 8 << 20,
                given: 2 << 20,
            },
            CutlassCtx {
                kernel: "fp8_gemm_v7",
                stream: 0xdeadbeef,
            },
        );
        let s = format!("{e}");
        assert!(s.contains("WorkspaceTooSmall"));
        assert!(s.contains("variant: 7"));
        assert!(s.contains("needed: 8388608"));
        assert!(s.contains("kernel=\"fp8_gemm_v7\""));
    }

    #[test]
    fn graph_carries_bucket_and_both_hashes() {
        let a = MetaLayoutHash([0xaa; 32]);
        let b = MetaLayoutHash([0xbb; 32]);
        let e = RvllmError::graph(
            GraphError::CaptureMetadataMismatch {
                captured: a,
                replay: b,
            },
            128,
        );
        let s = format!("{e}");
        assert!(s.contains("bucket=128"));
        assert!(s.contains("aaaaaaaa"));
        assert!(s.contains("bbbbbbbb"));
    }
}
