//! rvllm-mem: HBM arena, tensor views, stream, event, capture scope,
//! pinned pool, KV layout. Per `v3/specs/04-memory.md` + `05-concurrency.md`.
//!
//! The invariants this crate carries:
//! - Arena is fixed-size and non-relocating; once allocated, device
//!   pointers are stable for the arena's lifetime.
//! - `Region`, `Tensor`, and `Stream` are `!Send` `!Sync` (a worker's
//!   resources are pinned to its thread).
//! - `CaptureScope` only accepts `&T where T: GraphSafe` borrows, so
//!   realloc-capable allocators cannot enter a captured region.
//! - CUDA-handle-owning types implement `CudaOwned` so Drop fences the
//!   stream before destroying the handle (re-derived from v2 7c212c13c).
//!
//! CUDA FFI is gated on `feature = "cuda"`. Without the feature, the
//! crate compiles with host stubs so invariant-level tests run on any
//! machine.

pub mod capture;
pub mod context;
pub mod cuda_owned;
pub mod event;
pub mod graph_safe;
pub mod hbm;
pub mod kv_layout;
pub mod pinned;
pub mod stream;
pub mod tensor;

pub use capture::{record, BoundHandle, CaptureScope, HasDevicePtr};
pub use context::CudaContextHandle;
pub use cuda_owned::CudaOwned;
pub use event::Event;
pub use graph_safe::GraphSafe;
pub use hbm::{HbmArena, Region};
pub use kv_layout::KvLayout;
pub use pinned::{PinnedBuf, PinnedPool};
pub use stream::Stream;
pub use tensor::Tensor;
