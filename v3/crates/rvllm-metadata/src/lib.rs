//! rvllm-metadata: the ONE metadata upload path per spec 08.
//!
//! The invariant this crate carries:
//! - `MetadataLayout` is keyed on `(bucket, max_blocks_per_seq)`,
//!   computed once at engine init, byte-identical across replays.
//! - `upload()` is the only function that writes the packed buffer.
//!   There is no `patch`, no non-padded variant, no CoW-only fast
//!   path. The April-16 crash class is unrepresentable.
//! - `MetadataLayout::hash()` gives a sha256 that `rvllm-graph` stores
//!   at capture time and verifies on every replay; drift → typed err.

pub mod layout;
pub mod pack;
pub mod plan;

pub use layout::MetadataLayout;
pub use pack::upload;
pub use plan::BatchPlan;
