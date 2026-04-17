//! rvllm-kernels: manifest-verified loader + kernel signature catalog.
//!
//! The SHA-pinned invariant is the point of this crate. Every
//! downstream call that touches a PTX or .so goes through
//! `KernelLoader`, which is only constructible from a `VerifiedManifest`.

pub mod loader;
pub mod manifest;
pub mod sigs;

pub use loader::{KernelLoader, PtxBytes};
pub use manifest::{ArtifactEntry, KernelManifest, VerifiedManifest};
pub use sigs::{ArgKind, KernelSig, FUSED_KERNELS};
