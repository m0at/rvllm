//! rvllm-kernels: manifest-verified loader + kernel signature catalog.
//!
//! The SHA-pinned invariant is the point of this crate. Every
//! downstream call that touches a PTX or .so goes through
//! `KernelLoader`, which is only constructible from a `VerifiedManifest`.

pub mod gb10_dispatch;
pub mod loader;
pub mod manifest;
pub mod module;
pub mod sigs;

pub use gb10_dispatch::{Fp8GemvVariant, FP8_GEMV_PTX_STEM};
pub use loader::{KernelLoader, PtxBytes};
pub use manifest::{ArtifactEntry, KernelManifest, VerifiedManifest};
pub use module::{KernelFn, LoadedModule};
pub use sigs::{ArgKind, KernelSig, FUSED_KERNELS};
