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

pub use gb10_dispatch::{
    regime_from_clock_mhz, regime_from_elapsed, select_variant, ClockRegime, Fp8GemvVariant,
    FP8_GEMV_PTX_STEM, SUSTAINED_CLOCK_THRESHOLD_MHZ, THROTTLE_ONSET,
};
pub use loader::{KernelLoader, PtxBytes};
pub use manifest::{ArtifactEntry, KernelManifest, VerifiedManifest};
pub use module::{KernelFn, LoadedModule};
pub use sigs::{ArgKind, KernelSig, FUSED_KERNELS};
