//! GPU compile-target enumeration.
//!
//! Every kernel artifact is pinned to exactly one compute capability —
//! PTX built for `sm_90` does not execute on `sm_121` and vice versa.
//! The runtime queries the device's compute capability via
//! `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR})`,
//! maps it to a `CompileTarget`, and selects the matching kernel directory
//! under `RVLLM_KERNEL_DIR/<arch>/*.ptx`. A device whose compute capability
//! is not listed here is rejected at `bring_up` time (no silent fallback).
//!
//! Intentionally additive: the existing SM90 / H100+H200 pipeline keeps
//! working unchanged; new targets slot in as separate variants with their
//! own kernel tree and optional build features.

use serde::{Deserialize, Serialize};

/// Supported GPU compile targets.
///
/// Add a new variant when we need to support a new architecture; update
/// `from_compute_capability`, `as_sm_str`, and the build system's kernel
/// output directories in the same PR.
///
/// `#[non_exhaustive]` so future arches (sm_100, sm_120, sm_122, …) can
/// join without breaking downstream `match` expressions. Internal
/// matches inside this crate stay exhaustive by design.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[must_use]
pub enum CompileTarget {
    /// Ampere A100/A800 — original baseline.
    Sm80,
    /// Ada / RTX 4090 / L40S (SM89 attn + cuBLASLt-only path; see SUPERDEPLOY.md).
    Sm89,
    /// Hopper H100/H200 — primary production target (FA3 WGMMA+TMA, CUTLASS SM90).
    Sm90,
    /// Blackwell GB10 aka "Project DIGITS" aka DGX Spark — Grace+Blackwell consumer.
    /// Distinct from sm_120 (RTX 5090) and sm_122 (RTX 5080): same Blackwell ISA
    /// but different unified-memory (LPDDR5X) + firmware power-cap profile.
    Sm121,
}

impl CompileTarget {
    /// Map a compute-capability tuple to a compile target.
    ///
    /// Returns `None` for compute capabilities we do not yet build PTX for.
    /// The caller is responsible for turning that `None` into a hard error
    /// (the runtime refuses to boot on an unsupported device rather than
    /// falling back to a generic path).
    #[inline]
    #[must_use]
    pub const fn from_compute_capability(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (8, 0) => Some(CompileTarget::Sm80),
            (8, 9) => Some(CompileTarget::Sm89),
            (9, 0) => Some(CompileTarget::Sm90),
            (12, 1) => Some(CompileTarget::Sm121),
            _ => None,
        }
    }

    /// The `sm_XYZ` string as accepted by `nvcc -arch=` and used as the
    /// kernel subdirectory name (e.g. `kernels/sm_121/fp8_gemv.ptx`).
    #[inline]
    #[must_use]
    pub const fn as_sm_str(self) -> &'static str {
        match self {
            CompileTarget::Sm80 => "sm_80",
            CompileTarget::Sm89 => "sm_89",
            CompileTarget::Sm90 => "sm_90",
            CompileTarget::Sm121 => "sm_121",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_str_matches_nvcc_flag() {
        assert_eq!(CompileTarget::Sm80.as_sm_str(), "sm_80");
        assert_eq!(CompileTarget::Sm89.as_sm_str(), "sm_89");
        assert_eq!(CompileTarget::Sm90.as_sm_str(), "sm_90");
        assert_eq!(CompileTarget::Sm121.as_sm_str(), "sm_121");
    }

    #[test]
    fn compute_cap_to_target() {
        assert_eq!(
            CompileTarget::from_compute_capability(9, 0),
            Some(CompileTarget::Sm90),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(12, 1),
            Some(CompileTarget::Sm121),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(8, 0),
            Some(CompileTarget::Sm80),
        );
    }

    #[test]
    fn unknown_cc_returns_none() {
        // cc 12.0 (RTX 5090, sm_120) — not currently supported at this layer.
        assert_eq!(CompileTarget::from_compute_capability(12, 0), None);
        // cc 12.2 (RTX 5080, sm_122) — also not yet.
        assert_eq!(CompileTarget::from_compute_capability(12, 2), None);
        // Future Hopper revision.
        assert_eq!(CompileTarget::from_compute_capability(9, 5), None);
    }

    #[test]
    fn sm121_is_distinct_from_sm120_and_sm122() {
        // Guard against accidental conflation: Blackwell data-center (sm_100),
        // consumer (sm_120/sm_122), and Grace+Blackwell (sm_121) share the ISA
        // family but have different memory / power profiles. We only model
        // sm_121 here for now; sm_120/sm_122 stay `None` until someone needs
        // them.
        assert_ne!(
            CompileTarget::from_compute_capability(12, 1),
            CompileTarget::from_compute_capability(12, 0),
        );
    }
}
