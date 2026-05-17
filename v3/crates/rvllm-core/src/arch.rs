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

use core::fmt;

use serde::{Deserialize, Serialize};

/// Kernel manifest file name under each per-target kernel directory.
pub const KERNEL_MANIFEST_FILE: &str = "manifest.json";

/// Architecture-gated GPU features.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[must_use]
pub enum ArchFeature {
    Fp8TensorCores,
    W4a8Cutlass,
    Fa3,
    Fa2Ptx,
    RotorQuantKv,
}

impl ArchFeature {
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            ArchFeature::Fp8TensorCores => "FP8 tensor cores",
            ArchFeature::W4a8Cutlass => "W4A8 CUTLASS",
            ArchFeature::Fa3 => "FA3",
            ArchFeature::Fa2Ptx => "FA2 PTX",
            ArchFeature::RotorQuantKv => "RotorQuant KV",
        }
    }
}

impl fmt::Display for ArchFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when an arch-gated feature is requested on an unsupported target.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[must_use]
pub struct UnsupportedArchFeature {
    pub target: CompileTarget,
    pub feature: ArchFeature,
    pub reason: &'static str,
}

impl fmt::Display for UnsupportedArchFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} does not support {}: {}",
            self.target.as_sm_str(),
            self.feature,
            self.reason
        )
    }
}

impl std::error::Error for UnsupportedArchFeature {}

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
    /// Turing T4 / RTX 6000 — compatibility-only target until kernels prove out.
    Sm75,
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
            (7, 5) => Some(CompileTarget::Sm75),
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
            CompileTarget::Sm75 => "sm_75",
            CompileTarget::Sm80 => "sm_80",
            CompileTarget::Sm89 => "sm_89",
            CompileTarget::Sm90 => "sm_90",
            CompileTarget::Sm121 => "sm_121",
        }
    }

    /// Manifest arch string. Matches the `arch` field in `manifest.json`.
    #[inline]
    #[must_use]
    pub const fn manifest_arch(self) -> &'static str {
        self.as_sm_str()
    }

    /// Components for `<kernels_root>/<sm_xx>/manifest.json`.
    #[inline]
    #[must_use]
    pub const fn manifest_path_components(self) -> (&'static str, &'static str) {
        (self.as_sm_str(), KERNEL_MANIFEST_FILE)
    }

    /// Native FP8 tensor-core MMA is available on Ada, Hopper, and Blackwell.
    #[inline]
    #[must_use]
    pub const fn supports_fp8_tensor_cores(self) -> bool {
        matches!(
            self,
            CompileTarget::Sm89 | CompileTarget::Sm90 | CompileTarget::Sm121
        )
    }

    /// rvLLM's AWQ/W4A8 CUTLASS path is currently the H100-verified SM90 .so.
    #[inline]
    #[must_use]
    pub const fn supports_w4a8_cutlass(self) -> bool {
        matches!(self, CompileTarget::Sm90)
    }

    /// FlashAttention-3 requires Hopper WGMMA/TMA.
    #[inline]
    #[must_use]
    pub const fn supports_fa3(self) -> bool {
        matches!(self, CompileTarget::Sm90)
    }

    /// rvLLM's current PTX-launched FA2 backend is the GB10 / SM121 path.
    #[inline]
    #[must_use]
    pub const fn supports_fa2_ptx(self) -> bool {
        matches!(self, CompileTarget::Sm121)
    }

    /// RotorQuant KV is config-only in this branch; no arch has kernels yet.
    #[inline]
    #[must_use]
    pub const fn supports_rotorquant_kv(self) -> bool {
        false
    }

    /// Generic predicate for feature-gated runtime/controllers.
    #[inline]
    #[must_use]
    pub const fn supports_feature(self, feature: ArchFeature) -> bool {
        match feature {
            ArchFeature::Fp8TensorCores => self.supports_fp8_tensor_cores(),
            ArchFeature::W4a8Cutlass => self.supports_w4a8_cutlass(),
            ArchFeature::Fa3 => self.supports_fa3(),
            ArchFeature::Fa2Ptx => self.supports_fa2_ptx(),
            ArchFeature::RotorQuantKv => self.supports_rotorquant_kv(),
        }
    }

    /// Return an explicit unsupported-feature error without touching runtime dispatch.
    #[inline]
    #[must_use]
    pub const fn require_feature(self, feature: ArchFeature) -> Result<(), UnsupportedArchFeature> {
        if self.supports_feature(feature) {
            Ok(())
        } else {
            Err(UnsupportedArchFeature {
                target: self,
                feature,
                reason: self.unsupported_feature_reason(feature),
            })
        }
    }

    #[inline]
    #[must_use]
    pub const fn unsupported_feature_reason(self, feature: ArchFeature) -> &'static str {
        match feature {
            ArchFeature::Fp8TensorCores => match self {
                CompileTarget::Sm75 => {
                    "SM75 has no native FP8 tensor cores; FP8 tensor-core routes require SM89, SM90, or SM121 in rvLLM"
                }
                CompileTarget::Sm80 => {
                    "SM80 has no native FP8 tensor cores; FP8 tensor-core routes require SM89, SM90, or SM121 in rvLLM"
                }
                _ => "supported",
            },
            ArchFeature::W4a8Cutlass => match self {
                CompileTarget::Sm90 => "supported",
                CompileTarget::Sm75 => {
                    "SM75 cannot run rvLLM's H100 W4A8 CUTLASS path; add a Turing-specific INT4 route before enabling W4A8"
                }
                _ => "rvLLM W4A8 CUTLASS is currently H100/SM90-only",
            },
            ArchFeature::Fa3 => match self {
                CompileTarget::Sm90 => "supported",
                CompileTarget::Sm75 => {
                    "SM75 is not Hopper; FA3 requires SM90 WGMMA/TMA"
                }
                _ => "FA3 requires Hopper SM90 WGMMA/TMA",
            },
            ArchFeature::Fa2Ptx => match self {
                CompileTarget::Sm121 => "supported",
                _ => "rvLLM FA2 PTX backend is currently SM121-only",
            },
            ArchFeature::RotorQuantKv => "RotorQuant KV kernels are not implemented for any target yet",
        }
    }

    /// Short human-readable status for logs, diagnostics, and experiment UIs.
    #[inline]
    #[must_use]
    pub const fn support_note(self) -> &'static str {
        match self {
            CompileTarget::Sm75 => {
                "SM75 compatibility target: no FP8 tensor cores, W4A8, FA3, FA2 PTX, or RotorQuant KV support yet"
            }
            CompileTarget::Sm80 => {
                "SM80 baseline target: no native FP8 tensor cores, FA3, FA2 PTX, W4A8, or RotorQuant KV support"
            }
            CompileTarget::Sm89 => {
                "SM89 Ada target: FP8 tensor cores and custom attention ABI; no FA3, W4A8, FA2 PTX, or RotorQuant KV support"
            }
            CompileTarget::Sm90 => {
                "SM90 Hopper target: FA3, FP8 tensor cores, and H100 W4A8 CUTLASS support; no FA2 PTX or RotorQuant KV support"
            }
            CompileTarget::Sm121 => {
                "SM121 GB10 target: FP8 tensor cores and FA2 PTX support; no FA3, W4A8 CUTLASS, or RotorQuant KV support"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_str_matches_nvcc_flag() {
        assert_eq!(CompileTarget::Sm75.as_sm_str(), "sm_75");
        assert_eq!(CompileTarget::Sm80.as_sm_str(), "sm_80");
        assert_eq!(CompileTarget::Sm89.as_sm_str(), "sm_89");
        assert_eq!(CompileTarget::Sm90.as_sm_str(), "sm_90");
        assert_eq!(CompileTarget::Sm121.as_sm_str(), "sm_121");
    }

    #[test]
    fn manifest_names_are_pinned() {
        assert_eq!(KERNEL_MANIFEST_FILE, "manifest.json");
        assert_eq!(CompileTarget::Sm75.manifest_arch(), "sm_75");
        assert_eq!(
            CompileTarget::Sm75.manifest_path_components(),
            ("sm_75", "manifest.json")
        );
    }

    #[test]
    fn compute_cap_to_target() {
        assert_eq!(
            CompileTarget::from_compute_capability(7, 5),
            Some(CompileTarget::Sm75),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(8, 0),
            Some(CompileTarget::Sm80),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(8, 9),
            Some(CompileTarget::Sm89),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(9, 0),
            Some(CompileTarget::Sm90),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(12, 1),
            Some(CompileTarget::Sm121),
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

    #[test]
    fn capability_matrix_is_pinned() {
        assert_capabilities(CompileTarget::Sm75, [false, false, false, false, false]);
        assert_capabilities(CompileTarget::Sm80, [false, false, false, false, false]);
        assert_capabilities(CompileTarget::Sm89, [true, false, false, false, false]);
        assert_capabilities(CompileTarget::Sm90, [true, true, true, false, false]);
        assert_capabilities(CompileTarget::Sm121, [true, false, false, true, false]);
    }

    #[test]
    fn support_notes_name_each_target() {
        for (target, name) in [
            (CompileTarget::Sm75, "SM75"),
            (CompileTarget::Sm80, "SM80"),
            (CompileTarget::Sm89, "SM89"),
            (CompileTarget::Sm90, "SM90"),
            (CompileTarget::Sm121, "SM121"),
        ] {
            assert!(target.support_note().contains(name));
        }
    }

    #[test]
    fn sm75_fast_paths_have_explicit_unsupported_errors() {
        assert_unsupported(
            CompileTarget::Sm75,
            ArchFeature::Fp8TensorCores,
            "no native FP8 tensor cores",
        );
        assert_unsupported(
            CompileTarget::Sm75,
            ArchFeature::W4a8Cutlass,
            "Turing-specific INT4",
        );
        assert_unsupported(
            CompileTarget::Sm75,
            ArchFeature::Fa3,
            "requires SM90 WGMMA/TMA",
        );
    }

    #[test]
    fn sm90_hopper_features_pass_requirement_checks() {
        assert!(CompileTarget::Sm90
            .require_feature(ArchFeature::Fp8TensorCores)
            .is_ok());
        assert!(CompileTarget::Sm90
            .require_feature(ArchFeature::W4a8Cutlass)
            .is_ok());
        assert!(CompileTarget::Sm90
            .require_feature(ArchFeature::Fa3)
            .is_ok());
    }

    fn assert_capabilities(target: CompileTarget, expected: [bool; 5]) {
        assert_eq!(target.supports_fp8_tensor_cores(), expected[0]);
        assert_eq!(target.supports_w4a8_cutlass(), expected[1]);
        assert_eq!(target.supports_fa3(), expected[2]);
        assert_eq!(target.supports_fa2_ptx(), expected[3]);
        assert_eq!(target.supports_rotorquant_kv(), expected[4]);
    }

    fn assert_unsupported(target: CompileTarget, feature: ArchFeature, reason: &str) {
        match target.require_feature(feature) {
            Ok(()) => panic!("{target:?} unexpectedly supports {feature:?}"),
            Err(err) => {
                assert_eq!(err.target, target);
                assert_eq!(err.feature, feature);
                assert!(err.reason.contains(reason), "{err}");
                assert!(err.to_string().contains(target.as_sm_str()));
                assert!(err.to_string().contains(feature.as_str()));
            }
        }
    }
}
