//! GB10 FP8-GEMV kernel variants in `fp8_gemv.ptx`.
//!
//! Three warp-per-row (WPR) variants ship in the same module — they
//! differ only in the `__global__` entry-point symbol:
//!
//!   * [`Fp8GemvVariant::WprLut`] — branchless shared-memory LUT
//!     decode, ~24 ALU instructions per FP8 byte. Works on every
//!     arch (sm_80 through sm_121).
//!   * [`Fp8GemvVariant::WprNative`] — native
//!     `cvt.rn.f16x2.e4m3x2` PTX decode, ~3 ALU per byte. Gated on
//!     `__CUDA_ARCH__ >= 1000` in `kernels/fp8_gemv.cu`; only present
//!     in PTX built for sm_100 / sm_121 / sm_122.
//!   * [`Fp8GemvVariant::WprNativeF16In`] — `WprNative` with f16
//!     activations + f16 output instead of f32/f32. Used by the
//!     Sm121 Gemma 4 decode fast path to avoid the activation
//!     FP8-quant round-trip. Same arch gate as `WprNative`.
//!
//! Empirically on GB10 (driver 595.58, CUDA 13.2) the SM clock stays
//! at ~2520 MHz under sustained GEMV load, so `WprNative` /
//! `WprNativeF16In` unconditionally beats the LUT path. An earlier
//! PR #28 reported a firmware-enforced 851 MHz → 507 MHz throttle
//! after ~3 s of sustained compute, where the ALU-instruction count
//! would matter; we didn't reproduce that on this DGX Spark. If a
//! future firmware revives the plateau, a clock-regime-aware
//! dispatcher can be reintroduced above this module — the kernel
//! variants themselves stay as the primitive.

use rvllm_core::CompileTarget;

/// FP8-GEMV kernel variant shipped in `fp8_gemv.ptx`.
///
/// `#[non_exhaustive]` so adding a future variant (e.g. an FP8
/// tensor-core MMA kernel) isn't a breaking change for downstream
/// `match` expressions.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[must_use]
pub enum Fp8GemvVariant {
    /// `fp8_gemv_blockwise_wpr_lut_kernel`. Every arch.
    WprLut,
    /// `fp8_gemv_blockwise_wpr_native_kernel`. sm_100+ only.
    WprNative,
    /// `fp8_gemv_blockwise_wpr_native_f16in_kernel`. sm_100+ only.
    /// f16 activation + f16 output variant used by the Sm121
    /// decode fast path.
    WprNativeF16In,
}

/// Logical PTX module name (stem as it appears in `manifest.json`).
/// All variants live in the same module.
pub const FP8_GEMV_PTX_STEM: &str = "fp8_gemv";

impl Fp8GemvVariant {
    /// The `__global__` function symbol inside `fp8_gemv.ptx`.
    /// Paired with [`FP8_GEMV_PTX_STEM`] to resolve a variant
    /// through the kernel loader.
    #[inline]
    #[must_use]
    pub const fn entry_point(self) -> &'static str {
        match self {
            Fp8GemvVariant::WprLut => "fp8_gemv_blockwise_wpr_lut_kernel",
            Fp8GemvVariant::WprNative => "fp8_gemv_blockwise_wpr_native_kernel",
            Fp8GemvVariant::WprNativeF16In => {
                "fp8_gemv_blockwise_wpr_native_f16in_kernel"
            }
        }
    }

    /// Whether this variant's entry point is present in the PTX
    /// built for `target`. Single source of truth for the
    /// `#if __CUDA_ARCH__ >= 1000` gate around the native-CVT
    /// kernels in `kernels/fp8_gemv.cu`.
    ///
    /// **Maintenance:** when new Blackwell variants are added to
    /// [`CompileTarget`] (sm_100, sm_122, …), extend the
    /// native-variant arms to include them — otherwise this returns
    /// `false` for a target whose PTX actually does expose the
    /// native entry symbol. The
    /// [`tests::available_for_tracks_arch_gate`] test catches the
    /// oversight if updated in the same PR.
    #[inline]
    #[must_use]
    pub const fn available_for(self, target: CompileTarget) -> bool {
        match self {
            Fp8GemvVariant::WprLut => true,
            Fp8GemvVariant::WprNative => matches!(target, CompileTarget::Sm121),
            Fp8GemvVariant::WprNativeF16In => matches!(target, CompileTarget::Sm121),
        }
    }
}

// Compile-time regression guard: `entry_point` and `available_for`
// must stay `const fn` so callers can materialise variants at
// compile time. Evaluated on every build.
const _CONST_CALLABLE: () = {
    let _ = Fp8GemvVariant::WprNativeF16In.entry_point();
    let _ = Fp8GemvVariant::WprLut.available_for(CompileTarget::Sm90);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_points_match_kernel_source() {
        // These symbol names must track the __global__ function
        // names in kernels/fp8_gemv.cu. If those names change, this
        // test forces an audit.
        assert_eq!(FP8_GEMV_PTX_STEM, "fp8_gemv");
        assert_eq!(
            Fp8GemvVariant::WprLut.entry_point(),
            "fp8_gemv_blockwise_wpr_lut_kernel",
        );
        assert_eq!(
            Fp8GemvVariant::WprNative.entry_point(),
            "fp8_gemv_blockwise_wpr_native_kernel",
        );
        assert_eq!(
            Fp8GemvVariant::WprNativeF16In.entry_point(),
            "fp8_gemv_blockwise_wpr_native_f16in_kernel",
        );
    }

    #[test]
    fn available_for_tracks_arch_gate() {
        // WprLut is built for every arch.
        for t in [
            CompileTarget::Sm80,
            CompileTarget::Sm89,
            CompileTarget::Sm90,
            CompileTarget::Sm121,
        ] {
            assert!(Fp8GemvVariant::WprLut.available_for(t));
        }
        // Native variants are sm_100+ only — today that means Sm121.
        for v in [Fp8GemvVariant::WprNative, Fp8GemvVariant::WprNativeF16In] {
            assert!(!v.available_for(CompileTarget::Sm80));
            assert!(!v.available_for(CompileTarget::Sm89));
            assert!(!v.available_for(CompileTarget::Sm90));
            assert!(v.available_for(CompileTarget::Sm121));
        }
    }
}
