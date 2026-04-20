//! GB10 clock-regime dispatch for `fp8_gemv` variants.
//!
//! Picks between two warp-per-row kernels in `fp8_gemv.ptx`:
//!
//!   * `WprNative` — native `cvt.rn.f16x2.e4m3x2` PTX (sm_100+ only),
//!     ~3 ALU instructions per FP8 byte.
//!   * `WprLut` — branchless shared-memory LUT, ~24 ALU per byte.
//!     Works on every arch, including sm_80/sm_89/sm_90.
//!
//! **Empirical result on measured GB10 (driver 595.58.03, CUDA 13.2,
//! see `v3/GB10_SPEC.md`):** the SM clock stays at ~2520 MHz under
//! sustained GEMV load and `WprNative` wins unconditionally — ~2× over
//! `WprLut` in the L2-hot regime and tied at the LPDDR5X limit
//! otherwise.
//!
//! The clock-regime machinery below (`ClockRegime` + the two
//! classifiers + `select_variant`) exists as defence-in-depth. It
//! encodes the behaviour PR #28 originally reported — a
//! firmware-enforced 851 MHz → 507 MHz throttle after ~3 s of
//! sustained compute, where the instruction-rate cap would invert the
//! usual "fewer instructions = faster" heuristic: at 507 MHz `WprLut`
//! would saturate LPDDR5X better, at 851 MHz `WprNative` would hold
//! the clock by drawing less power. We couldn't reproduce that
//! plateau on this DGX Spark, so the policy routes every probed
//! regime other than a definite Sustained-at-high-clock to `WprLut`
//! out of caution.
//!
//! A caller can feed the regime from any of:
//!   * `regime_from_elapsed(Duration)` — time-window heuristic, no
//!     NVML required,
//!   * `regime_from_clock_mhz(u32)` — from an NVML / `nvidia-smi`
//!     reading of `clocks.sm`,
//!   * a `ClockRegime` value constructed directly for test / override
//!     paths.
//!
//! ## Integration
//!
//! This module ships the policy only; no call site exists yet. The
//! future runtime FP8-GEMV dispatcher is the intended consumer:
//!
//! 1. At bring-up, resolve `CompileTarget` from
//!    `CudaContextHandle::compute_capability()` →
//!    `CompileTarget::from_compute_capability()`.
//! 2. Per kernel launch (or per decode step), compute the current
//!    `ClockRegime` from whichever probe is cheapest in that path.
//! 3. Call `select_variant(regime, target)`. The returned variant's
//!    `entry_point()` is guaranteed to resolve through
//!    `KernelLoader::load_function(FP8_GEMV_PTX_STEM, variant.entry_point())`.

use core::time::Duration;

use rvllm_core::CompileTarget;

/// Which `fp8_gemv_blockwise_wpr_*_kernel` variant to dispatch on GB10.
///
/// `#[non_exhaustive]` so adding a future variant (e.g. an FP8
/// tensor-core MMA kernel) isn't a breaking change for external
/// `match` expressions. Internal matches inside this crate stay
/// exhaustive by design.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[must_use]
pub enum Fp8GemvVariant {
    /// `fp8_gemv_blockwise_wpr_lut_kernel` — branchless shared-memory
    /// LUT dequant, ~24 ALU instructions per FP8 byte. Runs on every
    /// target arch (sm_80 through sm_121). Theoretically the right
    /// pick if the device enters the PR #28 throttle plateau where
    /// issue rate caps memory throughput — not observed on this DGX
    /// Spark (see module docs).
    WprLut,
    /// `fp8_gemv_blockwise_wpr_native_kernel` — native
    /// `cvt.rn.f16x2.e4m3x2` PTX, ~3 ALU instructions per byte. Only
    /// present when compiled for `__CUDA_ARCH__ >= 1000` (sm_100,
    /// sm_121, sm_122). Unconditionally wins on observed GB10
    /// hardware (SM clock stays ~2520 MHz under load).
    WprNative,
    /// `fp8_gemv_blockwise_wpr_native_f16in_kernel` — same as
    /// `WprNative` but takes f16 activations and writes f16 output.
    /// Enables routing Gemma 4 decode projections (QKV / O / gate_up /
    /// down) through fp8_gemv without an extra activation-quant pass:
    /// the M=1 decode activation stays in f16, the kernel does the
    /// f16→f32 promotion inline via hardware `cvt.f32.f16`. sm_100+
    /// only, same `__CUDA_ARCH__` gate as `WprNative`.
    WprNativeF16In,
}

/// Logical PTX name (stem as it appears in `manifest.json`). Both
/// variants live in the same `fp8_gemv.ptx` module — they differ only
/// in their `__global__` entry-point symbol. Kept as a plain constant
/// rather than a per-variant method so callers don't accidentally
/// treat the two variants as separate modules.
pub const FP8_GEMV_PTX_STEM: &str = "fp8_gemv";

impl Fp8GemvVariant {
    /// The `__global__` function symbol inside the `fp8_gemv.ptx`
    /// module. Paired with `FP8_GEMV_PTX_STEM` to resolve a variant
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

    /// Whether this variant's entry point is actually present in the
    /// PTX built for `target`. Single source of truth for the
    /// `#if __CUDA_ARCH__ >= 1000` gate around the native-CVT kernel
    /// in `kernels/fp8_gemv.cu`. `select_variant` uses this so the
    /// policy can never hand back a symbol the loader won't resolve.
    ///
    /// **Maintenance:** the native-CVT `__CUDA_ARCH__ >= 1000` gate
    /// covers every Blackwell arch (sm_100, sm_121, sm_122, …), but
    /// `CompileTarget` today only models `Sm121`. When additional
    /// Blackwell variants are added to `CompileTarget`, extend the
    /// `WprNative` arm to include them — otherwise this returns
    /// `false` for a target whose PTX actually does expose the
    /// native entry symbol, and `select_variant` silently
    /// down-picks to `WprLut`. A matching test case in
    /// `available_for_tracks_arch_gate` catches the oversight if
    /// updated in the same PR.
    #[inline]
    #[must_use]
    pub const fn available_for(self, target: CompileTarget) -> bool {
        match self {
            // Branchless LUT decode compiles on every arch we build.
            Fp8GemvVariant::WprLut => true,
            // Native CVT requires Blackwell PTX ISA (compute cap >= 10.0).
            // See the maintenance note above — extend this when Sm100 /
            // Sm120 / Sm122 land in `CompileTarget`.
            Fp8GemvVariant::WprNative => matches!(target, CompileTarget::Sm121),
            // Same __CUDA_ARCH__ >= 1000 gate as WprNative.
            Fp8GemvVariant::WprNativeF16In => matches!(target, CompileTarget::Sm121),
        }
    }
}

/// Default: `WprLut`. Runs on every arch, carries no throttle
/// assumptions — the right choice when no probe has constrained the
/// pick yet.
impl Default for Fp8GemvVariant {
    fn default() -> Self {
        Fp8GemvVariant::WprLut
    }
}

/// Current clock regime on GB10.
///
/// Reflects the PR #28 model (which didn't reproduce on our hardware —
/// see module docs). Retained so the policy stays robust if a future
/// firmware revives the described plateaus.
///
/// `#[non_exhaustive]` leaves room for future regime labels (e.g. a
/// cold-clock warm-up state or a thermal-slowdown state) without a
/// breaking match-arm change downstream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[must_use]
pub enum ClockRegime {
    /// Clock is known to be above `SUSTAINED_CLOCK_THRESHOLD_MHZ`
    /// (observed plateau on this DGX Spark is ~2520 MHz; PR #28
    /// reported 851 MHz).
    Sustained,
    /// Clock is known to be below the sustained threshold. Under the
    /// PR #28 model this was 507 MHz.
    Throttled,
    /// Regime is unknown (no probe, or probe returned an unreadable
    /// value). Policy falls back to `WprLut`, which runs on every
    /// arch — `WprNative` is sm_100+ only, so picking it blindly
    /// here would fail to resolve on sm_80/sm_89/sm_90.
    Unknown,
}

/// Pure dispatch policy.
///
/// Given a clock regime and the target the runtime is compiled for,
/// returns a variant whose entry point is guaranteed to be present in
/// the loaded PTX. `target` is the output of
/// `CompileTarget::from_compute_capability` on the live device; a
/// caller on sm_80/sm_89/sm_90 with a Sustained regime (e.g. a test
/// override) gets `WprLut` rather than a `WprNative` pick that would
/// 404 at `KernelLoader::load_function`.
///
/// Keep this function free of I/O so it stays trivially unit-testable
/// — the I/O side (probing the clock, reading a wall-clock elapsed)
/// lives in the helpers below.
///
/// # Example
///
/// Full integration sketch for a future FP8-GEMV dispatcher:
///
/// ```
/// use rvllm_kernels::{select_variant, ClockRegime, FP8_GEMV_PTX_STEM};
/// use rvllm_core::CompileTarget;
///
/// // 1. Arch resolved once at bring-up.
/// let target = CompileTarget::Sm121;
///
/// // 2. Regime sampled per launch (here: we assume sustained).
/// let regime = ClockRegime::Sustained;
///
/// // 3. Policy picks an always-available variant.
/// let variant = select_variant(regime, target);
/// assert_eq!(variant.entry_point(), "fp8_gemv_blockwise_wpr_native_kernel");
///
/// // 4. PTX stem + entry point feed the kernel loader.
/// assert_eq!(FP8_GEMV_PTX_STEM, "fp8_gemv");
///
/// // Same call on pre-Blackwell falls back to WprLut (WprNative
/// // isn't compiled for sm_80/sm_89/sm_90).
/// assert_eq!(
///     select_variant(ClockRegime::Sustained, CompileTarget::Sm90).entry_point(),
///     "fp8_gemv_blockwise_wpr_lut_kernel",
/// );
/// ```
#[inline]
pub const fn select_variant(regime: ClockRegime, target: CompileTarget) -> Fp8GemvVariant {
    // Only pick `WprNative` when the regime argues for it AND the
    // target actually has it. Everything else falls through to the
    // universally-available `WprLut`.
    match regime {
        ClockRegime::Sustained if Fp8GemvVariant::WprNative.available_for(target) => {
            Fp8GemvVariant::WprNative
        }
        _ => Fp8GemvVariant::WprLut,
    }
}

/// Onset of the PR #28 firmware throttle — ~3 s of sustained compute,
/// padded by ~500 ms so time-window callers land on the stable side
/// either way. Re-exported so a monitoring harness or CI gate can
/// assert the same boundary the runtime uses.
pub const THROTTLE_ONSET: Duration = Duration::from_millis(2_500);

/// Time-window heuristic: before `THROTTLE_ONSET` assume `Sustained`;
/// after, assume `Throttled`.
///
/// Caller tracks `elapsed_since_warmup` — the wall-clock time since
/// the engine started executing inference work (NOT since process
/// start). Resetting the timer after any idle window > ~2 s is the
/// caller's responsibility (the clock recovers during idle).
///
/// Implemented via `Duration::as_millis()` rather than `PartialOrd`
/// so the function stays `const fn` (Duration's comparison traits
/// are not yet const-stable).
#[inline]
pub const fn regime_from_elapsed(elapsed_since_warmup: Duration) -> ClockRegime {
    if elapsed_since_warmup.as_millis() < THROTTLE_ONSET.as_millis() {
        ClockRegime::Sustained
    } else {
        ClockRegime::Throttled
    }
}

/// SM-clock boundary between `Sustained` and `Throttled`. Set at
/// **1500 MHz**: well above anything that would be called a
/// "throttled" regime on any Blackwell consumer part, well below the
/// ~2520 MHz observed on this DGX Spark under sustained GEMV load.
/// The PR #28 historical plateaus (507 MHz throttled / 851 MHz
/// sustained) both classify as `Throttled` under this boundary, so
/// the policy stays correct if a future firmware revives them.
pub const SUSTAINED_CLOCK_THRESHOLD_MHZ: u32 = 1500;

/// Clock-probe heuristic: classify a measured SM clock in MHz using
/// `SUSTAINED_CLOCK_THRESHOLD_MHZ`.
#[inline]
pub const fn regime_from_clock_mhz(sm_clock_mhz: u32) -> ClockRegime {
    if sm_clock_mhz == 0 {
        ClockRegime::Unknown
    } else if sm_clock_mhz >= SUSTAINED_CLOCK_THRESHOLD_MHZ {
        ClockRegime::Sustained
    } else {
        ClockRegime::Throttled
    }
}

// Compile-time regression guard: `select_variant`, the two regime
// classifiers, and the helpers must stay `const fn` so callers can
// materialise a variant at compile time. Evaluated on every build
// (not just under `cargo test`), so dropping a `const` qualifier
// fails compilation instead of only breaking tests.
const _CONST_CALLABLE: () = {
    let _ = select_variant(ClockRegime::Sustained, CompileTarget::Sm121);
    let _ = regime_from_clock_mhz(2520);
    let _ = regime_from_elapsed(Duration::from_millis(0));
    let _ = Fp8GemvVariant::WprNative.entry_point();
    let _ = Fp8GemvVariant::WprLut.available_for(CompileTarget::Sm90);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_on_sm121_picks_native() {
        assert_eq!(
            select_variant(ClockRegime::Sustained, CompileTarget::Sm121),
            Fp8GemvVariant::WprNative,
        );
    }

    #[test]
    fn sustained_on_pre_blackwell_downgrades_to_lut() {
        // WprNative isn't compiled for sm_80/sm_89/sm_90 (gated on
        // __CUDA_ARCH__ >= 1000). Even with a Sustained regime the
        // policy must pick an available variant.
        for t in [CompileTarget::Sm80, CompileTarget::Sm89, CompileTarget::Sm90] {
            assert_eq!(
                select_variant(ClockRegime::Sustained, t),
                Fp8GemvVariant::WprLut,
                "target {t:?} should fall back to WprLut",
            );
        }
    }

    #[test]
    fn throttled_picks_lut_on_every_target() {
        for t in [
            CompileTarget::Sm80,
            CompileTarget::Sm89,
            CompileTarget::Sm90,
            CompileTarget::Sm121,
        ] {
            assert_eq!(
                select_variant(ClockRegime::Throttled, t),
                Fp8GemvVariant::WprLut,
            );
        }
    }

    #[test]
    fn unknown_falls_back_to_lut() {
        for t in [
            CompileTarget::Sm80,
            CompileTarget::Sm89,
            CompileTarget::Sm90,
            CompileTarget::Sm121,
        ] {
            assert_eq!(
                select_variant(ClockRegime::Unknown, t),
                Fp8GemvVariant::WprLut,
            );
        }
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
        // WprNative is sm_100+ only — today that means Sm121.
        assert!(!Fp8GemvVariant::WprNative.available_for(CompileTarget::Sm80));
        assert!(!Fp8GemvVariant::WprNative.available_for(CompileTarget::Sm89));
        assert!(!Fp8GemvVariant::WprNative.available_for(CompileTarget::Sm90));
        assert!(Fp8GemvVariant::WprNative.available_for(CompileTarget::Sm121));
    }

    #[test]
    fn default_variant_is_wpr_lut() {
        assert_eq!(Fp8GemvVariant::default(), Fp8GemvVariant::WprLut);
    }

    #[test]
    fn elapsed_heuristic_crosses_at_2s5() {
        assert_eq!(
            regime_from_elapsed(Duration::from_millis(0)),
            ClockRegime::Sustained,
        );
        assert_eq!(
            regime_from_elapsed(Duration::from_millis(2_499)),
            ClockRegime::Sustained,
        );
        assert_eq!(
            regime_from_elapsed(Duration::from_millis(2_500)),
            ClockRegime::Throttled,
        );
        assert_eq!(
            regime_from_elapsed(Duration::from_secs(10)),
            ClockRegime::Throttled,
        );
    }

    #[test]
    fn clock_heuristic_plateaus() {
        // Zero = no probe data available.
        assert_eq!(regime_from_clock_mhz(0), ClockRegime::Unknown);

        // PR #28 historical plateau: 507 MHz → still `Throttled`.
        assert_eq!(regime_from_clock_mhz(507), ClockRegime::Throttled);
        // PR #28 historical plateau: 851 MHz → still `Throttled`
        // under the 1500 MHz boundary (well below observed sustained).
        assert_eq!(regime_from_clock_mhz(851), ClockRegime::Throttled);

        // Just below the 1500 MHz boundary.
        assert_eq!(regime_from_clock_mhz(1499), ClockRegime::Throttled);
        // At the boundary.
        assert_eq!(regime_from_clock_mhz(1500), ClockRegime::Sustained);

        // Observed DGX Spark GEMV-load clock (~2520 MHz in bench).
        assert_eq!(regime_from_clock_mhz(2520), ClockRegime::Sustained);
    }

    #[test]
    fn entry_points_match_kernel_source() {
        // These symbol names must track the __global__ function names
        // in kernels/fp8_gemv.cu. If those names change, this test
        // forces an audit.
        assert_eq!(FP8_GEMV_PTX_STEM, "fp8_gemv");
        assert_eq!(
            Fp8GemvVariant::WprLut.entry_point(),
            "fp8_gemv_blockwise_wpr_lut_kernel",
        );
        assert_eq!(
            Fp8GemvVariant::WprNative.entry_point(),
            "fp8_gemv_blockwise_wpr_native_kernel",
        );
    }
}
