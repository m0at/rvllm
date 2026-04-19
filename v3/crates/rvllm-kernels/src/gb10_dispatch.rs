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

use core::time::Duration;

/// Which `fp8_gemv_blockwise_wpr_*_kernel` variant to dispatch on GB10.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
    pub const fn entry_point(self) -> &'static str {
        match self {
            Fp8GemvVariant::WprLut => "fp8_gemv_blockwise_wpr_lut_kernel",
            Fp8GemvVariant::WprNative => "fp8_gemv_blockwise_wpr_native_kernel",
        }
    }
}

/// Current clock regime on GB10.
///
/// Reflects the PR #28 model (which didn't reproduce on our hardware —
/// see module docs). Retained so the policy stays robust if a future
/// firmware revives the described plateaus.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
/// Given a regime signal, picks the variant. Keep this function free
/// of I/O so it's trivially unit-testable; the I/O side (probing the
/// clock, reading a wall-clock elapsed) lives in the helpers below.
pub const fn select_variant(regime: ClockRegime) -> Fp8GemvVariant {
    match regime {
        ClockRegime::Sustained => Fp8GemvVariant::WprNative,
        ClockRegime::Throttled => Fp8GemvVariant::WprLut,
        ClockRegime::Unknown => Fp8GemvVariant::WprLut,
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
pub fn regime_from_elapsed(elapsed_since_warmup: Duration) -> ClockRegime {
    if elapsed_since_warmup < THROTTLE_ONSET {
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
pub fn regime_from_clock_mhz(sm_clock_mhz: u32) -> ClockRegime {
    if sm_clock_mhz == 0 {
        ClockRegime::Unknown
    } else if sm_clock_mhz >= SUSTAINED_CLOCK_THRESHOLD_MHZ {
        ClockRegime::Sustained
    } else {
        ClockRegime::Throttled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_picks_native() {
        assert_eq!(
            select_variant(ClockRegime::Sustained),
            Fp8GemvVariant::WprNative,
        );
    }

    #[test]
    fn throttled_picks_lut() {
        assert_eq!(
            select_variant(ClockRegime::Throttled),
            Fp8GemvVariant::WprLut,
        );
    }

    #[test]
    fn unknown_falls_back_to_lut() {
        // Safe default: LUT runs on every arch (no sm_100+ gate) and
        // is robust under throttle. Never regresses accuracy.
        assert_eq!(
            select_variant(ClockRegime::Unknown),
            Fp8GemvVariant::WprLut,
        );
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
