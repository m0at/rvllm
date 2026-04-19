//! GB10 clock-regime dispatch for `fp8_gemv` variants.
//!
//! GB10 / DGX Spark exhibits a power-governed clock regime shift:
//! ~851 MHz sustained for the first few seconds of a workload, then
//! firmware drops to ~507 MHz once the power budget is exhausted. The
//! throttle is not defeatable via `nvidia-smi -lgc` — it's enforced in
//! firmware.
//!
//! This asymmetry inverts the usual "fewer instructions = faster"
//! heuristic, because at 507 MHz the instruction issue rate caps
//! memory-bandwidth utilisation: *more* ALU per byte keeps more loads
//! in flight. Consequently:
//!
//!   * Sustained regime (851 MHz): `wpr_native` wins — native
//!     `cvt.rn.f16x2.e4m3x2` PTX = 3 instructions / byte, less power
//!     draw, keeps the clock up longer.
//!   * Throttled regime (507 MHz): `wpr_lut` wins — branchless LUT =
//!     24 instructions / byte, but at 507 MHz the extra ALU is free
//!     and the pipeline depth saturates LPDDR5X better.
//!
//! This module exposes the policy as a pure function so it can be unit
//! tested without hardware. The clock-regime signal can come from any
//! of:
//!   * an explicit override (`ClockRegime::Forced`)
//!   * a time-window heuristic (works today, no NVML required)
//!   * a runtime clock probe (requires NVML; future work)

use core::time::Duration;

/// Which `fp8_gemv_blockwise_wpr_*_kernel` variant to dispatch on GB10.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Fp8GemvVariant {
    /// `fp8_gemv_blockwise_wpr_lut_kernel` — branchless LUT dequant,
    /// 24 ALU instructions per FP8 byte. Best under the 507 MHz
    /// throttle regime.
    WprLut,
    /// `fp8_gemv_blockwise_wpr_native_kernel` — native
    /// `cvt.rn.f16x2.e4m3x2` PTX (sm_100+), ~3 instructions per byte.
    /// Best under the 851 MHz sustained regime. Compiled only for
    /// `__CUDA_ARCH__ >= 1000`.
    WprNative,
}

impl Fp8GemvVariant {
    /// The PTX logical name (stem) as it appears in `manifest.json`.
    pub const fn kernel_name(self) -> &'static str {
        match self {
            Fp8GemvVariant::WprLut => "fp8_gemv",      // fp8_gemv_blockwise_wpr_lut_kernel
            Fp8GemvVariant::WprNative => "fp8_gemv",   // same PTX, different entry
        }
    }

    /// The `__global__` function symbol inside the PTX module.
    pub const fn entry_point(self) -> &'static str {
        match self {
            Fp8GemvVariant::WprLut => "fp8_gemv_blockwise_wpr_lut_kernel",
            Fp8GemvVariant::WprNative => "fp8_gemv_blockwise_wpr_native_kernel",
        }
    }
}

/// Current clock regime on GB10.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ClockRegime {
    /// Clock is known to be near the 851 MHz cap.
    Sustained,
    /// Clock is known to be at or below the 507 MHz throttle.
    Throttled,
    /// Regime is unknown; caller has no probe. Policy falls back to a
    /// conservative default (`WprLut`) which is the safe bet because
    /// it never triggers the throttle cascade faster than already.
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

/// Time-window heuristic: before the firmware throttle kicks in
/// (~3 s of sustained compute on GB10), assume `Sustained`; after,
/// assume `Throttled`.
///
/// Caller tracks `elapsed_since_warmup` — the wall-clock time since
/// the engine started executing inference work (NOT since process
/// start). Resetting the timer after any idle window > ~2 s is the
/// caller's responsibility (the clock recovers during idle).
pub fn regime_from_elapsed(elapsed_since_warmup: Duration) -> ClockRegime {
    // Firmware transition measured at ~3 s of sustained load; pad by
    // ~500 ms to land on the stable side either way.
    const THROTTLE_ONSET: Duration = Duration::from_millis(2_500);
    if elapsed_since_warmup < THROTTLE_ONSET {
        ClockRegime::Sustained
    } else {
        ClockRegime::Throttled
    }
}

/// Clock-probe heuristic: classify a measured SM clock in MHz.
///
/// Thresholds are set generously around the two known firmware
/// plateaus (507 / 851) so jitter doesn't flip the regime every
/// iteration.
pub fn regime_from_clock_mhz(sm_clock_mhz: u32) -> ClockRegime {
    if sm_clock_mhz >= 700 {
        ClockRegime::Sustained
    } else if sm_clock_mhz > 0 {
        ClockRegime::Throttled
    } else {
        ClockRegime::Unknown
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
        // 507 MHz throttle regime.
        assert_eq!(regime_from_clock_mhz(507), ClockRegime::Throttled);
        // Just below the sustained threshold.
        assert_eq!(regime_from_clock_mhz(699), ClockRegime::Throttled);
        // Sustained plateau.
        assert_eq!(regime_from_clock_mhz(851), ClockRegime::Sustained);
        // Zero = no probe data available.
        assert_eq!(regime_from_clock_mhz(0), ClockRegime::Unknown);
    }

    #[test]
    fn entry_points_match_kernel_source() {
        // These symbol names must track the __global__ function names
        // in kernels/fp8_gemv.cu. If those names change, this test
        // forces an audit.
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
