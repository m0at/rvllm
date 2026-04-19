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
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ClockRegime {
    /// Clock is known to be near the 851 MHz cap.
    Sustained,
    /// Clock is known to be at or below the 507 MHz throttle.
    Throttled,
    /// Regime is unknown; caller has no probe. Policy falls back to a
    /// conservative default (`WprLut`) — it runs on every arch
    /// (`WprNative` is sm_100+ only) and stays correct under throttle
    /// if one ever materialises on this hardware.
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
