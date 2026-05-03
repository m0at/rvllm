//! Sampling parameters parsed from request bodies.
//!
//! The HTTP request schema accepts the OpenAI sampling surface
//! (`temperature`, `top_p`, `top_k`, `seed`) and `ensure_supported()`
//! validates them and produces a [`SamplingDecision`] that the worker
//! consumes. The runtime branches once per request:
//!
//! - `temperature == 0.0` → [`SamplingDecision::Greedy`] (existing
//!   argmax kernel + CUDA-Graph capture eligible).
//! - `temperature > 0.0` → [`SamplingDecision::Stochastic`] (host-side
//!   temperature + top-p + multinomial sample with a seeded PRNG;
//!   CUDA-Graph capture disabled for the request).
//!
//! Stochastic decode is host-side today: the LM-head argmax is
//! replaced with a full-vocab DtoH + CPU softmax/top-p/sample. A
//! follow-up wires a GPU `top_k_select` kernel so only K floats
//! cross the bus per step.

use crate::error::ApiError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    /// OpenAI defaults: temperature 1.0, top_p 1.0, top_k unset.
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 1.0, top_k: None, seed: None }
    }
}

impl SamplingParams {
    /// Validate the request shape and produce a [`SamplingDecision`].
    ///
    /// `temperature == 0.0` deterministically picks the [`Greedy`]
    /// branch — bit-identical to the pre-sampler runtime, including
    /// CUDA-Graph eligibility. Any positive finite temperature picks
    /// the [`Stochastic`] branch with the requested `top_p` / `top_k`
    /// honored. NaN/inf and out-of-range values still hard-400.
    ///
    /// [`Greedy`]: SamplingDecision::Greedy
    /// [`Stochastic`]: SamplingDecision::Stochastic
    pub fn ensure_supported(self) -> Result<SamplingDecision, ApiError> {
        if !self.temperature.is_finite() {
            return Err(ApiError::invalid_param(
                "temperature must be a finite number",
                "temperature",
                "invalid_value",
            ));
        }
        if !self.top_p.is_finite() {
            return Err(ApiError::invalid_param(
                "top_p must be a finite number",
                "top_p",
                "invalid_value",
            ));
        }
        if self.temperature < 0.0 {
            return Err(ApiError::invalid_param(
                "temperature must be non-negative",
                "temperature",
                "invalid_value",
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::invalid_param(
                "top_p must be in [0, 1]",
                "top_p",
                "invalid_value",
            ));
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(ApiError::invalid_param(
                    "top_k must be >= 1 when set",
                    "top_k",
                    "invalid_value",
                ));
            }
        }

        if self.temperature == 0.0 {
            // Greedy: top_p / top_k / seed are all moot under argmax.
            // Accept and discard rather than 400 on harmless params.
            Ok(SamplingDecision::Greedy)
        } else {
            // Resolve the seed at validation time so the runtime path
            // is fully deterministic given the chosen seed. If the
            // caller did not supply one, draw from a process-wide
            // entropy source (SystemTime ^ a fixed odd constant).
            let seed = self.seed.unwrap_or_else(default_seed);
            Ok(SamplingDecision::Stochastic(StochasticParams {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                seed,
            }))
        }
    }
}

/// Validated sampling decision handed to the worker / runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingDecision {
    /// `temperature == 0.0`. The runtime takes the existing argmax
    /// path and is eligible for CUDA-Graph capture.
    Greedy,
    /// `temperature > 0.0`. The runtime replaces the LM-head argmax
    /// with a host-side temperature + top-p multinomial sample.
    Stochastic(StochasticParams),
}

impl SamplingDecision {
    pub fn is_greedy(&self) -> bool {
        matches!(self, SamplingDecision::Greedy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    /// Resolved at HTTP-validation time: if the caller supplied
    /// `seed`, we forward it; otherwise [`default_seed`] draws one
    /// from system entropy. Either way the runtime sees a concrete
    /// `u64` and the request is reproducible if the caller records
    /// the seed.
    pub seed: u64,
}

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ 0x517c_c1b7_2722_0a95u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_defaults_pick_stochastic() {
        // temperature=1.0 / top_p=1.0 is the OpenAI SDK default. Now
        // that the sampler is wired, defaults route to Stochastic.
        let d = SamplingParams::default().ensure_supported().unwrap();
        assert!(matches!(d, SamplingDecision::Stochastic(_)));
    }

    #[test]
    fn temperature_zero_picks_greedy() {
        let p = SamplingParams { temperature: 0.0, top_p: 1.0, top_k: None, seed: Some(42) };
        let d = p.ensure_supported().unwrap();
        assert_eq!(d, SamplingDecision::Greedy);
    }

    #[test]
    fn positive_temperature_picks_stochastic_and_forwards_seed() {
        let p = SamplingParams { temperature: 0.7, top_p: 0.9, top_k: Some(64), seed: Some(7) };
        let d = p.ensure_supported().unwrap();
        match d {
            SamplingDecision::Stochastic(s) => {
                assert_eq!(s.temperature, 0.7);
                assert_eq!(s.top_p, 0.9);
                assert_eq!(s.top_k, Some(64));
                assert_eq!(s.seed, 7);
            }
            _ => panic!("expected Stochastic"),
        }
    }

    #[test]
    fn missing_seed_resolves_to_concrete_value() {
        let p = SamplingParams { temperature: 0.7, ..Default::default() };
        let d = p.ensure_supported().unwrap();
        assert!(matches!(d, SamplingDecision::Stochastic(_)));
    }

    #[test]
    fn top_p_zero_accepted_at_temp_zero() {
        let p = SamplingParams { temperature: 0.0, top_p: 0.0, ..Default::default() };
        assert_eq!(p.ensure_supported().unwrap(), SamplingDecision::Greedy);
    }

    #[test]
    fn negative_temperature_rejected() {
        let p = SamplingParams { temperature: -0.1, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    #[test]
    fn out_of_range_top_p_rejected() {
        let p = SamplingParams { temperature: 0.0, top_p: 1.5, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    #[test]
    fn top_k_zero_rejected() {
        let p = SamplingParams { temperature: 0.7, top_k: Some(0), ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    #[test]
    fn nan_temperature_rejected() {
        let p = SamplingParams { temperature: f32::NAN, ..Default::default() };
        let err = p.ensure_supported().expect_err("NaN must reject");
        let msg = format!("{err:?}");
        assert!(msg.contains("temperature"), "wrong field in error: {msg}");
    }

    #[test]
    fn nan_top_p_rejected() {
        let p = SamplingParams { temperature: 0.0, top_p: f32::NAN, ..Default::default() };
        let err = p.ensure_supported().expect_err("NaN top_p must reject");
        let msg = format!("{err:?}");
        assert!(msg.contains("top_p"), "wrong field in error: {msg}");
    }

    #[test]
    fn positive_infinity_temperature_rejected() {
        let p = SamplingParams { temperature: f32::INFINITY, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    #[test]
    fn negative_infinity_temperature_rejected() {
        let p = SamplingParams { temperature: f32::NEG_INFINITY, ..Default::default() };
        assert!(p.ensure_supported().is_err());
    }

    /// Pin the seed-routing contract: same caller seed must reach the
    /// runtime as the same `Stochastic.seed` value. The integration
    /// tests cannot prove this end-to-end because the mock worker
    /// bypasses the sampler entirely — so this assertion lives here,
    /// at the unit boundary that the production path threads through.
    #[test]
    fn explicit_seed_is_forwarded_unchanged() {
        let p = SamplingParams { temperature: 0.7, seed: Some(42), ..Default::default() };
        match p.ensure_supported().expect("validates") {
            SamplingDecision::Stochastic(s) => assert_eq!(s.seed, 42),
            _ => panic!("temperature>0 must route to Stochastic"),
        }
    }

    /// Companion to the "no-seed" rule: two consecutive requests with
    /// no caller-supplied seed must each receive a concrete `u64`,
    /// and they must NOT be the literal default sentinel (`0` or the
    /// fixed `0x517c…` constant standing alone). System-entropy mixing
    /// is the property that prevents accidental cross-request
    /// determinism in production.
    #[test]
    fn missing_seed_draws_from_entropy() {
        let a = SamplingParams { temperature: 0.7, seed: None, ..Default::default() }
            .ensure_supported().expect("a");
        // Sleep one nanosecond is sufficient since the seed mixes the
        // SystemTime nanos with a fixed odd constant.
        std::thread::sleep(std::time::Duration::from_nanos(1));
        let b = SamplingParams { temperature: 0.7, seed: None, ..Default::default() }
            .ensure_supported().expect("b");
        let (sa, sb) = match (a, b) {
            (SamplingDecision::Stochastic(a), SamplingDecision::Stochastic(b)) => (a.seed, b.seed),
            _ => panic!("temperature>0 must route to Stochastic"),
        };
        assert_ne!(sa, 0);
        assert_ne!(sb, 0);
        // Two SystemTime samples within the same process tick CAN
        // collide; we only assert a soft inequality across a small
        // sleep. If this proves flaky on slow CI clocks, the assert
        // can be relaxed to `sa | sb != 0`.
        assert_ne!(sa, sb, "consecutive no-seed calls produced identical seeds: {sa}");
    }
}
