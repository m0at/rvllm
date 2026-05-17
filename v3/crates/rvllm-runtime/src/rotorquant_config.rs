use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_loader::RotorQuantMode;

use crate::experiment_controller::{ExperimentPlan, KvPath, ENV_KV_PATH};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RotorQuantConfig {
    pub mode: Option<RotorQuantMode>,
    pub bits: u8,
    pub chunk_dim: u16,
}

impl RotorQuantConfig {
    pub fn disabled() -> Self {
        Self {
            mode: None,
            bits: 4,
            chunk_dim: 128,
        }
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_enabled(false)
    }

    pub fn from_env_for_plan(plan: &ExperimentPlan) -> Result<Self> {
        Self::from_env_enabled(matches!(plan.kv_path, KvPath::RotorQuant))
    }

    pub fn enabled(&self) -> bool {
        self.mode.is_some()
    }

    pub fn mode_label(&self) -> &'static str {
        self.mode.map(RotorQuantMode::as_str).unwrap_or("off")
    }

    fn from_env_enabled(enabled_by_plan: bool) -> Result<Self> {
        let mode = std::env::var("RVLLM_ROTORQUANT").ok();
        let bits = std::env::var("RVLLM_ROTORQUANT_BITS").ok();
        let chunk_dim = std::env::var("RVLLM_ROTORQUANT_CHUNK_DIM").ok();
        Self::from_values(
            enabled_by_plan,
            mode.as_deref(),
            bits.as_deref(),
            chunk_dim.as_deref(),
            std::env::var_os(ENV_KV_PATH).is_some(),
        )
    }

    fn from_values(
        enabled_by_plan: bool,
        raw_mode: Option<&str>,
        raw_bits: Option<&str>,
        raw_chunk_dim: Option<&str>,
        kv_path_present: bool,
    ) -> Result<Self> {
        let raw_mode = if enabled_by_plan {
            raw_mode.unwrap_or("rotor_cl3")
        } else if kv_path_present {
            "off"
        } else {
            raw_mode.unwrap_or("off")
        };
        let mode = match raw_mode
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "0" | "off" | "false" | "no" => None,
            other => Some(other.parse::<RotorQuantMode>().map_err(|_| {
                RvllmError::config(
                    ConfigError::InvalidField {
                        name: "RVLLM_ROTORQUANT",
                        reason: format!("unsupported mode {other:?}"),
                    },
                    "RVLLM_ROTORQUANT",
                )
            })?),
        };
        let bits = raw_bits.and_then(|v| v.parse::<u8>().ok()).unwrap_or(4);
        if !(2..=4).contains(&bits) {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "RVLLM_ROTORQUANT_BITS",
                    reason: "expected 2, 3, or 4".into(),
                },
                "RVLLM_ROTORQUANT_BITS",
            ));
        }
        let chunk_dim = raw_chunk_dim
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(128);
        if chunk_dim != 128 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "RVLLM_ROTORQUANT_CHUNK_DIM",
                    reason: "v1 supports 128 only".into(),
                },
                "RVLLM_ROTORQUANT_CHUNK_DIM",
            ));
        }
        Ok(Self {
            mode,
            bits,
            chunk_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_controller::{
        ArchitecturePolicy, AttentionPath, ExperimentConfig, ExperimentPlan, ValidationMode,
        WeightPath,
    };

    fn rotor_plan() -> ExperimentPlan {
        ExperimentPlan::from_config(ExperimentConfig {
            weight_path: WeightPath::Fp8Default,
            kv_path: KvPath::RotorQuant,
            attention_path: AttentionPath::Auto,
            architecture_policy: ArchitecturePolicy::Auto,
            validation_mode: ValidationMode::Smoke,
        })
        .unwrap()
    }

    #[test]
    fn defaults_to_off_without_plan() {
        let config = RotorQuantConfig::from_values(false, None, None, None, false).unwrap();

        assert!(!config.enabled());
        assert_eq!(config.mode_label(), "off");
        assert_eq!(config.bits, 4);
        assert_eq!(config.chunk_dim, 128);
    }

    #[test]
    fn plan_enables_rotor_cl3_by_default() {
        let plan = rotor_plan();
        let config = RotorQuantConfig::from_values(
            matches!(plan.kv_path, KvPath::RotorQuant),
            None,
            None,
            None,
            false,
        )
        .unwrap();

        assert_eq!(config.mode, Some(RotorQuantMode::RotorCl3));
        assert_eq!(config.mode_label(), "rotor_cl3");
    }

    #[test]
    fn env_overrides_mode_and_metadata() {
        let config =
            RotorQuantConfig::from_values(true, Some("iso4"), Some("3"), None, false).unwrap();

        assert_eq!(config.mode, Some(RotorQuantMode::Iso4));
        assert_eq!(config.bits, 3);
        assert_eq!(config.chunk_dim, 128);
    }

    #[test]
    fn explicit_kv_path_keeps_legacy_env_disabled() {
        let config =
            RotorQuantConfig::from_values(false, Some("rotor_cl3"), Some("4"), Some("128"), true)
                .unwrap();

        assert_eq!(config.mode, None);
        assert_eq!(config.mode_label(), "off");
    }
}
