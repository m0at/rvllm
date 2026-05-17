use rvllm_core::{CompileTarget, ConfigError, Result, RvllmError};

pub const ENV_WEIGHT_PATH: &str = "RVLLM_EXPERIMENT_WEIGHT";
pub const ENV_KV_PATH: &str = "RVLLM_EXPERIMENT_KV";
pub const ENV_ATTENTION_PATH: &str = "RVLLM_EXPERIMENT_ATTENTION";
pub const ENV_ARCHITECTURE_POLICY: &str = "RVLLM_EXPERIMENT_ARCH";
pub const ENV_VALIDATION_MODE: &str = "RVLLM_EXPERIMENT_VALIDATION";

const ENV_EXPERIMENT: &str = "RVLLM_EXPERIMENT";
const ENV_W4A8: &str = "RVLLM_W4A8";
const ENV_AWQ_METADATA_ONLY: &str = "RVLLM_AWQ_METADATA_ONLY";
const ENV_ROTORQUANT: &str = "RVLLM_ROTORQUANT";
const ENV_F16_KV: &str = "RVLLM_F16_KV";
const ENV_F16_ONLY: &str = "RVLLM_F16_ONLY";
const ENV_FA3: &str = "RVLLM_FA3";
const ENV_FA2_FALLBACK: &str = "RVLLM_FA2_FALLBACK";
const ENV_FA_FALLBACK_SO: &str = "RVLLM_FA_FALLBACK_SO";
const ENV_FORCE_SM75_COMPAT: &str = "RVLLM_FORCE_SM75_COMPAT";
const ENV_FORCE_HOPPER: &str = "RVLLM_FORCE_HOPPER";

const AUTO_COMPILE_TARGETS: [CompileTarget; 5] = [
    CompileTarget::Sm75,
    CompileTarget::Sm80,
    CompileTarget::Sm89,
    CompileTarget::Sm90,
    CompileTarget::Sm121,
];
const SM75_COMPILE_TARGETS: [CompileTarget; 1] = [CompileTarget::Sm75];
const HOPPER_COMPILE_TARGETS: [CompileTarget; 1] = [CompileTarget::Sm90];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightPath {
    Fp8Default,
    W4A8Awq,
    AwqMetadataOnly,
}

impl Default for WeightPath {
    fn default() -> Self {
        Self::Fp8Default
    }
}

impl WeightPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8Default => "fp8-default",
            Self::W4A8Awq => "w4a8-awq",
            Self::AwqMetadataOnly => "awq-metadata-only",
        }
    }

    fn parse(field: &'static str, value: &str) -> Result<Self> {
        match normalized(value).as_str() {
            "fp8" | "fp8_default" | "default" => Ok(Self::Fp8Default),
            "w4a8" | "w4a8_awq" | "awq" => Ok(Self::W4A8Awq),
            "awq_metadata" | "awq_metadata_only" | "metadata_only" => {
                Ok(Self::AwqMetadataOnly)
            }
            other => Err(invalid_field(
                field,
                format!(
                    "unsupported weight path {other:?}; expected fp8-default, w4a8-awq, or awq-metadata-only"
                ),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvPath {
    F16,
    Fp8,
    RotorQuant,
}

impl Default for KvPath {
    fn default() -> Self {
        Self::F16
    }
}

impl KvPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Fp8 => "fp8",
            Self::RotorQuant => "rotorquant",
        }
    }

    fn parse(field: &'static str, value: &str) -> Result<Self> {
        match normalized(value).as_str() {
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "fp8" | "e4m3" => Ok(Self::Fp8),
            "rotor" | "rotorquant" | "rotor_quant" => Ok(Self::RotorQuant),
            other => Err(invalid_field(
                field,
                format!("unsupported kv path {other:?}; expected f16, fp8, or rotorquant"),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AttentionPath {
    Auto,
    Fa3,
    Fa2Fallback,
}

impl Default for AttentionPath {
    fn default() -> Self {
        Self::Auto
    }
}

impl AttentionPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fa3 => "fa3",
            Self::Fa2Fallback => "fa2-fallback",
        }
    }

    fn parse(field: &'static str, value: &str) -> Result<Self> {
        match normalized(value).as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "fa3" | "flash3" | "flash_attention_3" => Ok(Self::Fa3),
            "fa2" | "fa2_fallback" | "fallback" | "flash_attention_2" => Ok(Self::Fa2Fallback),
            other => Err(invalid_field(
                field,
                format!(
                    "unsupported attention path {other:?}; expected auto, fa3, or fa2-fallback"
                ),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArchitecturePolicy {
    Auto,
    ForceSm75Compat,
    ForceHopper,
}

impl Default for ArchitecturePolicy {
    fn default() -> Self {
        Self::Auto
    }
}

impl ArchitecturePolicy {
    pub fn compile_target_candidates(self) -> &'static [CompileTarget] {
        match self {
            Self::Auto => &AUTO_COMPILE_TARGETS,
            Self::ForceSm75Compat => &SM75_COMPILE_TARGETS,
            Self::ForceHopper => &HOPPER_COMPILE_TARGETS,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceSm75Compat => "force-sm75-compat",
            Self::ForceHopper => "force-hopper",
        }
    }

    fn parse(field: &'static str, value: &str) -> Result<Self> {
        match normalized(value).as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "sm75" | "force_sm75" | "sm75_compat" | "force_sm75_compat" => {
                Ok(Self::ForceSm75Compat)
            }
            "hopper" | "force_hopper" | "sm90" => Ok(Self::ForceHopper),
            other => Err(invalid_field(
                field,
                format!(
                    "unsupported architecture policy {other:?}; expected auto, force-sm75-compat, or force-hopper"
                ),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ValidationMode {
    Smoke,
    Ppl,
    Throughput,
    Chat,
}

impl Default for ValidationMode {
    fn default() -> Self {
        Self::Smoke
    }
}

impl ValidationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Ppl => "ppl",
            Self::Throughput => "throughput",
            Self::Chat => "chat",
        }
    }

    fn parse(field: &'static str, value: &str) -> Result<Self> {
        match normalized(value).as_str() {
            "smoke" | "load" => Ok(Self::Smoke),
            "ppl" | "perplexity" => Ok(Self::Ppl),
            "throughput" | "bench" | "benchmark" => Ok(Self::Throughput),
            "chat" | "interactive" => Ok(Self::Chat),
            other => Err(invalid_field(
                field,
                format!(
                    "unsupported validation mode {other:?}; expected smoke, ppl, throughput, or chat"
                ),
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FeatureGates {
    pub cuda_feature: bool,
    pub gb10_feature: bool,
    pub fp8_weights: bool,
    pub awq_weights: bool,
    pub awq_metadata: bool,
    pub w4a8_dispatch_candidate: bool,
    pub fp8_kv: bool,
    pub rotorquant_kv: bool,
    pub rotorquant_kernel_support: bool,
    pub fa3_attention: bool,
    pub fa2_fallback_attention: bool,
    pub sm75_compat: bool,
    pub hopper: bool,
}

impl FeatureGates {
    pub fn for_plan(
        weight_path: WeightPath,
        kv_path: KvPath,
        attention_path: AttentionPath,
        architecture_policy: ArchitecturePolicy,
    ) -> Self {
        let w4a8_requested = matches!(weight_path, WeightPath::W4A8Awq);
        let rotorquant_requested = matches!(kv_path, KvPath::RotorQuant);
        Self {
            cuda_feature: cfg!(feature = "cuda"),
            gb10_feature: cfg!(feature = "gb10"),
            fp8_weights: matches!(weight_path, WeightPath::Fp8Default),
            awq_weights: w4a8_requested,
            awq_metadata: matches!(
                weight_path,
                WeightPath::W4A8Awq | WeightPath::AwqMetadataOnly
            ),
            w4a8_dispatch_candidate: w4a8_requested
                && supports_w4a8_dispatch_candidate(architecture_policy),
            fp8_kv: matches!(kv_path, KvPath::Fp8),
            rotorquant_kv: rotorquant_requested,
            rotorquant_kernel_support: rotorquant_requested
                && supports_rotorquant_kernel_candidate(architecture_policy),
            fa3_attention: matches!(attention_path, AttentionPath::Fa3),
            fa2_fallback_attention: matches!(attention_path, AttentionPath::Fa2Fallback),
            sm75_compat: matches!(architecture_policy, ArchitecturePolicy::ForceSm75Compat),
            hopper: matches!(architecture_policy, ArchitecturePolicy::ForceHopper),
        }
    }

    pub fn describe(self) -> String {
        let mut parts = Vec::new();
        if self.fp8_weights {
            parts.push("fp8-weights");
        }
        if self.awq_weights {
            parts.push("awq-weights");
        }
        if self.awq_metadata {
            parts.push("awq-metadata");
        }
        if self.w4a8_dispatch_candidate {
            parts.push("w4a8-dispatch-candidate");
        }
        if self.fp8_kv {
            parts.push("fp8-kv");
        }
        if self.rotorquant_kv {
            parts.push("rotorquant-kv-planned");
        }
        if self.rotorquant_kernel_support {
            parts.push("rotorquant-kv-kernel");
        }
        if self.fa3_attention {
            parts.push("fa3");
        }
        if self.fa2_fallback_attention {
            parts.push("fa2-fallback");
        }
        if self.sm75_compat {
            parts.push("sm75-compat");
        }
        if self.hopper {
            parts.push("hopper");
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(",")
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ExperimentConfig {
    pub weight_path: WeightPath,
    pub kv_path: KvPath,
    pub attention_path: AttentionPath,
    pub architecture_policy: ArchitecturePolicy,
    pub validation_mode: ValidationMode,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            weight_path: WeightPath::default(),
            kv_path: KvPath::default(),
            attention_path: AttentionPath::default(),
            architecture_policy: ArchitecturePolicy::default(),
            validation_mode: ValidationMode::default(),
        }
    }
}

impl ExperimentConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            weight_path: weight_path_from_env()?,
            kv_path: kv_path_from_env()?,
            attention_path: attention_path_from_env()?,
            architecture_policy: architecture_policy_from_env()?,
            validation_mode: validation_mode_from_env()?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentPlan {
    pub weight_path: WeightPath,
    pub kv_path: KvPath,
    pub attention_path: AttentionPath,
    pub architecture_policy: ArchitecturePolicy,
    pub validation_mode: ValidationMode,
    pub feature_gates: FeatureGates,
}

impl Default for ExperimentPlan {
    fn default() -> Self {
        let config = ExperimentConfig::default();
        Self {
            weight_path: config.weight_path,
            kv_path: config.kv_path,
            attention_path: config.attention_path,
            architecture_policy: config.architecture_policy,
            validation_mode: config.validation_mode,
            feature_gates: FeatureGates::for_plan(
                config.weight_path,
                config.kv_path,
                config.attention_path,
                config.architecture_policy,
            ),
        }
    }
}

impl ExperimentPlan {
    pub fn from_config(config: ExperimentConfig) -> Result<Self> {
        validate_config(config)?;
        Ok(Self {
            weight_path: config.weight_path,
            kv_path: config.kv_path,
            attention_path: config.attention_path,
            architecture_policy: config.architecture_policy,
            validation_mode: config.validation_mode,
            feature_gates: FeatureGates::for_plan(
                config.weight_path,
                config.kv_path,
                config.attention_path,
                config.architecture_policy,
            ),
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "weights={} kv={} attention={} arch={} validation={} gates={}",
            self.weight_path.as_str(),
            self.kv_path.as_str(),
            self.attention_path.as_str(),
            self.architecture_policy.as_str(),
            self.validation_mode.as_str(),
            self.feature_gates.describe()
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentController {
    plan: ExperimentPlan,
}

impl ExperimentController {
    pub fn from_env() -> Result<Self> {
        Self::from_config(ExperimentConfig::from_env()?)
    }

    pub fn from_config(config: ExperimentConfig) -> Result<Self> {
        Ok(Self {
            plan: ExperimentPlan::from_config(config)?,
        })
    }

    pub fn plan(&self) -> &ExperimentPlan {
        &self.plan
    }

    pub fn into_plan(self) -> ExperimentPlan {
        self.plan
    }

    pub fn describe(&self) -> String {
        self.plan.describe()
    }
}

fn weight_path_from_env() -> Result<WeightPath> {
    if let Some(value) = env_var(ENV_WEIGHT_PATH)? {
        return WeightPath::parse(ENV_WEIGHT_PATH, &value);
    }
    if env_flag(ENV_AWQ_METADATA_ONLY)?.unwrap_or(false) {
        return Ok(WeightPath::AwqMetadataOnly);
    }
    if env_flag(ENV_W4A8)?.unwrap_or(false) {
        return Ok(WeightPath::W4A8Awq);
    }
    Ok(WeightPath::default())
}

fn kv_path_from_env() -> Result<KvPath> {
    if let Some(value) = env_var(ENV_KV_PATH)? {
        return KvPath::parse(ENV_KV_PATH, &value);
    }
    if rotorquant_enabled()? {
        return Ok(KvPath::RotorQuant);
    }
    if env_flag(ENV_F16_ONLY)?.unwrap_or(false) {
        return Ok(KvPath::F16);
    }
    if env_flag(ENV_F16_KV)?.is_some_and(|enabled| !enabled) {
        return Ok(KvPath::Fp8);
    }
    Ok(KvPath::default())
}

fn attention_path_from_env() -> Result<AttentionPath> {
    if let Some(value) = env_var(ENV_ATTENTION_PATH)? {
        return AttentionPath::parse(ENV_ATTENTION_PATH, &value);
    }
    if env_flag(ENV_FA3)?.unwrap_or(false) {
        return Ok(AttentionPath::Fa3);
    }
    if env_flag(ENV_FA2_FALLBACK)?.unwrap_or(false)
        || std::env::var_os(ENV_FA_FALLBACK_SO).is_some()
    {
        return Ok(AttentionPath::Fa2Fallback);
    }
    Ok(AttentionPath::default())
}

fn architecture_policy_from_env() -> Result<ArchitecturePolicy> {
    if let Some(value) = env_var(ENV_ARCHITECTURE_POLICY)? {
        return ArchitecturePolicy::parse(ENV_ARCHITECTURE_POLICY, &value);
    }
    let sm75 = env_flag(ENV_FORCE_SM75_COMPAT)?.unwrap_or(false);
    let hopper = env_flag(ENV_FORCE_HOPPER)?.unwrap_or(false);
    if sm75 && hopper {
        return Err(inconsistent(vec![
            "RVLLM_FORCE_SM75_COMPAT and RVLLM_FORCE_HOPPER cannot both be enabled".to_string(),
        ]));
    }
    if sm75 {
        return Ok(ArchitecturePolicy::ForceSm75Compat);
    }
    if hopper {
        return Ok(ArchitecturePolicy::ForceHopper);
    }
    Ok(ArchitecturePolicy::default())
}

fn validation_mode_from_env() -> Result<ValidationMode> {
    if let Some(value) = env_var(ENV_VALIDATION_MODE)? {
        return ValidationMode::parse(ENV_VALIDATION_MODE, &value);
    }
    Ok(ValidationMode::default())
}

fn validate_config(config: ExperimentConfig) -> Result<()> {
    if matches!(config.weight_path, WeightPath::W4A8Awq)
        && !supports_w4a8_dispatch_candidate(config.architecture_policy)
    {
        return Err(inconsistent(vec![format!(
            "weights=w4a8-awq has no W4A8 dispatch candidate under arch={} (candidates={})",
            config.architecture_policy.as_str(),
            compile_target_candidate_list(config.architecture_policy)
        )]));
    }
    if matches!(config.attention_path, AttentionPath::Fa3)
        && matches!(
            config.architecture_policy,
            ArchitecturePolicy::ForceSm75Compat
        )
    {
        return Err(inconsistent(vec![
            "attention=fa3 conflicts with arch=force-sm75-compat".to_string(),
        ]));
    }
    Ok(())
}

fn supports_w4a8_dispatch_candidate(architecture_policy: ArchitecturePolicy) -> bool {
    architecture_policy
        .compile_target_candidates()
        .iter()
        .copied()
        .any(CompileTarget::supports_w4a8_cutlass)
}

fn supports_rotorquant_kernel_candidate(architecture_policy: ArchitecturePolicy) -> bool {
    architecture_policy
        .compile_target_candidates()
        .iter()
        .copied()
        .any(CompileTarget::supports_rotorquant_kv)
}

fn compile_target_candidate_list(architecture_policy: ArchitecturePolicy) -> String {
    architecture_policy
        .compile_target_candidates()
        .iter()
        .map(|target| target.as_sm_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn rotorquant_enabled() -> Result<bool> {
    let Some(value) = env_var(ENV_ROTORQUANT)? else {
        return Ok(false);
    };
    match normalized(&value).as_str() {
        "0" | "off" | "false" | "no" => Ok(false),
        "1" | "on" | "true" | "yes" | "rotor" | "rotor_cl3" | "planar2" | "iso4" => Ok(true),
        other => Err(invalid_field(
            ENV_ROTORQUANT,
            format!("unsupported rotorquant mode {other:?}"),
        )),
    }
}

fn env_var(name: &'static str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(invalid_field(name, "expected unicode env value"))
        }
    }
}

fn env_flag(name: &'static str) -> Result<Option<bool>> {
    env_var(name)?
        .map(|value| parse_flag(name, &value))
        .transpose()
}

fn parse_flag(name: &'static str, value: &str) -> Result<bool> {
    match normalized(value).as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(invalid_field(
            name,
            format!("unsupported boolean {other:?}; expected 1/0, true/false, yes/no, or on/off"),
        )),
    }
}

fn normalized(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn invalid_field(name: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name,
            reason: reason.into(),
        },
        name,
    )
}

fn inconsistent(reasons: Vec<String>) -> RvllmError {
    RvllmError::config(ConfigError::Inconsistent { reasons }, ENV_EXPERIMENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const TEST_ENV: &[&str] = &[
        ENV_WEIGHT_PATH,
        ENV_KV_PATH,
        ENV_ATTENTION_PATH,
        ENV_ARCHITECTURE_POLICY,
        ENV_VALIDATION_MODE,
        ENV_W4A8,
        ENV_AWQ_METADATA_ONLY,
        ENV_ROTORQUANT,
        ENV_F16_KV,
        ENV_F16_ONLY,
        ENV_FA3,
        ENV_FA2_FALLBACK,
        ENV_FA_FALLBACK_SO,
        ENV_FORCE_SM75_COMPAT,
        ENV_FORCE_HOPPER,
    ];

    struct EnvGuard {
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn clear() -> Self {
            let saved = TEST_ENV
                .iter()
                .map(|&name| (name, std::env::var_os(name)))
                .collect();
            for &name in TEST_ENV {
                std::env::remove_var(name);
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(*name, value),
                    None => std::env::remove_var(*name),
                }
            }
        }
    }

    #[test]
    fn defaults_plan_fp8_weights_f16_kv_auto_smoke() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();

        let controller = ExperimentController::from_env().unwrap();
        let plan = controller.plan();

        assert_eq!(plan.weight_path, WeightPath::Fp8Default);
        assert_eq!(plan.kv_path, KvPath::F16);
        assert_eq!(plan.attention_path, AttentionPath::Auto);
        assert_eq!(plan.architecture_policy, ArchitecturePolicy::Auto);
        assert_eq!(plan.validation_mode, ValidationMode::Smoke);
        assert!(plan.feature_gates.fp8_weights);
        assert!(!plan.feature_gates.rotorquant_kv);
        assert_eq!(
            controller.describe(),
            "weights=fp8-default kv=f16 attention=auto arch=auto validation=smoke gates=fp8-weights"
        );
    }

    #[test]
    fn explicit_env_builds_awq_rotorquant_hopper_ppl_plan() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();
        std::env::set_var(ENV_WEIGHT_PATH, "w4a8-awq");
        std::env::set_var(ENV_KV_PATH, "rotorquant");
        std::env::set_var(ENV_ATTENTION_PATH, "fa3");
        std::env::set_var(ENV_ARCHITECTURE_POLICY, "force-hopper");
        std::env::set_var(ENV_VALIDATION_MODE, "ppl");

        let plan = ExperimentController::from_env().unwrap().into_plan();

        assert_eq!(plan.weight_path, WeightPath::W4A8Awq);
        assert_eq!(plan.kv_path, KvPath::RotorQuant);
        assert_eq!(plan.attention_path, AttentionPath::Fa3);
        assert_eq!(plan.architecture_policy, ArchitecturePolicy::ForceHopper);
        assert_eq!(plan.validation_mode, ValidationMode::Ppl);
        assert!(plan.feature_gates.awq_weights);
        assert!(plan.feature_gates.awq_metadata);
        assert!(plan.feature_gates.w4a8_dispatch_candidate);
        assert!(plan.feature_gates.rotorquant_kv);
        assert!(!plan.feature_gates.rotorquant_kernel_support);
        assert!(plan.feature_gates.fa3_attention);
        assert!(plan.feature_gates.hopper);
        assert_eq!(
            plan.describe(),
            "weights=w4a8-awq kv=rotorquant attention=fa3 arch=force-hopper validation=ppl gates=awq-weights,awq-metadata,w4a8-dispatch-candidate,rotorquant-kv-planned,fa3,hopper"
        );
    }

    #[test]
    fn env_aliases_cover_metadata_fp8kv_fa2_sm75_chat() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();
        std::env::set_var(ENV_WEIGHT_PATH, "metadata-only");
        std::env::set_var(ENV_KV_PATH, "e4m3");
        std::env::set_var(ENV_ATTENTION_PATH, "fa2");
        std::env::set_var(ENV_ARCHITECTURE_POLICY, "sm75");
        std::env::set_var(ENV_VALIDATION_MODE, "interactive");

        let plan = ExperimentController::from_env().unwrap().into_plan();

        assert_eq!(plan.weight_path, WeightPath::AwqMetadataOnly);
        assert_eq!(plan.kv_path, KvPath::Fp8);
        assert_eq!(plan.attention_path, AttentionPath::Fa2Fallback);
        assert_eq!(
            plan.architecture_policy,
            ArchitecturePolicy::ForceSm75Compat
        );
        assert_eq!(plan.validation_mode, ValidationMode::Chat);
        assert!(plan.feature_gates.awq_metadata);
        assert!(!plan.feature_gates.w4a8_dispatch_candidate);
        assert!(plan.feature_gates.fp8_kv);
        assert!(plan.feature_gates.fa2_fallback_attention);
        assert!(plan.feature_gates.sm75_compat);
    }

    #[test]
    fn legacy_env_flags_map_to_routes() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();
        std::env::set_var(ENV_W4A8, "1");
        std::env::set_var(ENV_ROTORQUANT, "planar2");
        std::env::set_var(ENV_FA_FALLBACK_SO, "/tmp/libflash_attn.so");
        std::env::set_var(ENV_FORCE_HOPPER, "on");

        let plan = ExperimentController::from_env().unwrap().into_plan();

        assert_eq!(plan.weight_path, WeightPath::W4A8Awq);
        assert_eq!(plan.kv_path, KvPath::RotorQuant);
        assert_eq!(plan.attention_path, AttentionPath::Fa2Fallback);
        assert_eq!(plan.architecture_policy, ArchitecturePolicy::ForceHopper);
        assert_eq!(plan.validation_mode, ValidationMode::Smoke);
        assert!(plan.feature_gates.w4a8_dispatch_candidate);
        assert!(plan.feature_gates.rotorquant_kv);
        assert!(!plan.feature_gates.rotorquant_kernel_support);
    }

    #[test]
    fn invalid_env_uses_config_error_style() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();
        std::env::set_var(ENV_KV_PATH, "quantum");

        let err = ExperimentController::from_env().unwrap_err();

        match err {
            RvllmError::Config {
                err:
                    ConfigError::InvalidField {
                        name,
                        reason: _reason,
                    },
                field,
            } => {
                assert_eq!(name, ENV_KV_PATH);
                assert_eq!(field, ENV_KV_PATH);
            }
            other => panic!("expected config invalid field, got {other:?}"),
        }
    }

    #[test]
    fn fa3_sm75_combo_is_rejected_in_config_validation() {
        let err = ExperimentController::from_config(ExperimentConfig {
            weight_path: WeightPath::Fp8Default,
            kv_path: KvPath::F16,
            attention_path: AttentionPath::Fa3,
            architecture_policy: ArchitecturePolicy::ForceSm75Compat,
            validation_mode: ValidationMode::Smoke,
        })
        .unwrap_err();

        match err {
            RvllmError::Config {
                err: ConfigError::Inconsistent { reasons },
                field,
            } => {
                assert_eq!(field, ENV_EXPERIMENT);
                assert_eq!(reasons.len(), 1);
            }
            other => panic!("expected config inconsistency, got {other:?}"),
        }
    }

    #[test]
    fn w4a8_sm75_combo_is_rejected_by_compile_target_capability() {
        let err = ExperimentController::from_config(ExperimentConfig {
            weight_path: WeightPath::W4A8Awq,
            kv_path: KvPath::F16,
            attention_path: AttentionPath::Auto,
            architecture_policy: ArchitecturePolicy::ForceSm75Compat,
            validation_mode: ValidationMode::Smoke,
        })
        .unwrap_err();

        match err {
            RvllmError::Config {
                err: ConfigError::Inconsistent { reasons },
                field,
            } => {
                assert_eq!(field, ENV_EXPERIMENT);
                assert_eq!(reasons.len(), 1);
                assert!(reasons[0].contains("w4a8-awq"));
                assert!(reasons[0].contains("sm_75"));
            }
            other => panic!("expected config inconsistency, got {other:?}"),
        }
    }

    #[test]
    fn legacy_w4a8_sm75_env_combo_is_rejected() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::clear();
        std::env::set_var(ENV_W4A8, "1");
        std::env::set_var(ENV_FORCE_SM75_COMPAT, "1");

        let err = ExperimentController::from_env().unwrap_err();

        match err {
            RvllmError::Config {
                err: ConfigError::Inconsistent { reasons },
                field,
            } => {
                assert_eq!(field, ENV_EXPERIMENT);
                assert_eq!(reasons.len(), 1);
                assert!(reasons[0].contains("w4a8-awq"));
                assert!(reasons[0].contains("force-sm75-compat"));
            }
            other => panic!("expected config inconsistency, got {other:?}"),
        }
    }

    #[test]
    fn rotorquant_is_planned_gate_without_kernel_support_claim() {
        let plan = ExperimentController::from_config(ExperimentConfig {
            weight_path: WeightPath::Fp8Default,
            kv_path: KvPath::RotorQuant,
            attention_path: AttentionPath::Auto,
            architecture_policy: ArchitecturePolicy::Auto,
            validation_mode: ValidationMode::Smoke,
        })
        .unwrap()
        .into_plan();

        assert!(plan.feature_gates.rotorquant_kv);
        assert!(!plan.feature_gates.rotorquant_kernel_support);
        assert!(plan.describe().contains("rotorquant-kv-planned"));
        assert!(!plan.describe().contains("rotorquant-kv-kernel"));
    }
}
