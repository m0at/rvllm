//! Env-driven config. No fallbacks: a missing required var aborts
//! startup with a clear error. The only "optional" vars are the
//! defaultable ones (port, host, max_num_seqs, served model name) and
//! the SM90-only `.so` paths, which can be unset when the live device
//! is not Hopper (CUTLASS/FA3 then dlopen-skip).

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
    pub host: String,
    pub port: u16,
    pub max_num_seqs: usize,
    pub served_model_name: String,
    pub dry_run: bool,
}

fn env_required(k: &str) -> Result<String, String> {
    std::env::var(k).map_err(|_| format!("missing required env var: {k}"))
}

fn env_path_required(k: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(env_required(k)?))
}

/// Optional path. Used for SM90-only artifacts (CUTLASS .so, FA3 .so,
/// policy.json). When unset we use `/dev/null` so dlopen on Hopper
/// fails loudly with a clean message rather than a confusing "missing
/// env var" abort.
fn env_path_optional(k: &str) -> PathBuf {
    std::env::var(k)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, String> {
        let dry_run = std::env::var("RVLLM_DRY_RUN").is_ok();

        let model_dir = if dry_run {
            std::env::var("RVLLM_MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/dev/null"))
        } else {
            env_path_required("RVLLM_MODEL_DIR")?
        };
        let kernels_dir = if dry_run {
            std::env::var("RVLLM_KERNELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/dev/null"))
        } else {
            env_path_required("RVLLM_KERNELS_DIR")?
        };

        let cutlass_so = env_path_optional("RVLLM_CUTLASS_SO");
        let fa3_so = env_path_optional("RVLLM_FA3_SO");
        let policy_json = env_path_optional("RVLLM_POLICY");

        let host = std::env::var("RVLLM_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port: u16 = std::env::var("RVLLM_PORT")
            .ok()
            .map(|s| {
                s.parse::<u16>()
                    .map_err(|e| format!("RVLLM_PORT parse: {e}"))
            })
            .transpose()?
            .unwrap_or(8080);
        let max_num_seqs: usize = std::env::var("RVLLM_MAX_NUM_SEQS")
            .ok()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| format!("RVLLM_MAX_NUM_SEQS parse: {e}"))
            })
            .transpose()?
            .unwrap_or(4);
        if max_num_seqs == 0 {
            return Err("RVLLM_MAX_NUM_SEQS must be >= 1".into());
        }
        let served_model_name = std::env::var("RVLLM_SERVED_MODEL_NAME")
            .unwrap_or_else(|_| "gemma4-31b-solidsf".to_string());

        Ok(ServerConfig {
            model_dir,
            kernels_dir,
            cutlass_so,
            fa3_so,
            policy_json,
            host,
            port,
            max_num_seqs,
            served_model_name,
            dry_run,
        })
    }
}
