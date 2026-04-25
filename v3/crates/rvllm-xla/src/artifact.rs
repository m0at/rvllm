use std::path::Path;

use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_fused::{M2PrefillKvDType, M2PrefillScanShape};
use serde::{Deserialize, Serialize};

use crate::ffi::PjrtElementType;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct XlaTensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct XlaArtifact {
    pub mlir_file: String,
    pub inputs: Vec<XlaTensorSpec>,
    pub outputs: Vec<XlaTensorSpec>,
    pub donate_indices: Vec<usize>,
    pub num_partitions: usize,
}

pub fn load_artifact(dir: &Path) -> Result<(XlaArtifact, Vec<u8>, Option<Vec<u8>>)> {
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes = read_file(&manifest_path, "manifest.json")?;
    let artifact: XlaArtifact = serde_json::from_slice(&manifest_bytes).map_err(|e| {
        invalid_owned(
            "manifest.json",
            format!("failed to parse {}: {e}", manifest_path.display()),
        )
    })?;
    let mlir_bytes = read_file(&dir.join(&artifact.mlir_file), "mlir_file")?;
    let compile_options_path = dir.join("compile_options.pb");
    let compile_options = if compile_options_path.exists() {
        Some(read_file(&compile_options_path, "compile_options.pb")?)
    } else {
        None
    };
    Ok((artifact, mlir_bytes, compile_options))
}

pub fn m2_prefill_artifact_manifest(
    mlir_file: impl Into<String>,
    shape: M2PrefillScanShape,
    num_partitions: usize,
) -> Result<XlaArtifact> {
    shape.validate()?;
    if num_partitions == 0 {
        return Err(invalid("num_partitions", "must be > 0"));
    }
    let kv_dtype = match shape.kv_dtype {
        M2PrefillKvDType::Bf16 => PjrtElementType::BF16,
        M2PrefillKvDType::Int8 => PjrtElementType::S8,
    };
    Ok(XlaArtifact {
        mlir_file: mlir_file.into(),
        inputs: vec![
            tensor(
                "token_ids",
                &[shape.batch, shape.prompt_len],
                PjrtElementType::S32,
            ),
            tensor("positions", &[shape.total_tokens()], PjrtElementType::S32),
            tensor(
                "slot_mapping",
                &[shape.total_tokens()],
                PjrtElementType::S32,
            ),
            tensor("cu_seqlens_q", &[shape.batch + 1], PjrtElementType::S32),
            tensor("context_lens", &[shape.batch], PjrtElementType::S32),
            tensor("kv_cache", &[shape.kv_cache_bytes()], kv_dtype),
        ],
        outputs: vec![tensor(
            "last_hidden",
            &[shape.batch, shape.hidden],
            PjrtElementType::BF16,
        )],
        donate_indices: vec![5],
        num_partitions,
    })
}

pub fn write_m2_prefill_artifact(
    dir: &Path,
    mlir_file: &str,
    mlir_text: &str,
    shape: M2PrefillScanShape,
    num_partitions: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| {
        invalid_owned(
            "artifact_dir",
            format!("failed to create {}: {e}", dir.display()),
        )
    })?;
    std::fs::write(dir.join(mlir_file), mlir_text).map_err(|e| {
        invalid_owned(
            "mlir_file",
            format!("failed to write {}: {e}", dir.join(mlir_file).display()),
        )
    })?;
    let artifact = m2_prefill_artifact_manifest(mlir_file, shape, num_partitions)?;
    let manifest = serde_json::to_vec_pretty(&artifact)
        .map_err(|e| invalid_owned("manifest.json", format!("failed to serialize: {e}")))?;
    std::fs::write(dir.join("manifest.json"), manifest).map_err(|e| {
        invalid_owned(
            "manifest.json",
            format!(
                "failed to write {}: {e}",
                dir.join("manifest.json").display()
            ),
        )
    })?;
    Ok(())
}

fn tensor(name: &str, shape: &[usize], dtype: PjrtElementType) -> XlaTensorSpec {
    XlaTensorSpec {
        name: name.to_string(),
        shape: shape.iter().map(|&x| x as i64).collect(),
        dtype,
    }
}

fn read_file(path: &Path, field: &'static str) -> Result<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| invalid_owned(field, format!("failed to read {}: {e}", path.display())))
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    invalid_owned(field, reason.to_string())
}

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
        },
        "xla_artifact",
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn m2_shape() -> M2PrefillScanShape {
        M2PrefillScanShape {
            batch: 1,
            prompt_len: 20,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Int8,
        }
    }

    #[test]
    fn m2_prefill_manifest_matches_scan_signature() {
        let artifact =
            m2_prefill_artifact_manifest("rvllm_m2_prefill_scan.mlir", m2_shape(), 8).unwrap();
        assert_eq!(artifact.inputs.len(), 6);
        assert_eq!(artifact.outputs.len(), 1);
        assert_eq!(artifact.donate_indices, vec![5]);
        assert_eq!(artifact.inputs[0].shape, vec![1, 20]);
        assert_eq!(artifact.inputs[5].dtype, PjrtElementType::S8);
        assert_eq!(artifact.inputs[5].shape, vec![260_046_848]);
        assert_eq!(artifact.outputs[0].shape, vec![1, 3072]);
        assert_eq!(artifact.num_partitions, 8);
    }

    #[test]
    fn write_and_load_m2_prefill_artifact_roundtrips() {
        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir: PathBuf = std::env::temp_dir().join(format!("rvllm-xla-prefill-{uniq}"));
        let shape = m2_shape();
        let mlir = shape.mlir("rvllm_m2_prefill_scan").unwrap();
        write_m2_prefill_artifact(&dir, "model.mlir", &mlir, shape, 8).unwrap();
        let (artifact, mlir_bytes, compile_options) = load_artifact(&dir).unwrap();
        assert_eq!(artifact.mlir_file, "model.mlir");
        assert_eq!(artifact.inputs[3].name, "cu_seqlens_q");
        assert!(String::from_utf8_lossy(&mlir_bytes).contains("rvllm.prefill"));
        assert!(compile_options.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
