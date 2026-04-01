//! Hugging Face snapshot resolution helpers.
//!
//! Resolves a model repo id to a local cache snapshot directory while handling
//! both single-file and indexed safetensors layouts.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::api::{RepoInfo, Siblings};
use rvllm_core::prelude::{hf_auth_hint, hf_token_from_env, LLMError, Result};
use serde::Deserialize;
use tracing::info;

const SAFETENSORS_INDEX_SUFFIX: &str = ".safetensors.index.json";
const SAFETENSORS_SUFFIX: &str = ".safetensors";

#[derive(Debug, Clone, PartialEq, Eq)]
enum WeightLayout {
    SingleFile { filename: String },
    Indexed { index_filename: String },
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn configured_hf_api() -> Result<Api> {
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token_from_env() {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|e| LLMError::ModelError(format!("failed to init hf-hub: {e}")))
}

fn hf_model_error(model_name: &str, action: &str, error: &str) -> LLMError {
    let mut message = format!("{action} for '{model_name}': {error}");
    if let Some(hint) = hf_auth_hint(error) {
        message.push(' ');
        message.push_str(hint);
    }
    LLMError::ModelError(message)
}

/// Resolve a Hugging Face repo id or local directory to a usable snapshot dir.
pub(crate) fn ensure_snapshot(model_name: &str) -> Result<PathBuf> {
    let path = Path::new(model_name);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    let snapshot_root = repo_snapshot_root(model_name);
    if let Some(snapshot_dir) = find_complete_snapshot(&snapshot_root)? {
        info!(
            path = %snapshot_dir.display(),
            "found complete model snapshot in HF cache"
        );
        return Ok(snapshot_dir);
    }

    info!(
        model = model_name,
        "downloading model snapshot from HuggingFace"
    );
    let api = configured_hf_api()?;
    let repo = api.model(model_name.to_string());

    let config_path = repo.get("config.json").map_err(|e| {
        hf_model_error(model_name, "failed to download config.json", &e.to_string())
    })?;
    let snapshot_dir = config_path.parent().ok_or_else(|| {
        LLMError::ModelError(format!(
            "config.json path '{}' has no parent snapshot dir",
            config_path.display()
        ))
    })?;

    match detect_weight_layout(&repo.info().map_err(|e| {
        hf_model_error(
            model_name,
            "failed to query HuggingFace repo info",
            &e.to_string(),
        )
    })?)? {
        WeightLayout::SingleFile { filename } => {
            repo.get(&filename).map_err(|e| {
                hf_model_error(
                    model_name,
                    &format!("failed to download {filename}"),
                    &e.to_string(),
                )
            })?;
        }
        WeightLayout::Indexed { index_filename } => {
            let index_path = repo.get(&index_filename).map_err(|e| {
                hf_model_error(
                    model_name,
                    &format!("failed to download {index_filename}"),
                    &e.to_string(),
                )
            })?;
            for shard_filename in parse_index_shards(&index_path)? {
                repo.get(&shard_filename).map_err(|e| {
                    hf_model_error(
                        model_name,
                        &format!("failed to download shard {shard_filename}"),
                        &e.to_string(),
                    )
                })?;
            }
        }
    }

    Ok(snapshot_dir.to_path_buf())
}

fn detect_weight_layout(info: &RepoInfo) -> Result<WeightLayout> {
    let index_files = matching_filenames(&info.siblings, |name| {
        name.ends_with(SAFETENSORS_INDEX_SUFFIX)
    });
    match index_files.as_slice() {
        [index_filename] => {
            return Ok(WeightLayout::Indexed {
                index_filename: index_filename.clone(),
            });
        }
        [] => {}
        _ => {
            return Err(LLMError::ModelError(format!(
                "multiple safetensors index files found in repo: {}",
                index_files.join(", ")
            )));
        }
    }

    let weight_files =
        matching_filenames(&info.siblings, |name| name.ends_with(SAFETENSORS_SUFFIX));
    match weight_files.as_slice() {
        [filename] => Ok(WeightLayout::SingleFile {
            filename: filename.clone(),
        }),
        [] => Err(LLMError::ModelError(
            "repo has no supported safetensors weights".into(),
        )),
        _ => Err(LLMError::ModelError(format!(
            "multiple safetensors weight files found without an index: {}",
            weight_files.join(", ")
        ))),
    }
}

fn parse_index_shards(index_path: &Path) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(index_path).map_err(|e| {
        LLMError::ModelError(format!(
            "failed to read safetensors index {}: {}",
            index_path.display(),
            e
        ))
    })?;
    parse_index_shards_str(&contents)
}

fn parse_index_shards_str(contents: &str) -> Result<Vec<String>> {
    let index: SafetensorsIndex = serde_json::from_str(contents)
        .map_err(|e| LLMError::ModelError(format!("invalid safetensors index json: {e}")))?;
    if index.weight_map.is_empty() {
        return Err(LLMError::ModelError(
            "safetensors index has an empty weight_map".into(),
        ));
    }

    let shards: BTreeSet<String> = index.weight_map.into_values().collect();
    Ok(shards.into_iter().collect())
}

fn find_complete_snapshot(snapshot_root: &Path) -> Result<Option<PathBuf>> {
    if !snapshot_root.is_dir() {
        return Ok(None);
    }

    let mut snapshots = std::fs::read_dir(snapshot_root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    snapshots.sort();

    for snapshot in snapshots {
        if snapshot_complete(&snapshot)? {
            return Ok(Some(snapshot));
        }
    }

    Ok(None)
}

fn snapshot_complete(snapshot_dir: &Path) -> Result<bool> {
    if !snapshot_dir.join("config.json").exists() {
        return Ok(false);
    }

    let index_files = matching_snapshot_files(snapshot_dir, |name| {
        name.ends_with(SAFETENSORS_INDEX_SUFFIX)
    })?;
    match index_files.as_slice() {
        [index_filename] => {
            let shards = parse_index_shards(&snapshot_dir.join(index_filename))?;
            return Ok(shards
                .iter()
                .all(|shard_filename| snapshot_dir.join(shard_filename).exists()));
        }
        [] => {}
        _ => {
            return Err(LLMError::ModelError(format!(
                "multiple safetensors index files found in cached snapshot {}",
                snapshot_dir.display()
            )));
        }
    }

    let weight_files =
        matching_snapshot_files(snapshot_dir, |name| name.ends_with(SAFETENSORS_SUFFIX))?;
    match weight_files.as_slice() {
        [_] => Ok(true),
        [] => Ok(false),
        _ => Err(LLMError::ModelError(format!(
            "multiple safetensors weight files found without an index in cached snapshot {}",
            snapshot_dir.display()
        ))),
    }
}

fn matching_filenames(siblings: &[Siblings], predicate: impl Fn(&str) -> bool) -> Vec<String> {
    let mut matches = siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .filter(|name| predicate(name))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    matches.sort();
    matches
}

fn matching_snapshot_files(
    snapshot_dir: &Path,
    predicate: impl Fn(&str) -> bool,
) -> Result<Vec<String>> {
    let mut matches = std::fs::read_dir(snapshot_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|filename| predicate(filename))
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches)
}

fn repo_snapshot_root(model_name: &str) -> PathBuf {
    dirs_hf_cache()
        .join(format!("models--{}", model_name.replace('/', "--")))
        .join("snapshots")
}

fn dirs_hf_cache() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache).join("hub");
    }
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::{RepoInfo, Siblings};
    use tempfile::tempdir;

    fn repo_info(files: &[&str]) -> RepoInfo {
        RepoInfo {
            siblings: files
                .iter()
                .map(|file| Siblings {
                    rfilename: (*file).to_string(),
                })
                .collect(),
            sha: "deadbeef".to_string(),
        }
    }

    #[test]
    fn detect_weight_layout_single_file() {
        let info = repo_info(&["config.json", "tokenizer.json", "model.safetensors"]);
        assert_eq!(
            detect_weight_layout(&info).unwrap(),
            WeightLayout::SingleFile {
                filename: "model.safetensors".into(),
            }
        );
    }

    #[test]
    fn detect_weight_layout_indexed() {
        let info = repo_info(&[
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]);
        assert_eq!(
            detect_weight_layout(&info).unwrap(),
            WeightLayout::Indexed {
                index_filename: "model.safetensors.index.json".into(),
            }
        );
    }

    #[test]
    fn parse_index_shards_dedupes_and_sorts() {
        let contents = r#"{
            "metadata": { "total_size": 123 },
            "weight_map": {
                "model.layers.0.weight": "model-00002-of-00002.safetensors",
                "model.layers.1.weight": "model-00001-of-00002.safetensors",
                "model.layers.2.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        assert_eq!(
            parse_index_shards_str(contents).unwrap(),
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string()
            ]
        );
    }

    #[test]
    fn find_complete_snapshot_accepts_single_file_layout() {
        let temp = tempdir().unwrap();
        let snapshot = temp.path().join("snapshots").join("sha123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), "{}").unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();

        assert_eq!(
            find_complete_snapshot(&temp.path().join("snapshots")).unwrap(),
            Some(snapshot)
        );
    }

    #[test]
    fn find_complete_snapshot_accepts_indexed_layout() {
        let temp = tempdir().unwrap();
        let snapshot = temp.path().join("snapshots").join("sha456");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), "{}").unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{
                "metadata": { "total_size": 1 },
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();
        std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), b"a").unwrap();
        std::fs::write(snapshot.join("model-00002-of-00002.safetensors"), b"b").unwrap();

        assert_eq!(
            find_complete_snapshot(&temp.path().join("snapshots")).unwrap(),
            Some(snapshot)
        );
    }

    #[test]
    fn find_complete_snapshot_rejects_incomplete_snapshot() {
        let temp = tempdir().unwrap();
        let snapshot = temp.path().join("snapshots").join("sha789");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), "{}").unwrap();

        assert_eq!(
            find_complete_snapshot(&temp.path().join("snapshots")).unwrap(),
            None
        );
    }

    #[test]
    fn gated_repo_errors_include_auth_hint() {
        let err = hf_model_error(
            "meta-llama/Llama-3.1-8B",
            "failed to download config.json",
            "HTTP status client error (401 Unauthorized)",
        );
        let err = err.to_string();
        assert!(err.contains("hf auth login"));
        assert!(err.contains("HF_TOKEN"));
    }
}
