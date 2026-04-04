//! Persistent disk cache for cuBLAS/cublasLt autotuning results.
//!
//! Stores the best algorithm selection per (gpu_name, m, n, k, dtype) so
//! subsequent model loads skip the benchmark phase for already-tuned shapes.
//! Cache format is JSON at `$RVLLM_AUTOTUNE_CACHE` or `~/.cache/rvllm/autotune.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{LLMError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AutotuneCacheKey {
    pub gpu_name: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneCacheEntry {
    pub workspace_size: usize,
    pub time_us: f64,
    pub algo_index: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    key: AutotuneCacheKey,
    value: AutotuneCacheEntry,
}

#[derive(Debug, Clone, Default)]
pub struct AutotuneCache {
    pub entries: HashMap<AutotuneCacheKey, AutotuneCacheEntry>,
}

impl Serialize for AutotuneCache {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let vec: Vec<CacheEntry> = self.entries.iter().map(|(k, v)| CacheEntry {
            key: k.clone(),
            value: v.clone(),
        }).collect();
        vec.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AutotuneCache {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let vec: Vec<CacheEntry> = Vec::deserialize(deserializer)?;
        let entries = vec.into_iter().map(|e| (e.key, e.value)).collect();
        Ok(Self { entries })
    }
}

impl AutotuneCache {
    pub fn load(cache_path: &Path) -> Self {
        match std::fs::read_to_string(cache_path) {
            Ok(data) => match serde_json::from_str::<AutotuneCache>(&data) {
                Ok(cache) => {
                    tracing::info!(
                        path = %cache_path.display(),
                        entries = cache.entries.len(),
                        "loaded autotune cache"
                    );
                    cache
                }
                Err(e) => {
                    tracing::warn!(
                        path = %cache_path.display(),
                        %e,
                        "corrupt autotune cache, starting fresh"
                    );
                    Self::default()
                }
            },
            Err(_) => {
                tracing::info!(
                    path = %cache_path.display(),
                    "no autotune cache found, starting fresh"
                );
                Self::default()
            }
        }
    }

    pub fn save(&self, cache_path: &Path) -> Result<()> {
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| LLMError::SerializationError(format!("autotune cache: {e}")))?;
        std::fs::write(cache_path, json)?;
        tracing::info!(
            path = %cache_path.display(),
            entries = self.entries.len(),
            "saved autotune cache"
        );
        Ok(())
    }

    pub fn get(&self, key: &AutotuneCacheKey) -> Option<&AutotuneCacheEntry> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: AutotuneCacheKey, entry: AutotuneCacheEntry) {
        self.entries.insert(key, entry);
    }

    pub fn cache_path() -> PathBuf {
        if let Ok(p) = std::env::var("RVLLM_AUTOTUNE_CACHE") {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home).join(".cache/rvllm/autotune.json")
    }

    pub fn is_disabled() -> bool {
        std::env::var("RVLLM_NO_AUTOTUNE_CACHE").map_or(false, |v| v == "1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn cache_round_trip_serialize() {
        let mut cache = AutotuneCache::default();
        let key = AutotuneCacheKey {
            gpu_name: "NVIDIA H100".to_string(),
            m: 32,
            n: 4096,
            k: 4096,
            dtype: "f16".to_string(),
        };
        let entry = AutotuneCacheEntry {
            workspace_size: 4194304,
            time_us: 12.5,
            algo_index: 3,
        };
        cache.insert(key.clone(), entry);

        let json = serde_json::to_string_pretty(&cache).unwrap();
        let loaded: AutotuneCache = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.entries.len(), 1);
        let got = loaded.get(&key).unwrap();
        assert_eq!(got.workspace_size, 4194304);
        assert!((got.time_us - 12.5).abs() < 1e-9);
        assert_eq!(got.algo_index, 3);
    }

    #[test]
    fn cache_save_load_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("autotune.json");

        let mut cache = AutotuneCache::default();
        for i in 0..5usize {
            cache.insert(
                AutotuneCacheKey {
                    gpu_name: "NVIDIA B200".to_string(),
                    m: 1 << i,
                    n: 4096,
                    k: 4096,
                    dtype: "fp8e4m3".to_string(),
                },
                AutotuneCacheEntry {
                    workspace_size: 1024 * (i + 1),
                    time_us: 10.0 + i as f64,
                    algo_index: i as i32,
                },
            );
        }

        cache.save(&path).unwrap();
        let loaded = AutotuneCache::load(&path);
        assert_eq!(loaded.entries.len(), 5);

        let key = AutotuneCacheKey {
            gpu_name: "NVIDIA B200".to_string(),
            m: 4,
            n: 4096,
            k: 4096,
            dtype: "fp8e4m3".to_string(),
        };
        let got = loaded.get(&key).unwrap();
        assert_eq!(got.workspace_size, 3072);
        assert_eq!(got.algo_index, 2);
    }

    #[test]
    fn cache_load_missing_file() {
        let cache = AutotuneCache::load(Path::new("/nonexistent/path/autotune.json"));
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn cache_load_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("autotune.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"not valid json {{{").unwrap();

        let cache = AutotuneCache::load(&path);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn cache_different_gpus_are_distinct() {
        let mut cache = AutotuneCache::default();
        let key_h100 = AutotuneCacheKey {
            gpu_name: "NVIDIA H100".to_string(),
            m: 16,
            n: 4096,
            k: 4096,
            dtype: "f16".to_string(),
        };
        let key_a100 = AutotuneCacheKey {
            gpu_name: "NVIDIA A100".to_string(),
            m: 16,
            n: 4096,
            k: 4096,
            dtype: "f16".to_string(),
        };
        cache.insert(key_h100.clone(), AutotuneCacheEntry {
            workspace_size: 100,
            time_us: 5.0,
            algo_index: 1,
        });
        cache.insert(key_a100.clone(), AutotuneCacheEntry {
            workspace_size: 200,
            time_us: 8.0,
            algo_index: 2,
        });

        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.get(&key_h100).unwrap().workspace_size, 100);
        assert_eq!(cache.get(&key_a100).unwrap().workspace_size, 200);
    }
}
