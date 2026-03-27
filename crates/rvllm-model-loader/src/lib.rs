#![forbid(unsafe_code)]
//! Model weight loading for vllm-rs.
//!
//! Supports safetensors and GGUF formats with memory-mapped I/O,
//! HuggingFace name remapping, and tensor-parallel sharding.

pub mod dtype;
pub mod gguf;
pub mod gpu_loader;
pub mod mapper;
pub mod safetensors;
pub mod shard;
pub mod weights;

use std::path::Path;

use tracing::info;
use rvllm_core::config::{ModelConfig, ParallelConfig};
use rvllm_core::error::{LLMError, Result};

use crate::gguf::GGUFLoader;
use crate::mapper::WeightMapper;
use crate::safetensors::SafeTensorsLoader;
use crate::shard::ShardedLoader;
use crate::weights::{GpuAllocator, ModelWeights, WeightTensor};

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    SafeTensors,
    GGUF,
}

/// Detect format from a path. Checks extension of files in a directory,
/// or the extension of a single file.
pub fn detect_format(path: &Path) -> Result<ModelFormat> {
    if path.is_file() {
        return match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => Ok(ModelFormat::SafeTensors),
            Some("gguf") => Ok(ModelFormat::GGUF),
            Some(ext) => Err(LLMError::ModelError(format!(
                "unknown model file extension: {}",
                ext
            ))),
            None => Err(LLMError::ModelError(
                "model file has no extension".into(),
            )),
        };
    }

    if path.is_dir() {
        // Check for safetensors files first, then gguf
        let has_safetensors = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            });
        if has_safetensors {
            return Ok(ModelFormat::SafeTensors);
        }

        let has_gguf = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            });
        if has_gguf {
            return Ok(ModelFormat::GGUF);
        }

        return Err(LLMError::ModelError(format!(
            "no safetensors or gguf files found in {}",
            path.display()
        )));
    }

    Err(LLMError::ModelError(format!(
        "model path does not exist: {}",
        path.display()
    )))
}

/// Load model weights from disk, applying name remapping and optional sharding.
///
/// - Detects format from file extension
/// - Applies HuggingFace -> internal name mapping
/// - Shards weights when tensor_parallel_size > 1
pub fn load_model_weights(
    model_path: &Path,
    config: &dyn ModelConfig,
    parallel: &dyn ParallelConfig,
    rank: usize,
    gpu: &dyn GpuAllocator,
) -> Result<ModelWeights> {
    let format = detect_format(model_path)?;
    info!(
        "loading model {} format={:?} tp={} rank={}",
        config.model_name(),
        format,
        parallel.tensor_parallel_size(),
        rank
    );

    // Load raw weights
    let raw_weights = match format {
        ModelFormat::SafeTensors => {
            if model_path.is_dir() {
                SafeTensorsLoader::load_sharded(model_path, gpu)?
            } else {
                SafeTensorsLoader::load(model_path, gpu)?
            }
        }
        ModelFormat::GGUF => {
            if model_path.is_dir() {
                // Find the first .gguf file in the directory
                let gguf_file = std::fs::read_dir(model_path)?
                    .filter_map(|e| e.ok())
                    .find(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "gguf")
                            .unwrap_or(false)
                    })
                    .map(|e| e.path())
                    .ok_or_else(|| {
                        LLMError::ModelError("no .gguf file found in directory".into())
                    })?;
                GGUFLoader::load(&gguf_file, gpu)?
            } else {
                GGUFLoader::load(model_path, gpu)?
            }
        }
    };

    // Remap weight names
    let mapper = WeightMapper::new(config.model_name());
    let mut mapped = ModelWeights::new();
    let names: Vec<String> = raw_weights.names().map(String::from).collect();
    for name in &names {
        if let Some(tensor) = raw_weights.get(name) {
            let new_name = mapper.map_name(name);
            mapped.insert(WeightTensor::new(
                new_name,
                tensor.shape().to_vec(),
                tensor.dtype(),
                tensor.data().clone(),
            ));
        }
    }

    // Shard if tensor parallel
    let tp_size = parallel.tensor_parallel_size();
    if tp_size > 1 {
        ShardedLoader::shard(mapped, tp_size, rank, gpu)
    } else {
        Ok(mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::MockGpuAllocator;
    use tempfile::NamedTempFile;
    use std::io::Write;

    struct TestModelConfig;
    impl ModelConfig for TestModelConfig {
        fn model_name(&self) -> &str { "llama" }
        fn hidden_size(&self) -> usize { 256 }
        fn num_layers(&self) -> usize { 2 }
        fn num_attention_heads(&self) -> usize { 4 }
        fn num_kv_heads(&self) -> usize { 4 }
        fn vocab_size(&self) -> usize { 1000 }
        fn max_model_len(&self) -> usize { 512 }
    }

    struct TestParallelConfig {
        tp: usize,
    }
    impl ParallelConfig for TestParallelConfig {
        fn tensor_parallel_size(&self) -> usize { self.tp }
        fn pipeline_parallel_size(&self) -> usize { 1 }
    }

    #[test]
    fn detect_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, &[0u8; 16]).unwrap();
        assert_eq!(detect_format(&path).unwrap(), ModelFormat::SafeTensors);
    }

    #[test]
    fn detect_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, &[0u8; 16]).unwrap();
        assert_eq!(detect_format(&path).unwrap(), ModelFormat::GGUF);
    }

    #[test]
    fn detect_safetensors_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &[0u8; 16]).unwrap();
        assert_eq!(detect_format(dir.path()).unwrap(), ModelFormat::SafeTensors);
    }

    #[test]
    fn detect_unknown_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        std::fs::write(&path, &[0u8; 16]).unwrap();
        assert!(detect_format(&path).is_err());
    }

    #[test]
    fn load_model_weights_safetensors() {
        use crate::dtype::DType;

        let data = vec![0u8; 16];
        let file_bytes = crate::safetensors::build_test_safetensors(&[
            ("model.layers.0.self_attn.q_proj.weight", &[2, 2], DType::F32, &data),
        ]);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let config = TestModelConfig;
        let parallel = TestParallelConfig { tp: 1 };

        let weights = load_model_weights(tmp.path(), &config, &parallel, 0, &gpu).unwrap();
        assert_eq!(weights.num_weights(), 1);
        // Name should be remapped from HF convention
        assert!(weights.get("layers.0.attn.q.weight").is_some());
    }

    #[test]
    fn load_model_weights_with_sharding() {
        use crate::dtype::DType;

        // 4x4 U8 tensor = 16 bytes, will be column-sharded (attn.q) into 2x4
        let data: Vec<u8> = (0..16).collect();
        let file_bytes = crate::safetensors::build_test_safetensors(&[
            ("model.layers.0.self_attn.q_proj.weight", &[4, 4], DType::U8, &data),
        ]);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let config = TestModelConfig;
        let parallel = TestParallelConfig { tp: 2 };

        let rank0 = load_model_weights(tmp.path(), &config, &parallel, 0, &gpu).unwrap();
        let w0 = rank0.get("layers.0.attn.q.weight").unwrap();
        assert_eq!(w0.shape(), &[2, 4]);

        let rank1 = load_model_weights(tmp.path(), &config, &parallel, 1, &gpu).unwrap();
        let w1 = rank1.get("layers.0.attn.q.weight").unwrap();
        assert_eq!(w1.shape(), &[2, 4]);
    }
}
