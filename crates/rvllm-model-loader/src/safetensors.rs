use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};
use rvllm_core::error::{LLMError, Result};

use crate::dtype::DType;
use crate::weights::{GpuAllocator, ModelWeights, WeightTensor};

/// Loads model weights from safetensors files.
///
/// Uses std::fs::read for safe file access. When rvllm-gpu provides a safe mmap
/// wrapper, this can be swapped to memory-mapped loading for zero-copy.
pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    /// Load all tensors from a single .safetensors file.
    pub fn load(path: &Path, gpu: &dyn GpuAllocator) -> Result<ModelWeights> {
        info!("loading safetensors from {}", path.display());

        let data = std::fs::read(path)?;

        if data.len() < 8 {
            return Err(LLMError::ModelError(
                "safetensors file too small".into(),
            ));
        }

        // First 8 bytes: little-endian u64 header size
        let header_size = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| LLMError::ModelError("invalid header size bytes".into()))?,
        ) as usize;

        if 8 + header_size > data.len() {
            return Err(LLMError::ModelError(
                "header size exceeds file length".into(),
            ));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {}", e)))?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| LLMError::SerializationError(format!("header json: {}", e)))?;

        let data_start = 8 + header_size;
        let mut weights = ModelWeights::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let obj = meta
                .as_object()
                .ok_or_else(|| LLMError::ModelError(format!("tensor {} has non-object meta", name)))?;

            let dtype_str = obj
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing dtype", name)))?;

            let dtype = DType::from_safetensors_str(dtype_str).ok_or_else(|| {
                LLMError::ModelError(format!("unsupported dtype {} for {}", dtype_str, name))
            })?;

            let shape: Vec<usize> = obj
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing shape", name)))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| LLMError::ModelError("invalid shape element".into()))
                })
                .collect::<Result<Vec<_>>>()?;

            let offsets = obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing data_offsets", name)))?;

            if offsets.len() != 2 {
                return Err(LLMError::ModelError(format!(
                    "tensor {} has {} offsets, expected 2",
                    name,
                    offsets.len()
                )));
            }

            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            let abs_start = data_start + start;
            let abs_end = data_start + end;

            if abs_end > data.len() {
                return Err(LLMError::ModelError(format!(
                    "tensor {} data range [{}, {}) exceeds file",
                    name, abs_start, abs_end
                )));
            }

            let tensor_data = &data[abs_start..abs_end];
            let gpu_buf = gpu.upload(tensor_data)?;

            debug!(
                tensor = name.as_str(),
                dtype = %dtype,
                shape = ?shape,
                bytes = tensor_data.len(),
                "loaded tensor"
            );

            weights.insert(WeightTensor::new(name.clone(), shape, dtype, gpu_buf));
        }

        info!(
            "loaded {} tensors from {}",
            weights.num_weights(),
            path.display()
        );
        Ok(weights)
    }

    /// Load from a directory containing sharded safetensors files.
    pub fn load_sharded(dir: &Path, gpu: &dyn GpuAllocator) -> Result<ModelWeights> {
        let mut all_weights = ModelWeights::new();

        let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        shard_files.sort();

        if shard_files.is_empty() {
            return Err(LLMError::ModelError(format!(
                "no .safetensors files found in {}",
                dir.display()
            )));
        }

        info!("loading {} safetensors shards from {}", shard_files.len(), dir.display());

        for shard_path in &shard_files {
            let shard = Self::load(shard_path, gpu)?;
            for name in shard.names().map(String::from).collect::<Vec<_>>() {
                if let Some(tensor) = shard.get(&name) {
                    all_weights.insert(tensor.clone());
                }
            }
        }

        Ok(all_weights)
    }
}

/// Build a minimal safetensors file in memory for testing.
#[cfg(test)]
pub(crate) fn build_test_safetensors(tensors: &[(&str, &[usize], DType, &[u8])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data_blob = Vec::new();

    for (name, shape, dtype, bytes) in tensors {
        let start = data_blob.len();
        data_blob.extend_from_slice(bytes);
        let end = data_blob.len();

        let dtype_str = match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I32 => "I32",
            DType::U8 => "U8",
            _ => "F32",
        };

        let mut meta = serde_json::Map::new();
        meta.insert("dtype".into(), serde_json::Value::String(dtype_str.into()));
        meta.insert(
            "shape".into(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|&s| serde_json::Value::Number(serde_json::Number::from(s)))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(serde_json::Number::from(start)),
                serde_json::Value::Number(serde_json::Number::from(end)),
            ]),
        );
        header.insert(name.to_string(), serde_json::Value::Object(meta));
    }

    let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::new();
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_blob);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::MockGpuAllocator;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn load_single_tensor() {
        let data = vec![0u8; 16];
        let file_bytes = build_test_safetensors(&[("weight", &[2, 2], DType::F32, &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let weights = SafeTensorsLoader::load(tmp.path(), &gpu).unwrap();
        assert_eq!(weights.num_weights(), 1);
        let w = weights.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), DType::F32);
        assert_eq!(w.size_bytes(), 16);
    }

    #[test]
    fn load_multiple_tensors() {
        let d1 = vec![0u8; 8];
        let d2 = vec![1u8; 4];
        let file_bytes = build_test_safetensors(&[
            ("a", &[2], DType::F32, &d1),
            ("b", &[4], DType::U8, &d2),
        ]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let weights = SafeTensorsLoader::load(tmp.path(), &gpu).unwrap();
        assert_eq!(weights.num_weights(), 2);
        assert!(weights.get("a").is_some());
        assert!(weights.get("b").is_some());
    }

    #[test]
    fn load_empty_file_errors() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&[0u8; 4]).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let result = SafeTensorsLoader::load(tmp.path(), &gpu);
        assert!(result.is_err());
    }

    #[test]
    fn load_sharded_directory() {
        let dir = tempfile::tempdir().unwrap();

        let d1 = vec![0u8; 8];
        let file1 = build_test_safetensors(&[("x", &[2], DType::F32, &d1)]);
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), &file1).unwrap();

        let d2 = vec![1u8; 4];
        let file2 = build_test_safetensors(&[("y", &[1], DType::F32, &d2)]);
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), &file2).unwrap();

        let gpu = MockGpuAllocator;
        let weights = SafeTensorsLoader::load_sharded(dir.path(), &gpu).unwrap();
        assert_eq!(weights.num_weights(), 2);
        assert!(weights.get("x").is_some());
        assert!(weights.get("y").is_some());
    }
}
