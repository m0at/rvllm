use std::path::Path;

use rvllm_core::error::{LLMError, Result};
use tracing::{debug, info};

use crate::dtype::DType;
use crate::weights::{GpuAllocator, ModelWeights, WeightTensor};

/// Magic bytes for GGUF format.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as read from little-endian bytes

/// Loads model weights from GGUF files using memory-mapped I/O.
pub struct GGUFLoader;

impl GGUFLoader {
    /// Load all tensors from a single .gguf file.
    pub fn load(path: &Path, gpu: &dyn GpuAllocator) -> Result<ModelWeights> {
        info!("loading GGUF from {}", path.display());

        let data = std::fs::read(path).map_err(|e| {
            LLMError::ModelError(format!("failed to read {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &data;

        let header = parse_gguf_header(data)?;

        info!(
            "GGUF v{}: {} tensors, {} metadata entries",
            header.version, header.tensor_count, header.metadata_kv_count
        );

        let mut weights = ModelWeights::new();
        let mut offset = header.tensor_info_offset;

        // Parse tensor info entries
        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let (info, new_offset) = parse_tensor_info(data, offset)?;
            tensor_infos.push(info);
            offset = new_offset;
        }

        // Data section starts after header, aligned to the alignment boundary
        // GGUF spec: data starts at the next alignment boundary after all tensor infos
        let alignment = header.alignment.unwrap_or(32) as usize;
        let data_offset = align_up(offset, alignment);

        for info in &tensor_infos {
            let dtype = DType::from_gguf_type(info.dtype_code).ok_or_else(|| {
                LLMError::ModelError(format!(
                    "unsupported GGUF tensor type {} for tensor {}",
                    info.dtype_code, info.name
                ))
            })?;

            let tensor_start = data_offset + info.offset as usize;
            let tensor_end = tensor_start + info.size_bytes;

            if tensor_end > data.len() {
                return Err(LLMError::ModelError(format!(
                    "tensor {} data [{}, {}) exceeds file size {}",
                    info.name,
                    tensor_start,
                    tensor_end,
                    data.len()
                )));
            }

            let tensor_data = &data[tensor_start..tensor_end];
            let gpu_buf = gpu.upload(tensor_data)?;

            debug!(
                tensor = info.name.as_str(),
                dtype = %dtype,
                shape = ?info.shape,
                bytes = tensor_data.len(),
                "loaded GGUF tensor"
            );

            weights.insert(WeightTensor::new(
                info.name.clone(),
                info.shape.clone(),
                dtype,
                gpu_buf,
            ));
        }

        info!(
            "loaded {} tensors from {}",
            weights.num_weights(),
            path.display()
        );
        Ok(weights)
    }
}

#[derive(Debug)]
struct GGUFHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    tensor_info_offset: usize,
    alignment: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GGUFModelInfo {
    pub architecture: String,
    pub name: Option<String>,
    pub file_type: Option<u32>,
    pub block_count: Option<usize>,
    pub context_length: Option<usize>,
    pub embedding_length: Option<usize>,
    pub feed_forward_length: Option<usize>,
    pub attention_head_count: Option<usize>,
    pub attention_head_count_kv: Option<usize>,
    pub attention_key_length: Option<usize>,
    pub attention_value_length: Option<usize>,
    pub rope_freq_base: Option<f32>,
    pub rms_norm_eps: Option<f32>,
    pub vocab_size: Option<usize>,
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype_code: u32,
    offset: u64,
    size_bytes: usize,
}

fn align_up(offset: usize, alignment: usize) -> usize {
    offset.div_ceil(alignment) * alignment
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(LLMError::ModelError("unexpected EOF reading u32".into()));
    }
    Ok(u32::from_le_bytes(
        data[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
    if offset + 8 > data.len() {
        return Err(LLMError::ModelError("unexpected EOF reading u64".into()));
    }
    Ok(u64::from_le_bytes(
        data[offset..offset + 8].try_into().unwrap(),
    ))
}

fn read_string(data: &[u8], offset: usize) -> Result<(String, usize)> {
    let len = read_u64(data, offset)? as usize;
    let str_start = offset + 8;
    let str_end = str_start + len;
    if str_end > data.len() {
        return Err(LLMError::ModelError("unexpected EOF reading string".into()));
    }
    let s = std::str::from_utf8(&data[str_start..str_end])
        .map_err(|e| LLMError::ModelError(format!("invalid utf8 in GGUF string: {}", e)))?
        .to_string();
    Ok((s, str_end))
}

/// GGUF metadata value types.
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// Skip over a metadata value, returning the new offset.
fn skip_metadata_value(data: &[u8], offset: usize, vtype: u32) -> Result<usize> {
    match vtype {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => Ok(offset + 1),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => Ok(offset + 2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => Ok(offset + 4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => Ok(offset + 8),
        GGUF_TYPE_STRING => {
            let (_, new_offset) = read_string(data, offset)?;
            Ok(new_offset)
        }
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset + 4)? as usize;
            let mut pos = offset + 12;
            for _ in 0..count {
                pos = skip_metadata_value(data, pos, elem_type)?;
            }
            Ok(pos)
        }
        _ => Err(LLMError::ModelError(format!(
            "unknown GGUF metadata type {}",
            vtype
        ))),
    }
}

/// Read a u64 metadata value (for alignment).
fn read_metadata_u64_value(data: &[u8], offset: usize, vtype: u32) -> Option<u64> {
    match vtype {
        GGUF_TYPE_UINT32 => read_u32(data, offset).ok().map(|v| v as u64),
        GGUF_TYPE_UINT64 => read_u64(data, offset).ok(),
        _ => None,
    }
}

fn read_metadata_f32_value(data: &[u8], offset: usize, vtype: u32) -> Option<f32> {
    match vtype {
        GGUF_TYPE_FLOAT32 => {
            if offset + 4 <= data.len() {
                Some(f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?))
            } else {
                None
            }
        }
        GGUF_TYPE_FLOAT64 => {
            if offset + 8 <= data.len() {
                Some(f64::from_le_bytes(data[offset..offset + 8].try_into().ok()?) as f32)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn inspect_gguf_model_info(path: &Path) -> Result<GGUFModelInfo> {
    let data = std::fs::read(path).map_err(|e| {
        LLMError::ModelError(format!("failed to read {}: {}", path.display(), e))
    })?;
    let data: &[u8] = &data;
    let header = parse_gguf_header(data)?;
    let mut offset = 24usize;
    let mut architecture = None;
    let mut name = None;
    let mut file_type = None;
    let mut numeric: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut floats: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut arrays: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for _ in 0..header.metadata_kv_count {
        let (key, new_offset) = read_string(data, offset)?;
        let vtype = read_u32(data, new_offset)?;
        let value_offset = new_offset + 4;

        if key == "general.architecture" && vtype == GGUF_TYPE_STRING {
            architecture = Some(read_string(data, value_offset)?.0);
        } else if key == "general.name" && vtype == GGUF_TYPE_STRING {
            name = Some(read_string(data, value_offset)?.0);
        } else if key == "general.file_type" {
            file_type = read_metadata_u64_value(data, value_offset, vtype).map(|v| v as u32);
        } else if vtype == GGUF_TYPE_ARRAY {
            let count = read_u64(data, value_offset + 4)? as usize;
            arrays.insert(key.clone(), count);
        } else if let Some(v) = read_metadata_u64_value(data, value_offset, vtype) {
            numeric.insert(key.clone(), v);
        } else if let Some(v) = read_metadata_f32_value(data, value_offset, vtype) {
            floats.insert(key.clone(), v);
        }

        offset = skip_metadata_value(data, value_offset, vtype)?;
    }

    let arch = architecture.ok_or_else(|| LLMError::ModelError("GGUF missing general.architecture".into()))?;
    let p = |suffix: &str| format!("{arch}.{suffix}");
    Ok(GGUFModelInfo {
        architecture: arch.clone(),
        name,
        file_type,
        block_count: numeric.get(&p("block_count")).copied().map(|v| v as usize),
        context_length: numeric.get(&p("context_length")).copied().map(|v| v as usize),
        embedding_length: numeric.get(&p("embedding_length")).copied().map(|v| v as usize),
        feed_forward_length: numeric.get(&p("feed_forward_length")).copied().map(|v| v as usize),
        attention_head_count: numeric.get(&p("attention.head_count")).copied().map(|v| v as usize),
        attention_head_count_kv: numeric.get(&p("attention.head_count_kv")).copied().map(|v| v as usize),
        attention_key_length: numeric.get(&p("attention.key_length")).copied().map(|v| v as usize),
        attention_value_length: numeric.get(&p("attention.value_length")).copied().map(|v| v as usize),
        rope_freq_base: floats.get(&p("rope.freq_base")).copied(),
        rms_norm_eps: floats.get(&p("attention.layer_norm_rms_epsilon")).copied().or_else(|| floats.get(&p("attention.layer_norm_epsilon")).copied()),
        vocab_size: numeric.get(&p("vocab_size")).copied().map(|v| v as usize).or_else(|| arrays.get("tokenizer.ggml.tokens").copied()),
        expert_count: numeric.get(&p("expert_count")).copied().map(|v| v as usize),
        expert_used_count: numeric.get(&p("expert_used_count")).copied().map(|v| v as usize),
    })
}

fn parse_gguf_header(data: &[u8]) -> Result<GGUFHeader> {
    if data.len() < 24 {
        return Err(LLMError::ModelError("GGUF file too small".into()));
    }

    let magic = read_u32(data, 0)?;
    if magic != GGUF_MAGIC {
        return Err(LLMError::ModelError(format!(
            "invalid GGUF magic: 0x{:08X}, expected 0x{:08X}",
            magic, GGUF_MAGIC
        )));
    }

    let version = read_u32(data, 4)?;
    if !(2..=3).contains(&version) {
        return Err(LLMError::ModelError(format!(
            "unsupported GGUF version {}",
            version
        )));
    }

    let tensor_count = read_u64(data, 8)?;
    let metadata_kv_count = read_u64(data, 16)?;

    // Skip metadata KV pairs to find where tensor info starts
    let mut offset = 24usize;
    let mut alignment = None;

    for _ in 0..metadata_kv_count {
        let (key, new_offset) = read_string(data, offset)?;
        let vtype = read_u32(data, new_offset)?;
        let value_offset = new_offset + 4;

        // Look for general.alignment
        if key == "general.alignment" {
            alignment = read_metadata_u64_value(data, value_offset, vtype);
        }

        offset = skip_metadata_value(data, value_offset, vtype)?;
    }

    Ok(GGUFHeader {
        version,
        tensor_count,
        metadata_kv_count,
        tensor_info_offset: offset,
        alignment,
    })
}

fn parse_tensor_info(data: &[u8], offset: usize) -> Result<(TensorInfo, usize)> {
    let (name, mut pos) = read_string(data, offset)?;

    let n_dims = read_u32(data, pos)? as usize;
    pos += 4;

    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        let dim = read_u64(data, pos)? as usize;
        shape.push(dim);
        pos += 8;
    }

    let dtype_code = read_u32(data, pos)?;
    pos += 4;

    let tensor_offset = read_u64(data, pos)?;
    pos += 8;

    // Compute size in bytes from shape and dtype
    let numel: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let dtype = DType::from_gguf_type(dtype_code).ok_or_else(|| {
        LLMError::ModelError(format!("unsupported GGUF tensor type {} for tensor {}", dtype_code, name))
    })?;
    let size_bytes = dtype.gguf_tensor_bytes(numel).ok_or_else(|| {
        LLMError::ModelError(format!(
            "tensor {} with dtype {} has incompatible element count {} for GGUF block sizing",
            name, dtype, numel
        ))
    })?;

    Ok((
        TensorInfo {
            name,
            shape,
            dtype_code,
            offset: tensor_offset,
            size_bytes,
        },
        pos,
    ))
}

/// Build a minimal GGUF file in memory for testing.
#[cfg(test)]
pub(crate) fn build_test_gguf(tensors: &[(&str, &[usize], u32, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    // Metadata KV count = 0
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Write tensor infos
    let mut data_offset = 0u64;
    for (name, shape, dtype_code, bytes) in tensors {
        // Name string: u64 len + bytes
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        // n_dims
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        // dims
        for &d in *shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        // dtype
        buf.extend_from_slice(&dtype_code.to_le_bytes());
        // offset into data section
        buf.extend_from_slice(&data_offset.to_le_bytes());

        data_offset += bytes.len() as u64;
    }

    // Align to 32 bytes for data section
    let alignment = 32;
    let current = buf.len();
    let aligned = (current + alignment - 1) / alignment * alignment;
    buf.resize(aligned, 0);

    // Write tensor data
    for (_, _, _, bytes) in tensors {
        buf.extend_from_slice(bytes);
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::MockGpuAllocator;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn load_single_gguf_tensor() {
        let data = vec![0u8; 16]; // 4 f32s
        let file_bytes = build_test_gguf(&[("weight", &[2, 2], 0, &data)]); // type 0 = F32

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let weights = GGUFLoader::load(tmp.path(), &gpu).unwrap();
        assert_eq!(weights.num_weights(), 1);
        let w = weights.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), DType::F32);
    }

    #[test]
    fn load_multiple_gguf_tensors() {
        let d1 = vec![0u8; 8];
        let d2 = vec![1u8; 4];
        let file_bytes = build_test_gguf(&[
            ("a", &[2], 0, &d1), // F32
            ("b", &[4], 8, &d2), // U8 (type 8)
        ]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let weights = GGUFLoader::load(tmp.path(), &gpu).unwrap();
        assert_eq!(weights.num_weights(), 2);
    }

    #[test]
    fn bad_magic_errors() {
        let mut data = vec![0u8; 64];
        // Write wrong magic
        data[..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let result = GGUFLoader::load(tmp.path(), &gpu);
        assert!(result.is_err());
    }

    #[test]
    fn too_small_errors() {
        let data = vec![0u8; 8];
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let gpu = MockGpuAllocator;
        let result = GGUFLoader::load(tmp.path(), &gpu);
        assert!(result.is_err());
    }
}
