//! Safetensors shard parsing.
//!
//! Layout: `[u64 header_bytes][JSON header][tensor bytes...]`. The JSON
//! header maps `name -> { dtype, shape, data_offsets: [start,end] }`
//! where offsets are RELATIVE to the start of the tensor payload region
//! (i.e. after header_bytes + 8).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

/// Tensor entry in a shard.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset inside the shard file (relative to file start, i.e.
    /// already includes the `8 + header_bytes` prefix).
    pub file_offset: u64,
    pub nbytes: u64,
}

/// A parsed shard's header. Backing file is mmap'd by the caller.
#[derive(Clone, Debug)]
pub struct ShardHeader {
    pub path: PathBuf,
    pub total_bytes: u64,
    pub tensors: BTreeMap<String, TensorEntry>,
}

impl ShardHeader {
    pub fn parse(path: &Path, file_bytes: &[u8]) -> Result<Self> {
        let loader_err = |detail: String| -> RvllmError {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ctx: LoaderCtx {
                    path: path.to_path_buf(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        };

        if file_bytes.len() < 8 {
            return Err(loader_err("shorter than 8-byte header prefix".into()));
        }
        let header_bytes = u64::from_le_bytes(file_bytes[..8].try_into().unwrap()) as usize;
        let payload_start = 8u64 + header_bytes as u64;
        if (payload_start as usize) > file_bytes.len() {
            return Err(loader_err(format!(
                "header claims {header_bytes} bytes but file is only {}",
                file_bytes.len()
            )));
        }
        let header_str = std::str::from_utf8(&file_bytes[8..8 + header_bytes])
            .map_err(|_| loader_err("header is not valid utf-8".into()))?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| loader_err(format!("header json: {e}")))?;

        let mut tensors = BTreeMap::new();
        for (name, meta) in header.into_iter() {
            if name == "__metadata__" {
                continue;
            }
            let obj = meta
                .as_object()
                .ok_or_else(|| loader_err(format!("{name}: meta not an object")))?;
            let dtype_str = obj
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| loader_err(format!("{name}: missing dtype")))?;
            let dtype = map_dtype(dtype_str)
                .ok_or_else(|| loader_err(format!("{name}: unsupported dtype {dtype_str}")))?;
            let shape: Vec<usize> = obj
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing shape")))?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| loader_err(format!("{name}: bad shape element")))?;
            let offsets = obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| loader_err(format!("{name}: missing data_offsets")))?;
            if offsets.len() != 2 {
                return Err(loader_err(format!(
                    "{name}: expected 2 offsets got {}",
                    offsets.len()
                )));
            }
            let start = offsets[0].as_u64().unwrap_or(0);
            let end = offsets[1].as_u64().unwrap_or(0);
            let nbytes = end - start;
            // Cycle 39: dtype_storage_bytes handles sub-byte packed
            // dtypes (U4Packed/I4Packed) which dtype_bytes returns 0 for.
            let n_elements = shape.iter().product::<usize>() as u64;
            let expected = dtype_storage_bytes(dtype, n_elements);
            if expected != nbytes {
                return Err(loader_err(format!(
                    "{name}: offset range {nbytes} != dtype*shape {expected}"
                )));
            }
            let file_offset = payload_start + start;
            tensors.insert(
                name.clone(),
                TensorEntry {
                    name,
                    dtype,
                    shape,
                    file_offset,
                    nbytes,
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            total_bytes: file_bytes.len() as u64,
            tensors,
        })
    }
}

fn map_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F8_E4M3" | "F8E4M3" => DType::Fp8E4M3,
        // Cycle 39: AWQ / compressed-tensors INT4 weight formats. HF
        // safetensors stores packed INT4 with these dtype strings or
        // (more commonly) as "U8"/"I32" with a separate compressed-
        // tensors metadata block — those need a different loader path.
        // The packed-as-named entries are kept for future-proofing.
        "U4" => DType::U4Packed,
        "I4" => DType::I4Packed,
        // Cycle 41 (AWQ step 3b): compressed-tensors AWQ Gemma 4 stores
        // weight_packed as I32 (8 INT4 per int32) and weight_shape as
        // I64 (original [N,K] dense shape, host metadata).
        // weight_zero_point also I32 (8 INT4 per int32 along N).
        "I32" => DType::I32,
        "I64" => DType::I64,
        "U32" => DType::U32,
        "U8" => DType::U8,
        _ => return None,
    })
}

// Cycle 56 step 1: removed the never-called `dtype_bytes` helper.
// Sub-byte / standard byte-width logic now lives on `DType::bits()` /
// `DType::bytes()` directly; this private safetensors helper was
// stale infrastructure.

/// Cycle 39: storage bytes for a tensor with `n_elements`. Handles
/// sub-byte packed dtypes (U4/I4) by computing bits-then-rounding-up.
/// Use this anywhere the safetensors header validates that a tensor's
/// `data_offsets` range matches its declared shape × dtype.
pub fn dtype_storage_bytes(d: DType, n_elements: u64) -> u64 {
    if d.is_sub_byte() {
        let bits = n_elements * d.bits() as u64;
        (bits + 7) / 8
    } else {
        n_elements * d.bytes() as u64
    }
}

/// HF often ships sharded models: `model.safetensors.index.json`
/// maps `weight_name -> "model-00001-of-00004.safetensors"`.
#[derive(Clone, Debug)]
pub struct ShardIndex {
    pub shards: Vec<PathBuf>,
    pub weight_to_shard: BTreeMap<String, PathBuf>,
}

impl ShardIndex {
    /// Resolve the shard set under `model_dir`.
    ///
    /// - If `model.safetensors.index.json` exists, parses the map.
    /// - Else falls back to a single `model.safetensors`.
    pub fn resolve(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let err_ctx = |detail: String| -> RvllmError {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ctx: LoaderCtx {
                    path: index_path.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            }
        };
        if index_path.exists() {
            let bytes = std::fs::read(&index_path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: index_path.clone(),
                source,
            })?;
            let obj: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| err_ctx(format!("index json: {e}")))?;
            let wmap = obj
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| err_ctx("index missing 'weight_map'".into()))?;
            let mut weight_to_shard = BTreeMap::new();
            let mut shards_set = std::collections::BTreeSet::new();
            for (k, v) in wmap {
                let shard = v
                    .as_str()
                    .ok_or_else(|| err_ctx(format!("{k}: shard value not string")))?;
                let p = model_dir.join(shard);
                shards_set.insert(p.clone());
                weight_to_shard.insert(k.clone(), p);
            }
            Ok(Self {
                shards: shards_set.into_iter().collect(),
                weight_to_shard,
            })
        } else {
            let single = model_dir.join("model.safetensors");
            if !single.exists() {
                return Err(err_ctx(format!(
                    "no index at {} and no model.safetensors at {}",
                    index_path.display(),
                    single.display()
                )));
            }
            Ok(Self {
                shards: vec![single],
                weight_to_shard: BTreeMap::new(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_shard(dir: &Path, tensors: &[(&str, DType, &[usize], &[u8])]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = payload.len();
            payload.extend_from_slice(data);
            let end = payload.len();
            let mut meta = serde_json::Map::new();
            let dt = match dtype {
                DType::F32 => "F32",
                DType::F16 => "F16",
                DType::Bf16 => "BF16",
                DType::Fp8E4M3 => "F8_E4M3",
                _ => "F32",
            };
            meta.insert("dtype".into(), serde_json::Value::String(dt.into()));
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|n| serde_json::Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number((start as u64).into()),
                    serde_json::Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), serde_json::Value::Object(meta));
        }
        let hjson = serde_json::to_string(&header).unwrap();
        let hb = hjson.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        f.write_all(&payload).unwrap();
        path
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-loader-st-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn parses_minimal_shard() {
        let dir = tempdir();
        let data_f32 = (0u32..4).flat_map(|i| (i as f32).to_le_bytes()).collect::<Vec<_>>();
        let path = write_shard(&dir, &[("w", DType::F32, &[4], &data_f32)]);
        let body = std::fs::read(&path).unwrap();
        let hdr = ShardHeader::parse(&path, &body).unwrap();
        let w = hdr.tensors.get("w").unwrap();
        assert_eq!(w.shape, vec![4]);
        assert!(matches!(w.dtype, DType::F32));
        assert_eq!(w.nbytes, 16);
    }

    #[test]
    fn rejects_wrong_offset_length() {
        let dir = tempdir();
        // Payload is 3 bytes but shape says 4 f32 = 16 bytes.
        let path = write_shard(&dir, &[("w", DType::F32, &[4], &[0u8, 1, 2])]);
        let body = std::fs::read(&path).unwrap();
        let err = ShardHeader::parse(&path, &body).unwrap_err();
        assert!(format!("{err}").contains("offset range"));
    }

    #[test]
    fn fallback_to_single_shard() {
        let dir = tempdir();
        let _ = write_shard(&dir, &[("w", DType::F32, &[1], &[0u8; 4])]);
        let idx = ShardIndex::resolve(&dir).unwrap();
        assert_eq!(idx.shards.len(), 1);
        assert!(idx.weight_to_shard.is_empty());
    }
}
