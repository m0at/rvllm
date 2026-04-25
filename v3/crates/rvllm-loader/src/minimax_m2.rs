use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_core::{DType, LoaderCtx, LoaderError};
use serde_json::Value;

use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2CheckpointIndex {
    pub index_path: PathBuf,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2CheckpointSummary {
    pub total_tensors: usize,
    pub shard_count: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub dense_tensors: usize,
    pub nvfp4_groups: usize,
    pub nvfp4_required_tensors: usize,
    pub input_scales_present: usize,
    pub input_scales_missing: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2Nvfp4TensorNames {
    pub base: String,
    pub weight: String,
    pub weight_scale: String,
    pub weight_scale_2: String,
    pub input_scale: Option<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2Projection {
    W1,
    W2,
    W3,
}

struct M2ShardMap {
    mmap: memmap2::Mmap,
    header: ShardHeader,
}

pub struct M2SafetensorsReader {
    model_dir: PathBuf,
    shards: Vec<M2ShardMap>,
    tensor_to_shard: BTreeMap<String, usize>,
    index: M2CheckpointIndex,
}

#[derive(Clone, Debug)]
pub struct M2TensorView<'a> {
    pub entry: &'a TensorEntry,
    pub bytes: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct M2Nvfp4TensorView<'a> {
    pub names: M2Nvfp4TensorNames,
    pub weight: M2TensorView<'a>,
    pub weight_scale: M2TensorView<'a>,
    pub weight_scale_2: M2TensorView<'a>,
    pub input_scale: Option<M2TensorView<'a>>,
}

impl M2CheckpointIndex {
    pub fn from_index_file(path: impl AsRef<Path>) -> Result<Self> {
        let index_path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&index_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: index_path.clone(),
            source,
        })?;
        let root: Value = serde_json::from_slice(&bytes).map_err(|e| {
            invalid_owned("model.safetensors.index.json", format!("index json: {e}"))
        })?;
        Self::from_index_value(index_path, &root)
    }

    pub fn from_index_value(index_path: PathBuf, root: &Value) -> Result<Self> {
        let map = root
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| invalid("weight_map", "missing or not an object"))?;
        let mut weight_map = BTreeMap::new();
        for (name, shard) in map {
            let shard = shard
                .as_str()
                .ok_or_else(|| invalid("weight_map", "shard value must be a string"))?;
            weight_map.insert(name.clone(), shard.to_string());
        }
        Ok(Self {
            index_path,
            weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn shard_for(&self, name: &str) -> Option<&str> {
        self.weight_map.get(name).map(String::as_str)
    }

    pub fn nvfp4_group(
        &self,
        layer: usize,
        expert: usize,
        projection: M2Projection,
    ) -> Result<M2Nvfp4TensorNames> {
        let base = format!(
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.{}",
            projection.as_str()
        );
        let weight = format!("{base}.weight");
        let weight_scale = format!("{base}.weight_scale");
        let weight_scale_2 = format!("{base}.weight_scale_2");
        self.require(&weight)?;
        self.require(&weight_scale)?;
        self.require(&weight_scale_2)?;
        let input_scale_name = format!("{base}.input_scale");
        let input_scale = self.contains(&input_scale_name).then_some(input_scale_name);
        Ok(M2Nvfp4TensorNames {
            base,
            weight,
            weight_scale,
            weight_scale_2,
            input_scale,
        })
    }

    pub fn validate_m2(
        &self,
        num_layers: usize,
        num_experts: usize,
    ) -> Result<M2CheckpointSummary> {
        if num_layers == 0 {
            return Err(invalid("num_layers", "must be > 0"));
        }
        if num_experts == 0 {
            return Err(invalid("num_experts", "must be > 0"));
        }

        let mut dense_tensors = 0usize;
        for name in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
        ] {
            self.require(name)?;
            dense_tensors += 1;
        }

        for layer in 0..num_layers {
            for name in dense_layer_names(layer) {
                self.require(&name)?;
                dense_tensors += 1;
            }
        }

        let mut nvfp4_groups = 0usize;
        let mut input_scales_present = 0usize;
        for layer in 0..num_layers {
            for expert in 0..num_experts {
                for projection in [M2Projection::W1, M2Projection::W2, M2Projection::W3] {
                    let group = self.nvfp4_group(layer, expert, projection)?;
                    nvfp4_groups += 1;
                    if group.input_scale.is_some() {
                        input_scales_present += 1;
                    }
                }
            }
        }

        let nvfp4_required_tensors = nvfp4_groups * 3;
        let input_scales_missing = nvfp4_groups - input_scales_present;
        Ok(M2CheckpointSummary {
            total_tensors: self.tensor_count(),
            shard_count: self.shard_count(),
            num_layers,
            num_experts,
            dense_tensors,
            nvfp4_groups,
            nvfp4_required_tensors,
            input_scales_present,
            input_scales_missing,
        })
    }

    fn require(&self, name: &str) -> Result<()> {
        if self.contains(name) {
            Ok(())
        } else {
            Err(invalid_owned("tensor", format!("missing tensor: {name}")))
        }
    }
}

impl M2SafetensorsReader {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let index_path = model_dir.join("model.safetensors.index.json");
        let index = M2CheckpointIndex::from_index_file(&index_path)?;
        let shard_index = ShardIndex::resolve(&model_dir)?;
        let mut shards = Vec::with_capacity(shard_index.shards.len());
        for path in &shard_index.shards {
            shards.push(M2ShardMap::open(path)?);
        }

        let mut path_to_idx = BTreeMap::new();
        for (i, shard) in shards.iter().enumerate() {
            path_to_idx.insert(shard.header.path.clone(), i);
        }

        let mut tensor_to_shard = BTreeMap::new();
        for (name, shard_path) in &shard_index.weight_to_shard {
            let idx = path_to_idx.get(shard_path).copied().ok_or_else(|| {
                loader_corrupt(
                    &model_dir,
                    format!(
                        "index maps {name} to unopened shard {}",
                        shard_path.display()
                    ),
                )
            })?;
            tensor_to_shard.insert(name.clone(), idx);
        }
        if tensor_to_shard.is_empty() {
            for (i, shard) in shards.iter().enumerate() {
                for name in shard.header.tensors.keys() {
                    tensor_to_shard.insert(name.clone(), i);
                }
            }
        }

        Ok(Self {
            model_dir,
            shards,
            tensor_to_shard,
            index,
        })
    }

    pub fn index(&self) -> &M2CheckpointIndex {
        &self.index
    }

    pub fn tensor(&self, name: &str) -> Result<M2TensorView<'_>> {
        let shard_idx = self
            .tensor_to_shard
            .get(name)
            .copied()
            .ok_or_else(|| loader_missing(&self.model_dir, name))?;
        let shard = &self.shards[shard_idx];
        let entry = shard
            .header
            .tensors
            .get(name)
            .ok_or_else(|| loader_missing(&shard.header.path, name))?;
        let start = entry.file_offset as usize;
        let end = start + entry.nbytes as usize;
        Ok(M2TensorView {
            entry,
            bytes: &shard.mmap[start..end],
        })
    }

    pub fn nvfp4_tensor(
        &self,
        layer: usize,
        expert: usize,
        projection: M2Projection,
    ) -> Result<M2Nvfp4TensorView<'_>> {
        let names = self.index.nvfp4_group(layer, expert, projection)?;
        let weight = self.tensor(&names.weight)?;
        let weight_scale = self.tensor(&names.weight_scale)?;
        let weight_scale_2 = self.tensor(&names.weight_scale_2)?;
        let input_scale = names
            .input_scale
            .as_deref()
            .map(|name| self.tensor(name))
            .transpose()?;
        validate_nvfp4_views(
            &names.base,
            &weight,
            &weight_scale,
            &weight_scale_2,
            input_scale.as_ref(),
        )?;
        Ok(M2Nvfp4TensorView {
            names,
            weight,
            weight_scale,
            weight_scale_2,
            input_scale,
        })
    }
}

impl M2ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let header = ShardHeader::parse(path, &mmap)?;
        Ok(Self { mmap, header })
    }
}

impl M2TensorView<'_> {
    pub fn f32_scalar(&self) -> Result<f32> {
        if self.entry.dtype != DType::F32 || self.bytes.len() != 4 {
            return Err(invalid_owned(
                "f32_scalar",
                format!(
                    "{}: expected scalar F32, got {:?} shape {:?} bytes={}",
                    self.entry.name,
                    self.entry.dtype,
                    self.entry.shape,
                    self.bytes.len()
                ),
            ));
        }
        Ok(f32::from_le_bytes(self.bytes.try_into().unwrap()))
    }
}

fn validate_nvfp4_views(
    base: &str,
    weight: &M2TensorView<'_>,
    weight_scale: &M2TensorView<'_>,
    weight_scale_2: &M2TensorView<'_>,
    input_scale: Option<&M2TensorView<'_>>,
) -> Result<()> {
    if weight.entry.dtype != DType::U8 || weight.entry.shape.len() != 2 {
        return Err(invalid_owned(
            "weight",
            format!("{base}.weight must be 2D U8 packed NVFP4"),
        ));
    }
    if weight_scale.entry.dtype != DType::U8 || weight_scale.entry.shape.len() != 2 {
        return Err(invalid_owned(
            "weight_scale",
            format!("{base}.weight_scale must be 2D U8 FP8 scales"),
        ));
    }
    let rows = weight.entry.shape[0];
    let cols = weight.entry.shape[1] * 2;
    let expected_scale = [rows, cols / 16];
    if cols % 16 != 0 || weight_scale.entry.shape.as_slice() != expected_scale {
        return Err(invalid_owned(
            "weight_scale",
            format!(
                "{base}.weight_scale shape {:?} does not match packed shape {:?}",
                weight_scale.entry.shape, weight.entry.shape
            ),
        ));
    }
    validate_scalar_f32(base, "weight_scale_2", weight_scale_2)?;
    if let Some(input_scale) = input_scale {
        validate_scalar_f32(base, "input_scale", input_scale)?;
    }
    Ok(())
}

fn validate_scalar_f32(base: &str, suffix: &'static str, tensor: &M2TensorView<'_>) -> Result<()> {
    if tensor.entry.dtype == DType::F32 && tensor.bytes.len() == 4 {
        Ok(())
    } else {
        Err(invalid_owned(
            suffix,
            format!(
                "{base}.{suffix} must be scalar F32, got {:?} shape {:?} bytes={}",
                tensor.entry.dtype,
                tensor.entry.shape,
                tensor.bytes.len()
            ),
        ))
    }
}

impl M2Projection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }
}

fn dense_layer_names(layer: usize) -> [String; 10] {
    let p = format!("model.layers.{layer}");
    [
        format!("{p}.input_layernorm.weight"),
        format!("{p}.post_attention_layernorm.weight"),
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.self_attn.k_proj.weight"),
        format!("{p}.self_attn.v_proj.weight"),
        format!("{p}.self_attn.o_proj.weight"),
        format!("{p}.self_attn.q_norm.weight"),
        format!("{p}.self_attn.k_norm.weight"),
        format!("{p}.block_sparse_moe.gate.weight"),
        format!("{p}.block_sparse_moe.e_score_correction_bias"),
    ]
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
        "minimax_m2_checkpoint",
    )
}

fn loader_missing(path: &Path, name: &str) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::MissingTensor {
            name: name.to_string(),
        },
        ctx: LoaderCtx {
            path: path.to_path_buf(),
            tensor: Some(name.to_string()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn loader_corrupt(path: &Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tiny_index() -> Value {
        let mut weight_map = serde_json::Map::new();
        for name in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
        ] {
            weight_map.insert(name.into(), Value::String("model.safetensors".into()));
        }
        for layer in 0..2 {
            for name in dense_layer_names(layer) {
                weight_map.insert(name, Value::String(format!("layer-{layer}.safetensors")));
            }
            for expert in 0..2 {
                for projection in [M2Projection::W1, M2Projection::W2, M2Projection::W3] {
                    let base = format!(
                        "model.layers.{layer}.block_sparse_moe.experts.{expert}.{}",
                        projection.as_str()
                    );
                    for suffix in [".weight", ".weight_scale", ".weight_scale_2"] {
                        weight_map.insert(
                            format!("{base}{suffix}"),
                            Value::String(format!("layer-{layer}.safetensors")),
                        );
                    }
                    if !(layer == 1 && expert == 1 && projection == M2Projection::W3) {
                        weight_map.insert(
                            format!("{base}.input_scale"),
                            Value::String("model-inputscales.safetensors".into()),
                        );
                    }
                }
            }
        }
        serde_json::json!({ "weight_map": weight_map })
    }

    #[test]
    fn validates_tiny_modelopt_index() {
        let index =
            M2CheckpointIndex::from_index_value(PathBuf::from("tiny.json"), &tiny_index()).unwrap();
        let summary = index.validate_m2(2, 2).unwrap();
        assert_eq!(summary.total_tensors, 70);
        assert_eq!(summary.dense_tensors, 23);
        assert_eq!(summary.nvfp4_groups, 12);
        assert_eq!(summary.nvfp4_required_tensors, 36);
        assert_eq!(summary.input_scales_present, 11);
        assert_eq!(summary.input_scales_missing, 1);

        let group = index.nvfp4_group(1, 1, M2Projection::W3).unwrap();
        assert_eq!(group.base, "model.layers.1.block_sparse_moe.experts.1.w3");
        assert!(group.input_scale.is_none());
    }

    #[test]
    fn catches_missing_required_expert_tensor() {
        let mut root = tiny_index();
        root["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove("model.layers.1.block_sparse_moe.experts.1.w3.weight_scale_2");
        let index = M2CheckpointIndex::from_index_value(PathBuf::from("tiny.json"), &root).unwrap();
        assert!(index.validate_m2(2, 2).is_err());
    }

    #[test]
    fn opens_tiny_modelopt_shards_and_reads_nvfp4_group() {
        let dir = tempdir("m2-reader");
        std::fs::create_dir_all(&dir).unwrap();
        let root = tiny_index();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&root).unwrap(),
        )
        .unwrap();

        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, shard) in root["weight_map"].as_object().unwrap() {
            by_shard
                .entry(shard.as_str().unwrap().to_string())
                .or_default()
                .push(name.clone());
        }
        for (shard, names) in by_shard {
            write_shard(&dir.join(shard), &names);
        }

        let reader = M2SafetensorsReader::open(&dir).unwrap();
        let summary = reader.index().validate_m2(2, 2).unwrap();
        assert_eq!(summary.input_scales_missing, 1);

        let w1 = reader.nvfp4_tensor(0, 0, M2Projection::W1).unwrap();
        assert_eq!(w1.weight.entry.dtype, DType::U8);
        assert_eq!(w1.weight.entry.shape, vec![2, 8]);
        assert_eq!(w1.weight.bytes.len(), 16);
        assert_eq!(w1.weight_scale.entry.shape, vec![2, 1]);
        assert_eq!(w1.weight_scale_2.f32_scalar().unwrap(), 1.0);
        assert_eq!(w1.input_scale.unwrap().f32_scalar().unwrap(), 2.0);

        let missing_optional = reader.nvfp4_tensor(1, 1, M2Projection::W3).unwrap();
        assert!(missing_optional.input_scale.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn real_checked_in_m2_schema_matches_modelopt_layout() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tpu/harness/m2_checkpoint_schema/model.safetensors.index.json");
        let index = M2CheckpointIndex::from_index_file(path).unwrap();
        let summary = index.validate_m2(62, 256).unwrap();
        assert_eq!(summary.total_tensors, 191_069);
        assert_eq!(summary.shard_count, 26);
        assert_eq!(summary.dense_tensors, 623);
        assert_eq!(summary.nvfp4_groups, 47_616);
        assert_eq!(summary.nvfp4_required_tensors, 142_848);
        assert_eq!(summary.input_scales_present, 47_598);
        assert_eq!(summary.input_scales_missing, 18);
        assert_eq!(
            index
                .shard_for("model.layers.0.block_sparse_moe.experts.0.w1.input_scale")
                .unwrap(),
            "model-inputscales.safetensors"
        );
    }

    fn tempdir(prefix: &str) -> PathBuf {
        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rvllm-{prefix}-{uniq}"))
    }

    fn write_shard(path: &Path, names: &[String]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for name in names {
            let (dtype, shape, bytes) = tensor_fixture(name);
            let start = payload.len();
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let hjson = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(hjson.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hjson).unwrap();
        f.write_all(&payload).unwrap();
    }

    fn tensor_fixture(name: &str) -> (&'static str, Vec<usize>, Vec<u8>) {
        if name.ends_with(".weight_scale_2") {
            return ("F32", vec![1], 1.0f32.to_le_bytes().to_vec());
        }
        if name.ends_with(".input_scale") {
            return ("F32", vec![1], 2.0f32.to_le_bytes().to_vec());
        }
        if name.contains(".experts.") && name.ends_with(".weight_scale") {
            return ("U8", vec![2, 1], vec![0x38; 2]);
        }
        if name.contains(".experts.") && name.ends_with(".weight") {
            return ("U8", vec![2, 8], vec![0x12; 16]);
        }
        ("BF16", vec![1], vec![0, 0])
    }
}
