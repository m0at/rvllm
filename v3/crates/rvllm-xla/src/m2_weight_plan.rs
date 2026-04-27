use std::path::{Path, PathBuf};

use rvllm_core::DType;
use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_fused::{nvfp4_to_int8_matrix, M2Nvfp4MatmulShape};
use rvllm_loader::M2TensorView;
use rvllm_loader::{M2CheckpointIndex, M2Projection, M2SafetensorsReader};

use crate::{M2GraphAbi, M2GraphTensorSpec, M2Nvfp4ProjectionAbi, M2_NUM_EXPERTS, M2_NUM_LAYERS};
#[cfg(feature = "tpu")]
use crate::{PjrtBufferHandle, PjrtClientHandle};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2WeightRole {
    Global,
    LayerDense,
    Nvfp4Packed,
    Nvfp4Scale,
    Nvfp4GlobalScale,
    Nvfp4InputScale,
    Int8Weight,
    Int8RowScale,
    Int8UnitScale,
    Int8InputScale,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2WeightUploadSpec {
    pub role: M2WeightRole,
    pub name: String,
    pub shard: String,
    pub tensor: M2GraphTensorSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2WeightUploadPlan {
    pub model_dir: PathBuf,
    pub specs: Vec<M2WeightUploadSpec>,
    pub required_tensors: usize,
    pub optional_input_scales_present: usize,
    pub optional_input_scales_missing: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2WeightArenaEntry {
    pub role: M2WeightRole,
    pub name: String,
    pub offset: usize,
    pub nbytes: usize,
    pub shape: Vec<i64>,
    pub dtype: crate::PjrtElementType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2WeightArenaPlan {
    pub alignment: usize,
    pub total_bytes: usize,
    pub entries: Vec<M2WeightArenaEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M2FlatArenaHostBuffer {
    pub bytes: Vec<u8>,
    pub total_bytes: usize,
    pub entries: usize,
}

#[cfg(feature = "tpu")]
pub struct M2UploadedWeightBuffer {
    pub role: M2WeightRole,
    pub name: String,
    pub buffer: PjrtBufferHandle,
}

#[cfg(feature = "tpu")]
pub struct M2UploadedWeights {
    pub buffers: Vec<M2UploadedWeightBuffer>,
    pub total_bytes: usize,
}

impl M2WeightUploadPlan {
    pub fn from_model_dir(model_dir: impl AsRef<Path>, abi: &M2GraphAbi) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let reader = M2SafetensorsReader::open(&model_dir)?;
        Self::from_index(model_dir, reader.index(), abi)
    }

    pub fn from_index_dir(model_dir: impl AsRef<Path>, abi: &M2GraphAbi) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let index =
            M2CheckpointIndex::from_index_file(model_dir.join("model.safetensors.index.json"))?;
        Self::from_index(model_dir, &index, abi)
    }

    pub fn from_index(
        model_dir: PathBuf,
        index: &M2CheckpointIndex,
        abi: &M2GraphAbi,
    ) -> Result<Self> {
        abi.validate_checkpoint_index(index)?;
        let mut specs = Vec::with_capacity(191_069);

        for tensor in &abi.global_weights {
            push_required(&mut specs, index, M2WeightRole::Global, tensor)?;
        }
        for layer in &abi.layer_weights {
            for tensor in &layer.dense {
                push_required(&mut specs, index, M2WeightRole::LayerDense, tensor)?;
            }
            for expert in 0..layer.experts_per_layer {
                for projection in [M2Projection::W1, M2Projection::W2, M2Projection::W3] {
                    let group = M2Nvfp4ProjectionAbi::new(layer.layer, expert, projection);
                    push_required(&mut specs, index, M2WeightRole::Nvfp4Packed, &group.weight)?;
                    push_required(
                        &mut specs,
                        index,
                        M2WeightRole::Nvfp4Scale,
                        &group.weight_scale,
                    )?;
                    push_required(
                        &mut specs,
                        index,
                        M2WeightRole::Nvfp4GlobalScale,
                        &group.weight_scale_2,
                    )?;
                    if index.contains(&group.input_scale.name) {
                        push_required(
                            &mut specs,
                            index,
                            M2WeightRole::Nvfp4InputScale,
                            &group.input_scale,
                        )?;
                    }
                }
            }
        }

        let optional_input_scales_present = specs
            .iter()
            .filter(|spec| spec.role == M2WeightRole::Nvfp4InputScale)
            .count();
        let expected_input_scales = M2_NUM_LAYERS * M2_NUM_EXPERTS * 3;
        let optional_input_scales_missing = expected_input_scales - optional_input_scales_present;
        Ok(Self {
            model_dir,
            required_tensors: specs.len(),
            specs,
            optional_input_scales_present,
            optional_input_scales_missing,
        })
    }

    pub fn total_device_bytes(&self) -> usize {
        self.specs.iter().map(|spec| spec.tensor.nbytes).sum()
    }

    pub fn role_count(&self, role: M2WeightRole) -> usize {
        self.specs.iter().filter(|spec| spec.role == role).count()
    }

    pub fn flat_arena(&self, alignment: usize) -> Result<M2WeightArenaPlan> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(invalid("alignment", "must be a nonzero power of two"));
        }
        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            offset = align_up(offset, alignment);
            entries.push(M2WeightArenaEntry {
                role: spec.role,
                name: spec.name.clone(),
                offset,
                nbytes: spec.tensor.nbytes,
                shape: spec.tensor.shape.clone(),
                dtype: spec.tensor.dtype,
            });
            offset = offset
                .checked_add(spec.tensor.nbytes)
                .ok_or_else(|| invalid("total_bytes", "weight arena overflow"))?;
        }
        Ok(M2WeightArenaPlan {
            alignment,
            total_bytes: align_up(offset, alignment),
            entries,
        })
    }

    pub fn int8_flat_arena(&self, alignment: usize) -> Result<M2WeightArenaPlan> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(invalid("alignment", "must be a nonzero power of two"));
        }
        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            let (role, shape, dtype, nbytes) = match spec.role {
                M2WeightRole::Global | M2WeightRole::LayerDense => (
                    spec.role,
                    spec.tensor.shape.clone(),
                    spec.tensor.dtype,
                    spec.tensor.nbytes,
                ),
                M2WeightRole::Nvfp4Packed => {
                    let packed_shape = packed_shape(&spec.tensor.shape)?;
                    let logical_cols = packed_shape.1 * 2;
                    (
                        M2WeightRole::Int8Weight,
                        vec![packed_shape.0 as i64, logical_cols as i64],
                        crate::PjrtElementType::S8,
                        packed_shape.0 * logical_cols,
                    )
                }
                M2WeightRole::Nvfp4Scale => {
                    let rows = *spec
                        .tensor
                        .shape
                        .first()
                        .ok_or_else(|| invalid("shape", "missing rows"))?
                        as usize;
                    (
                        M2WeightRole::Int8RowScale,
                        vec![rows as i64],
                        crate::PjrtElementType::F32,
                        rows * 4,
                    )
                }
                M2WeightRole::Nvfp4GlobalScale => (
                    M2WeightRole::Int8UnitScale,
                    spec.tensor.shape.clone(),
                    crate::PjrtElementType::F32,
                    4,
                ),
                M2WeightRole::Nvfp4InputScale => (
                    M2WeightRole::Int8InputScale,
                    spec.tensor.shape.clone(),
                    spec.tensor.dtype,
                    spec.tensor.nbytes,
                ),
                M2WeightRole::Int8Weight
                | M2WeightRole::Int8RowScale
                | M2WeightRole::Int8UnitScale
                | M2WeightRole::Int8InputScale => {
                    return Err(invalid("role", "upload plan is already int8"))
                }
            };
            offset = align_up(offset, alignment);
            entries.push(M2WeightArenaEntry {
                role,
                name: spec.name.clone(),
                offset,
                nbytes,
                shape,
                dtype,
            });
            offset = offset
                .checked_add(nbytes)
                .ok_or_else(|| invalid("total_bytes", "weight arena overflow"))?;
        }
        Ok(M2WeightArenaPlan {
            alignment,
            total_bytes: align_up(offset, alignment),
            entries,
        })
    }

    #[cfg(feature = "tpu")]
    pub fn upload_to_pjrt(
        &self,
        reader: &M2SafetensorsReader,
        client: &PjrtClientHandle,
        device_idx: usize,
    ) -> Result<M2UploadedWeights> {
        let mut buffers = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            let view = reader.tensor(&spec.name)?;
            validate_upload_tensor(spec, &view)?;
            let buffer = client.buffer_from_host(
                view.bytes,
                &spec.tensor.shape,
                spec.tensor.dtype,
                device_idx,
            )?;
            buffers.push(M2UploadedWeightBuffer {
                role: spec.role,
                name: spec.name.clone(),
                buffer,
            });
        }
        Ok(M2UploadedWeights {
            buffers,
            total_bytes: self.total_device_bytes(),
        })
    }
}

impl M2WeightArenaPlan {
    pub fn get(&self, name: &str) -> Option<&M2WeightArenaEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn entry(&self, name: &str) -> Result<&M2WeightArenaEntry> {
        self.get(name)
            .ok_or_else(|| invalid_owned("tensor", format!("missing arena tensor: {name}")))
    }

    pub fn materialize_host_buffer(
        &self,
        reader: &M2SafetensorsReader,
        max_bytes: usize,
    ) -> Result<M2FlatArenaHostBuffer> {
        if self.total_bytes > max_bytes {
            return Err(invalid_owned(
                "max_bytes",
                format!(
                    "flat arena is {} bytes, above configured max {}",
                    self.total_bytes, max_bytes
                ),
            ));
        }
        let mut bytes = vec![0u8; self.total_bytes];
        for entry in &self.entries {
            let view = reader.tensor(&entry.name)?;
            validate_arena_tensor(entry, &view)?;
            let start = entry.offset;
            let end = start + entry.nbytes;
            bytes[start..end].copy_from_slice(view.bytes);
        }
        Ok(M2FlatArenaHostBuffer {
            bytes,
            total_bytes: self.total_bytes,
            entries: self.entries.len(),
        })
    }

    pub fn materialize_int8_host_buffer(
        &self,
        reader: &M2SafetensorsReader,
        max_bytes: usize,
    ) -> Result<M2FlatArenaHostBuffer> {
        self.materialize_int8_host_buffer_inner(reader, max_bytes, true)
    }

    pub fn materialize_int8_expert_host_buffer(
        &self,
        reader: &M2SafetensorsReader,
        max_bytes: usize,
    ) -> Result<M2FlatArenaHostBuffer> {
        self.materialize_int8_host_buffer_inner(reader, max_bytes, false)
    }

    fn materialize_int8_host_buffer_inner(
        &self,
        reader: &M2SafetensorsReader,
        max_bytes: usize,
        copy_dense: bool,
    ) -> Result<M2FlatArenaHostBuffer> {
        if self.total_bytes > max_bytes {
            return Err(invalid_owned(
                "max_bytes",
                format!(
                    "flat int8 arena is {} bytes, above configured max {}",
                    self.total_bytes, max_bytes
                ),
            ));
        }
        let mut bytes = vec![0u8; self.total_bytes];
        for entry in &self.entries {
            match entry.role {
                M2WeightRole::Global | M2WeightRole::LayerDense | M2WeightRole::Int8InputScale => {
                    if copy_dense {
                        let view = reader.tensor(&entry.name)?;
                        validate_arena_tensor(entry, &view)?;
                        let start = entry.offset;
                        let end = start + entry.nbytes;
                        bytes[start..end].copy_from_slice(view.bytes);
                    }
                }
                M2WeightRole::Int8Weight => {
                    materialize_int8_weight(entry, self, reader, &mut bytes)?;
                }
                M2WeightRole::Int8RowScale => {}
                M2WeightRole::Int8UnitScale => {
                    let start = entry.offset;
                    let end = start + entry.nbytes;
                    bytes[start..end].copy_from_slice(&1.0f32.to_le_bytes());
                }
                M2WeightRole::Nvfp4Packed
                | M2WeightRole::Nvfp4Scale
                | M2WeightRole::Nvfp4GlobalScale
                | M2WeightRole::Nvfp4InputScale => {
                    return Err(invalid(
                        "role",
                        "use materialize_host_buffer for NVFP4 arenas",
                    ));
                }
            }
        }
        Ok(M2FlatArenaHostBuffer {
            bytes,
            total_bytes: self.total_bytes,
            entries: self.entries.len(),
        })
    }

    #[cfg(feature = "tpu")]
    pub fn upload_flat_arena_to_pjrt(
        &self,
        reader: &M2SafetensorsReader,
        client: &PjrtClientHandle,
        device_idx: usize,
        max_host_bytes: usize,
    ) -> Result<PjrtBufferHandle> {
        let host = self.materialize_host_buffer(reader, max_host_bytes)?;
        client.buffer_from_host(
            &host.bytes,
            &[host.total_bytes as i64],
            crate::PjrtElementType::S8,
            device_idx,
        )
    }

    #[cfg(feature = "tpu")]
    pub fn upload_int8_flat_arena_to_pjrt(
        &self,
        reader: &M2SafetensorsReader,
        client: &PjrtClientHandle,
        device_idx: usize,
        max_host_bytes: usize,
    ) -> Result<PjrtBufferHandle> {
        let host = self.materialize_int8_host_buffer(reader, max_host_bytes)?;
        client.buffer_from_host(
            &host.bytes,
            &[host.total_bytes as i64],
            crate::PjrtElementType::S8,
            device_idx,
        )
    }
}

fn align_up(x: usize, alignment: usize) -> usize {
    (x + alignment - 1) & !(alignment - 1)
}

fn packed_shape(shape: &[i64]) -> Result<(usize, usize)> {
    if shape.len() != 2 || shape[0] <= 0 || shape[1] <= 0 {
        return Err(invalid("shape", "NVFP4 packed tensor must be 2D"));
    }
    Ok((shape[0] as usize, shape[1] as usize))
}

fn materialize_int8_weight(
    entry: &M2WeightArenaEntry,
    arena: &M2WeightArenaPlan,
    reader: &M2SafetensorsReader,
    bytes: &mut [u8],
) -> Result<()> {
    let packed = reader.tensor(&entry.name)?;
    validate_packed_source(entry, &packed)?;
    let scale_name = int8_row_scale_name(&entry.name)?;
    let global_scale_name = int8_unit_scale_name(&entry.name)?;
    let scale = reader.tensor(&scale_name)?;
    let global_scale = reader.tensor(&global_scale_name)?;
    let rows = packed.entry.shape[0];
    let cols = packed.entry.shape[1] * 2;
    if entry.shape != vec![rows as i64, cols as i64] || entry.nbytes != rows * cols {
        return Err(invalid_owned(
            "shape",
            format!(
                "{}: int8 arena shape does not match NVFP4 source",
                entry.name
            ),
        ));
    }
    validate_scale_source(&scale_name, &scale, rows, cols)?;
    let global_scale = global_scale.f32_scalar()?;
    let row_scale_entry = arena.entry(&scale_name)?;
    if row_scale_entry.role != M2WeightRole::Int8RowScale
        || row_scale_entry.dtype != crate::PjrtElementType::F32
        || row_scale_entry.shape != vec![rows as i64]
        || row_scale_entry.nbytes != rows * 4
    {
        return Err(invalid_owned(
            "row_scale",
            format!("{scale_name}: int8 row-scale arena entry mismatch"),
        ));
    }

    let shape = M2Nvfp4MatmulShape {
        m: 1,
        n: rows,
        k: cols,
    };
    let mut i8_weights = vec![0i8; rows * cols];
    let mut row_scales = vec![0.0f32; rows];
    nvfp4_to_int8_matrix(
        packed.bytes,
        scale.bytes,
        global_scale,
        shape,
        &mut i8_weights,
        &mut row_scales,
    )?;

    let weight_start = entry.offset;
    for (dst, src) in bytes[weight_start..weight_start + entry.nbytes]
        .iter_mut()
        .zip(i8_weights.iter())
    {
        *dst = *src as u8;
    }
    let scale_start = row_scale_entry.offset;
    for (row, scale) in row_scales.iter().enumerate() {
        let start = scale_start + row * 4;
        bytes[start..start + 4].copy_from_slice(&scale.to_le_bytes());
    }
    Ok(())
}

fn int8_row_scale_name(weight_name: &str) -> Result<String> {
    let base = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| invalid("weight", "expected .weight suffix"))?;
    Ok(format!("{base}.weight_scale"))
}

fn int8_unit_scale_name(weight_name: &str) -> Result<String> {
    let base = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| invalid("weight", "expected .weight suffix"))?;
    Ok(format!("{base}.weight_scale_2"))
}

fn push_required(
    specs: &mut Vec<M2WeightUploadSpec>,
    index: &M2CheckpointIndex,
    role: M2WeightRole,
    tensor: &M2GraphTensorSpec,
) -> Result<()> {
    let shard = index
        .shard_for(&tensor.name)
        .ok_or_else(|| invalid("tensor", "missing upload tensor"))?;
    specs.push(M2WeightUploadSpec {
        role,
        name: tensor.name.clone(),
        shard: shard.to_string(),
        tensor: tensor.clone(),
    });
    Ok(())
}

#[cfg(any(test, feature = "tpu"))]
fn validate_upload_tensor(spec: &M2WeightUploadSpec, view: &M2TensorView<'_>) -> Result<()> {
    let expected_dtype = pjrt_to_loader_dtype(spec.tensor.dtype)?;
    if view.entry.dtype != expected_dtype {
        return Err(invalid("dtype", "upload tensor dtype mismatch"));
    }
    let expected_shape = spec
        .tensor
        .shape
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();
    if view.entry.shape != expected_shape {
        return Err(invalid("shape", "upload tensor shape mismatch"));
    }
    if view.bytes.len() != spec.tensor.nbytes {
        return Err(invalid("nbytes", "upload tensor byte length mismatch"));
    }
    Ok(())
}

fn validate_arena_tensor(entry: &M2WeightArenaEntry, view: &M2TensorView<'_>) -> Result<()> {
    let expected_dtype = pjrt_to_loader_dtype(entry.dtype)?;
    if view.entry.dtype != expected_dtype {
        return Err(invalid_owned(
            "dtype",
            format!("{}: arena tensor dtype mismatch", entry.name),
        ));
    }
    let expected_shape = entry.shape.iter().map(|&x| x as usize).collect::<Vec<_>>();
    if view.entry.shape != expected_shape {
        return Err(invalid_owned(
            "shape",
            format!("{}: arena tensor shape mismatch", entry.name),
        ));
    }
    if view.bytes.len() != entry.nbytes {
        return Err(invalid_owned(
            "nbytes",
            format!("{}: arena tensor byte length mismatch", entry.name),
        ));
    }
    Ok(())
}

fn validate_packed_source(entry: &M2WeightArenaEntry, view: &M2TensorView<'_>) -> Result<()> {
    if view.entry.dtype != DType::U8 || view.entry.shape.len() != 2 {
        return Err(invalid_owned(
            "packed",
            format!("{}: source must be packed NVFP4 U8", entry.name),
        ));
    }
    let rows = view.entry.shape[0];
    let cols = view.entry.shape[1] * 2;
    if cols % 16 != 0 {
        return Err(invalid_owned(
            "packed",
            format!("{}: logical cols must be multiple of 16", entry.name),
        ));
    }
    if entry.dtype != crate::PjrtElementType::S8
        || entry.shape != vec![rows as i64, cols as i64]
        || entry.nbytes != rows * cols
    {
        return Err(invalid_owned(
            "packed",
            format!("{}: int8 arena entry does not match source", entry.name),
        ));
    }
    Ok(())
}

fn validate_scale_source(
    name: &str,
    view: &M2TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let expected_shape = vec![rows, cols / 16];
    if view.entry.dtype != DType::U8 || view.entry.shape != expected_shape {
        return Err(invalid_owned(
            "scale",
            format!("{name}: source must be FP8 scale U8 shape {expected_shape:?}"),
        ));
    }
    Ok(())
}

fn pjrt_to_loader_dtype(dtype: crate::PjrtElementType) -> Result<DType> {
    Ok(match dtype {
        crate::PjrtElementType::BF16 => DType::Bf16,
        crate::PjrtElementType::F32 => DType::F32,
        crate::PjrtElementType::S8 => DType::I8,
        crate::PjrtElementType::U8 => DType::U8,
        crate::PjrtElementType::S32 => DType::I32,
        _ => return Err(invalid("dtype", "unsupported upload dtype")),
    })
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_weight_plan",
    )
}

fn invalid_owned(field: &'static str, reason: String) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason,
        },
        "m2_weight_plan",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_loader::TensorEntry;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{M2GraphShape, PjrtElementType};

    fn schema_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tpu/harness/m2_checkpoint_schema")
    }

    #[test]
    fn builds_upload_plan_from_real_index_without_python() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        let plan = M2WeightUploadPlan::from_index_dir(schema_dir(), &abi).unwrap();
        assert_eq!(plan.required_tensors, 191_069);
        assert_eq!(plan.specs.len(), 191_069);
        assert_eq!(plan.role_count(M2WeightRole::Global), 3);
        assert_eq!(plan.role_count(M2WeightRole::LayerDense), 620);
        assert_eq!(plan.role_count(M2WeightRole::Nvfp4Packed), 47_616);
        assert_eq!(plan.role_count(M2WeightRole::Nvfp4Scale), 47_616);
        assert_eq!(plan.role_count(M2WeightRole::Nvfp4GlobalScale), 47_616);
        assert_eq!(plan.role_count(M2WeightRole::Nvfp4InputScale), 47_598);
        assert_eq!(plan.optional_input_scales_missing, 18);
    }

    #[test]
    fn upload_plan_carries_shapes_dtypes_and_shards() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        let plan = M2WeightUploadPlan::from_index_dir(schema_dir(), &abi).unwrap();
        let embed = &plan.specs[0];
        assert_eq!(embed.name, "model.embed_tokens.weight");
        assert_eq!(embed.tensor.shape, vec![200_064, 3_072]);
        assert_eq!(embed.shard, "model-00025-of-00025.safetensors");

        let w1 = plan
            .specs
            .iter()
            .find(|spec| spec.name == "model.layers.0.block_sparse_moe.experts.0.w1.weight")
            .unwrap();
        assert_eq!(w1.role, M2WeightRole::Nvfp4Packed);
        assert_eq!(w1.tensor.shape, vec![1_536, 1_536]);
        assert_eq!(w1.tensor.dtype, crate::PjrtElementType::U8);
    }

    #[test]
    fn total_device_bytes_is_the_flat_upload_budget() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        let plan = M2WeightUploadPlan::from_index_dir(schema_dir(), &abi).unwrap();
        assert!(plan.total_device_bytes() > 120_000_000_000);
        assert!(plan.total_device_bytes() < 150_000_000_000);
    }

    #[test]
    fn flat_arena_offsets_all_weight_tensors_without_allocating() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        let plan = M2WeightUploadPlan::from_index_dir(schema_dir(), &abi).unwrap();
        let arena = plan.flat_arena(128).unwrap();
        assert_eq!(arena.alignment, 128);
        assert_eq!(arena.entries.len(), 191_069);
        assert!(arena.total_bytes >= plan.total_device_bytes());
        assert_eq!(arena.total_bytes % 128, 0);
        assert_eq!(arena.entries[0].offset, 0);
        for entry in &arena.entries {
            assert_eq!(entry.offset % 128, 0);
        }
        let w1 = arena
            .entries
            .iter()
            .find(|entry| entry.name == "model.layers.0.block_sparse_moe.experts.0.w1.weight")
            .unwrap();
        assert_eq!(w1.shape, vec![1_536, 1_536]);
        assert_eq!(w1.dtype, PjrtElementType::U8);
        assert_eq!(
            arena
                .entry("model.layers.0.block_sparse_moe.experts.0.w1.weight")
                .unwrap(),
            w1
        );
        assert!(arena.entry("model.layers.99.nope.weight").is_err());
    }

    #[test]
    fn int8_flat_arena_replaces_nvfp4_experts_without_runtime_decode_tensors() {
        let abi = M2GraphAbi::new(M2GraphShape::decode(8, 2048, 1)).unwrap();
        let plan = M2WeightUploadPlan::from_index_dir(schema_dir(), &abi).unwrap();
        let nvfp4 = plan.flat_arena(128).unwrap();
        let int8 = plan.int8_flat_arena(128).unwrap();
        assert_eq!(int8.alignment, 128);
        assert_eq!(int8.entries.len(), nvfp4.entries.len());
        assert!(int8.total_bytes > nvfp4.total_bytes);
        assert_eq!(int8.total_bytes % 128, 0);
        assert_eq!(
            int8.entries
                .iter()
                .filter(|entry| entry.role == M2WeightRole::Int8Weight)
                .count(),
            47_616
        );

        let w1 = int8
            .entry("model.layers.0.block_sparse_moe.experts.0.w1.weight")
            .unwrap();
        assert_eq!(w1.role, M2WeightRole::Int8Weight);
        assert_eq!(w1.shape, vec![1_536, 3_072]);
        assert_eq!(w1.dtype, PjrtElementType::S8);

        let w1_scale = int8
            .entry("model.layers.0.block_sparse_moe.experts.0.w1.weight_scale")
            .unwrap();
        assert_eq!(w1_scale.role, M2WeightRole::Int8RowScale);
        assert_eq!(w1_scale.shape, vec![1_536]);
        assert_eq!(w1_scale.dtype, PjrtElementType::F32);

        let w1_global = int8
            .entry("model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_2")
            .unwrap();
        assert_eq!(w1_global.role, M2WeightRole::Int8UnitScale);
        assert_eq!(w1_global.dtype, PjrtElementType::F32);
    }

    #[test]
    fn upload_tensor_validation_checks_dtype_shape_and_nbytes() {
        let spec = M2WeightUploadSpec {
            role: M2WeightRole::Global,
            name: "x".to_string(),
            shard: "model.safetensors".to_string(),
            tensor: M2GraphTensorSpec {
                name: "x".to_string(),
                shape: vec![2, 3],
                dtype: PjrtElementType::BF16,
                nbytes: 12,
            },
        };
        let entry = TensorEntry {
            name: "x".to_string(),
            dtype: DType::Bf16,
            shape: vec![2, 3],
            file_offset: 0,
            nbytes: 12,
        };
        let bytes = [0u8; 12];
        let view = M2TensorView {
            entry: &entry,
            bytes: &bytes,
        };
        validate_upload_tensor(&spec, &view).unwrap();

        let bad_entry = TensorEntry {
            shape: vec![3, 2],
            ..entry.clone()
        };
        let bad_view = M2TensorView {
            entry: &bad_entry,
            bytes: &bytes,
        };
        assert!(validate_upload_tensor(&spec, &bad_view).is_err());
    }

    #[test]
    fn materializes_flat_arena_host_buffer_with_padding() {
        let dir = tiny_model_dir();
        let reader = M2SafetensorsReader::open(&dir).unwrap();
        let arena = M2WeightArenaPlan {
            alignment: 8,
            total_bytes: 16,
            entries: vec![
                M2WeightArenaEntry {
                    role: M2WeightRole::Global,
                    name: "a".to_string(),
                    offset: 0,
                    nbytes: 3,
                    shape: vec![3],
                    dtype: PjrtElementType::U8,
                },
                M2WeightArenaEntry {
                    role: M2WeightRole::Global,
                    name: "b".to_string(),
                    offset: 8,
                    nbytes: 4,
                    shape: vec![1],
                    dtype: PjrtElementType::F32,
                },
            ],
        };
        let host = arena.materialize_host_buffer(&reader, 64).unwrap();
        assert_eq!(host.total_bytes, 16);
        assert_eq!(host.entries, 2);
        assert_eq!(&host.bytes[0..3], &[1, 2, 3]);
        assert_eq!(&host.bytes[3..8], &[0, 0, 0, 0, 0]);
        assert_eq!(&host.bytes[8..12], &[0x78, 0x56, 0x34, 0x12]);
        assert!(arena.materialize_host_buffer(&reader, 8).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn materializes_boot_time_int8_arena_from_nvfp4_sources() {
        let dir = tiny_nvfp4_model_dir();
        let reader = M2SafetensorsReader::open(&dir).unwrap();
        let arena = M2WeightArenaPlan {
            alignment: 8,
            total_bytes: 48,
            entries: vec![
                M2WeightArenaEntry {
                    role: M2WeightRole::Int8Weight,
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.weight".to_string(),
                    offset: 0,
                    nbytes: 32,
                    shape: vec![2, 16],
                    dtype: PjrtElementType::S8,
                },
                M2WeightArenaEntry {
                    role: M2WeightRole::Int8RowScale,
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale".to_string(),
                    offset: 32,
                    nbytes: 8,
                    shape: vec![2],
                    dtype: PjrtElementType::F32,
                },
                M2WeightArenaEntry {
                    role: M2WeightRole::Int8UnitScale,
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_2".to_string(),
                    offset: 40,
                    nbytes: 4,
                    shape: vec![],
                    dtype: PjrtElementType::F32,
                },
            ],
        };
        let host = arena.materialize_int8_host_buffer(&reader, 64).unwrap();
        assert_eq!(host.total_bytes, 48);
        assert!(host.bytes[0..32].iter().all(|&x| x == 127));
        assert_eq!(
            f32::from_le_bytes(host.bytes[32..36].try_into().unwrap()),
            1.0 / 127.0
        );
        assert_eq!(
            f32::from_le_bytes(host.bytes[36..40].try_into().unwrap()),
            1.0 / 127.0
        );
        assert_eq!(
            f32::from_le_bytes(host.bytes[40..44].try_into().unwrap()),
            1.0
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn tiny_model_dir() -> PathBuf {
        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("rvllm-xla-flat-arena-{uniq}"));
        fs::create_dir_all(&dir).unwrap();
        let shard_name = "model-00001-of-00001.safetensors";
        let payload = [1u8, 2, 3, 0x78, 0x56, 0x34, 0x12];
        let header = json!({
            "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [3, 7]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header_bytes);
        shard.extend_from_slice(&payload);
        fs::write(dir.join(shard_name), shard).unwrap();
        let index = json!({
            "metadata": {},
            "weight_map": {
                "a": shard_name,
                "b": shard_name
            }
        });
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        dir
    }

    fn tiny_nvfp4_model_dir() -> PathBuf {
        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("rvllm-xla-int8-arena-{uniq}"));
        fs::create_dir_all(&dir).unwrap();
        let shard_name = "model-00001-of-00001.safetensors";
        let weight = [0x11u8; 16];
        let scale = [0x38u8; 2];
        let global = 2.0f32.to_le_bytes();
        let mut payload = Vec::new();
        payload.extend_from_slice(&weight);
        payload.extend_from_slice(&scale);
        payload.extend_from_slice(&global);
        let header = json!({
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": {
                "dtype": "U8", "shape": [2, 8], "data_offsets": [0, 16]
            },
            "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": {
                "dtype": "U8", "shape": [2, 1], "data_offsets": [16, 18]
            },
            "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_2": {
                "dtype": "F32", "shape": [], "data_offsets": [18, 22]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header_bytes);
        shard.extend_from_slice(&payload);
        fs::write(dir.join(shard_name), shard).unwrap();
        let index = json!({
            "metadata": {},
            "weight_map": {
                "model.layers.0.block_sparse_moe.experts.0.w1.weight": shard_name,
                "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": shard_name,
                "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_2": shard_name
            }
        });
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        dir
    }
}
