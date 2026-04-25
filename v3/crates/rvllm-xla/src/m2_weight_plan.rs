use std::path::{Path, PathBuf};

#[cfg(any(test, feature = "tpu"))]
use rvllm_core::DType;
use rvllm_core::{ConfigError, Result, RvllmError};
#[cfg(any(test, feature = "tpu"))]
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

#[cfg(any(test, feature = "tpu"))]
fn pjrt_to_loader_dtype(dtype: crate::PjrtElementType) -> Result<DType> {
    Ok(match dtype {
        crate::PjrtElementType::BF16 => DType::Bf16,
        crate::PjrtElementType::F32 => DType::F32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_loader::TensorEntry;

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
}
