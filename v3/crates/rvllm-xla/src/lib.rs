pub mod artifact;
#[cfg(feature = "tpu")]
pub mod client;
pub mod executable;
pub mod ffi;
pub mod m2_decode_bench;
pub mod m2_decode_graph;
pub mod m2_graph_abi;
pub mod m2_prefill;
pub mod m2_runtime;
pub mod m2_tpu_custom_call;
pub mod m2_weight_plan;

pub use artifact::{
    load_artifact, m2_prefill_artifact_manifest, write_m2_prefill_artifact, XlaArtifact,
    XlaTensorSpec,
};
#[cfg(feature = "tpu")]
pub use client::{CompiledExecutable, PjrtBufferHandle, PjrtClientHandle};
pub use executable::{
    tensor_nbytes, validate_argument_specs, PjrtExecutableSignature, PjrtHostBuffer,
    PjrtProgramFormat, PjrtTensorSpec,
};
pub use ffi::PjrtElementType;
pub use m2_decode_bench::{
    plan_m2_rust_decode_bench, M2RustDecodeBenchConfig, M2RustDecodeBenchReport,
    M2RustDecodeRuntimeReport, M2RustDecodeSweepItem,
};
pub use m2_decode_graph::{
    m2_decode_graph_mlir, m2_decode_graph_mlir_with_mosaic_body, m2_decode_smoke_mlir,
    M2ArenaTensor, M2DecodeGraphPlan, M2DecodeLayerPlan, M2ExpertDirectoryEntry, M2ExpertPlan,
    M2Nvfp4ProjectionPlan,
};
pub use m2_graph_abi::{
    M2GraphAbi, M2GraphPhase, M2GraphShape, M2GraphTensorSpec, M2LayerWeightAbi,
    M2Nvfp4ProjectionAbi, M2_HEAD_DIM, M2_HIDDEN, M2_MOE_INTER, M2_NUM_EXPERTS, M2_NUM_KV_HEADS,
    M2_NUM_LAYERS, M2_NUM_Q_HEADS, M2_NVFP4_GROUP, M2_ROTARY_DIM, M2_TOP_K, M2_VOCAB,
};
pub use m2_prefill::{
    make_m2_prefill_input_specs, make_m2_prefill_inputs, M2PrefillHostInput, M2PrefillHostInputSpec,
};
pub use m2_runtime::{
    m2_bf16_logits_nll, m2_ppl_from_nll, plan_m2_rust_prefill, plan_m2_rust_prefill_decode,
    M2DecodeRuntimeInputSpec, M2GenerateOutput, M2GenerateRequest, M2PplResult, M2Runtime,
    M2RuntimeConfig, M2RustPrefillConfig, M2RustPrefillDecodeConfig, M2RustPrefillDecodePlan,
    M2RustPrefillPlan,
};
pub use m2_tpu_custom_call::{
    tpu_custom_call_backend_config, tpu_custom_call_backend_config_for_body, TpuMosaicBodyFormat,
    TpuMosaicSerializedBody, TPU_CUSTOM_CALL_TARGET, TPU_MOSAIC_BYTECODE_VERSION,
    TPU_MOSAIC_SERDE_PASS, TPU_MOSAIC_SERIALIZATION_FORMAT,
};
pub use m2_weight_plan::{
    M2FlatArenaHostBuffer, M2WeightArenaEntry, M2WeightArenaPlan, M2WeightRole, M2WeightUploadPlan,
    M2WeightUploadSpec,
};
#[cfg(feature = "tpu")]
pub use m2_weight_plan::{M2UploadedWeightBuffer, M2UploadedWeights};
