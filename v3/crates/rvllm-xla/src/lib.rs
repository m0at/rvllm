pub mod artifact;
#[cfg(feature = "tpu")]
pub mod client;
pub mod ffi;
pub mod m2_decode_bench;
pub mod m2_graph_abi;
pub mod m2_prefill;
pub mod m2_runtime;
pub mod m2_weight_plan;

pub use artifact::{
    load_artifact, m2_prefill_artifact_manifest, write_m2_prefill_artifact, XlaArtifact,
    XlaTensorSpec,
};
#[cfg(feature = "tpu")]
pub use client::{CompiledExecutable, PjrtBufferHandle, PjrtClientHandle};
pub use ffi::PjrtElementType;
pub use m2_decode_bench::{
    plan_m2_rust_decode_bench, M2RustDecodeBenchConfig, M2RustDecodeBenchReport,
    M2RustDecodeRuntimeReport, M2RustDecodeSweepItem,
};
pub use m2_graph_abi::{
    M2GraphAbi, M2GraphPhase, M2GraphShape, M2GraphTensorSpec, M2LayerWeightAbi,
    M2Nvfp4ProjectionAbi, M2_HEAD_DIM, M2_HIDDEN, M2_MOE_INTER, M2_NUM_EXPERTS, M2_NUM_KV_HEADS,
    M2_NUM_LAYERS, M2_NUM_Q_HEADS, M2_NVFP4_GROUP, M2_ROTARY_DIM, M2_TOP_K, M2_VOCAB,
};
pub use m2_prefill::{
    make_m2_prefill_input_specs, make_m2_prefill_inputs, M2PrefillHostInput, M2PrefillHostInputSpec,
};
pub use m2_runtime::{plan_m2_rust_prefill, M2RustPrefillConfig, M2RustPrefillPlan};
#[cfg(feature = "tpu")]
pub use m2_weight_plan::{M2UploadedWeightBuffer, M2UploadedWeights};
pub use m2_weight_plan::{M2WeightRole, M2WeightUploadPlan, M2WeightUploadSpec};
