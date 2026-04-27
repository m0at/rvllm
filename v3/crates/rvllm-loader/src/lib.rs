//! rvllm-loader: HF safetensors → GPU, with FP8 quant at load.
//!
//! The invariants:
//! - Weights are stored in typed fields, not parallel Vecs indexed by
//!   integer (v2's frequent desync source).
//! - FP8 per-tensor quant runs the clamp-% gate; a tensor exceeding
//!   10 ppm clamp rate returns `LoaderError::Fp8MisScaled` — the model
//!   is mis-scaled, not a viable FP8 candidate, and the engine refuses
//!   to proceed.
//! - Full weight set resident before first forward; no lazy loading.

pub mod compressed_tensors;
pub mod fp8_quant;
pub mod gemma4_arch;
pub mod gemma4_load;
pub mod gemma4_weights;
pub mod load;
pub mod load_multiformat;
pub mod safetensors;
pub mod weights;

pub use fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, QuantResult, FP8_E4M3_MAX};
pub use load::{load_model, LayerAttnType, MlpActivation, ModelArch};
pub use safetensors::{ShardHeader, ShardIndex, TensorEntry};
pub use weights::{F16Weight, Fp8Weight, LayerWeights, LoadedModel};
