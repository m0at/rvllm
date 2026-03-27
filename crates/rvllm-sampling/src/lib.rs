//! Sampling and logit processing for vllm-rs.
//!
//! Converts raw logits from the model into token selections via temperature
//! scaling, top-k/p/min-p filtering, repetition penalties, and multinomial
//! or greedy decoding.

pub mod batch;
pub mod gpu_logprobs;
pub mod gpu_sampler;
pub mod guided;
pub mod json_schema;
pub mod logit_processors;
pub mod math;
pub mod sampler;

pub use batch::{sample_batch, sample_batch_parallel};
pub use gpu_logprobs::{
    compute_batch_logprobs, compute_position_logprobs, compute_prompt_logprobs,
    logprobs_to_output_format, PositionLogprobs,
};
pub use gpu_sampler::sample_from_gpu_logits;
pub use logit_processors::*;
pub use math::*;
pub use guided::{apply_guided_mask, GuidedDecodingState, VocabEntry, VocabTable};
pub use json_schema::{compile_schema, SchemaNode, ValidChars};
pub use sampler::{Sampler, SamplerOutput};
