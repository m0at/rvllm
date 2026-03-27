//! Layer primitives for transformer models.

pub mod activation;
#[cfg(feature = "cuda")]
pub mod activation_cuda;
pub mod linear;
#[cfg(feature = "cuda")]
pub mod linear_cuda;
pub mod mlp;
pub mod moe;
pub mod norm;
#[cfg(feature = "cuda")]
pub mod norm_cuda;
pub mod rotary;
#[cfg(feature = "cuda")]
pub mod rotary_cuda;
#[cfg(feature = "cuda")]
pub mod softmax_cuda;
#[cfg(feature = "cuda")]
pub mod fused_ops;
