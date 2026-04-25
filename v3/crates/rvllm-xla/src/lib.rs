#![forbid(unsafe_code)]

pub mod artifact;
pub mod ffi;

pub use artifact::{
    load_artifact, m2_prefill_artifact_manifest, write_m2_prefill_artifact, XlaArtifact,
    XlaTensorSpec,
};
pub use ffi::PjrtElementType;
