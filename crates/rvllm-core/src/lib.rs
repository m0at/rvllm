#![forbid(unsafe_code)]
//! Core primitives for vllm-rs: shared types, error hierarchy, config trait
//! interfaces, and output structures used across every other crate.

pub mod config;
pub mod error;
pub mod hf;
pub mod output;
pub mod prelude;
pub mod types;
