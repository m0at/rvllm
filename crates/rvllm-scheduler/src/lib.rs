#![forbid(unsafe_code)]
//! Continuous batching scheduler for vllm-rs.
//!
//! Decides which sequences run, wait, or get preempted each iteration.
//! Supports chunked prefill, multiple scheduling policies, and swap/recompute
//! preemption modes.

pub mod outputs;
pub mod policy;
pub mod scheduler;

pub use outputs::{ScheduledSequenceGroup, SchedulerOutputs};
pub use policy::{PreemptionMode, SchedulerPolicy};
pub use scheduler::{Scheduler, SchedulerConfig};
