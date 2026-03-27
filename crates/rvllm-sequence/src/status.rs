use serde::{Deserialize, Serialize};
use rvllm_core::prelude::LLMError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    FinishedStopped,
    FinishedLength,
    FinishedAborted,
}

impl SequenceStatus {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            Self::FinishedStopped | Self::FinishedLength | Self::FinishedAborted
        )
    }

    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Validate that transitioning from `self` to `target` is legal.
    /// Finished states are terminal -- no transitions out.
    pub fn validate_transition(&self, target: SequenceStatus) -> rvllm_core::prelude::Result<()> {
        if self.is_finished() {
            return Err(LLMError::SchedulerError(format!(
                "cannot transition from terminal state {:?} to {:?}",
                self, target
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finished_variants() {
        assert!(SequenceStatus::FinishedStopped.is_finished());
        assert!(SequenceStatus::FinishedLength.is_finished());
        assert!(SequenceStatus::FinishedAborted.is_finished());
        assert!(!SequenceStatus::Waiting.is_finished());
        assert!(!SequenceStatus::Running.is_finished());
        assert!(!SequenceStatus::Swapped.is_finished());
    }

    #[test]
    fn running_variant() {
        assert!(SequenceStatus::Running.is_running());
        assert!(!SequenceStatus::Waiting.is_running());
        assert!(!SequenceStatus::FinishedStopped.is_running());
    }

    #[test]
    fn transition_from_finished_is_error() {
        let s = SequenceStatus::FinishedStopped;
        assert!(s.validate_transition(SequenceStatus::Running).is_err());
    }

    #[test]
    fn transition_from_waiting_ok() {
        let s = SequenceStatus::Waiting;
        assert!(s.validate_transition(SequenceStatus::Running).is_ok());
    }

    #[test]
    fn serde_roundtrip() {
        let s = SequenceStatus::Swapped;
        let json = serde_json::to_string(&s).unwrap();
        let s2: SequenceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }
}
