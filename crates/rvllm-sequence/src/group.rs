use std::time::Instant;

use rvllm_core::prelude::{RequestId, SamplingParams};

use crate::sequence::Sequence;
use crate::status::SequenceStatus;

#[derive(Debug, Clone)]
pub struct SequenceGroup {
    pub request_id: RequestId,
    pub seqs: Vec<Sequence>,
    pub sampling_params: SamplingParams,
    pub arrival_time: Instant,
    pub prompt_text: String,
}

impl SequenceGroup {
    pub fn new(
        request_id: RequestId,
        seqs: Vec<Sequence>,
        sampling_params: SamplingParams,
        arrival_time: Instant,
        prompt_text: String,
    ) -> Self {
        Self {
            request_id,
            seqs,
            sampling_params,
            arrival_time,
            prompt_text,
        }
    }

    pub fn get_seqs(&self) -> &[Sequence] {
        &self.seqs
    }

    pub fn get_seqs_mut(&mut self) -> &mut [Sequence] {
        &mut self.seqs
    }

    pub fn get_seqs_by_status(&self, status: SequenceStatus) -> Vec<&Sequence> {
        self.seqs.iter().filter(|s| s.status == status).collect()
    }

    pub fn num_seqs(&self, status: Option<SequenceStatus>) -> usize {
        match status {
            Some(st) => self.seqs.iter().filter(|s| s.status == st).count(),
            None => self.seqs.len(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|s| s.is_finished())
    }

    /// Maximum number of sequences that could be running simultaneously.
    /// For beam search / best-of-N this is `best_of`; otherwise 1.
    pub fn get_max_num_running_seqs(&self) -> usize {
        if self.is_finished() {
            return 0;
        }
        self.sampling_params.best_of.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::SequenceId;

    fn make_group(n_seqs: usize) -> SequenceGroup {
        let seqs: Vec<Sequence> = (0..n_seqs)
            .map(|i| Sequence::new(SequenceId(i as u64), vec![1, 2, 3]))
            .collect();
        SequenceGroup::new(
            RequestId(0),
            seqs,
            SamplingParams::default(),
            Instant::now(),
            "hello".into(),
        )
    }

    #[test]
    fn basic_accessors() {
        let g = make_group(3);
        assert_eq!(g.get_seqs().len(), 3);
        assert_eq!(g.num_seqs(None), 3);
        assert_eq!(g.num_seqs(Some(SequenceStatus::Waiting)), 3);
        assert_eq!(g.num_seqs(Some(SequenceStatus::Running)), 0);
    }

    #[test]
    fn get_seqs_by_status() {
        let mut g = make_group(3);
        g.seqs[0].set_status(SequenceStatus::Running).unwrap();
        let running = g.get_seqs_by_status(SequenceStatus::Running);
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].seq_id, SequenceId(0));
    }

    #[test]
    fn is_finished() {
        let mut g = make_group(2);
        assert!(!g.is_finished());
        g.seqs[0]
            .set_status(SequenceStatus::FinishedStopped)
            .unwrap();
        assert!(!g.is_finished());
        g.seqs[1]
            .set_status(SequenceStatus::FinishedLength)
            .unwrap();
        assert!(g.is_finished());
    }

    #[test]
    fn max_running_seqs() {
        let g = make_group(1);
        assert_eq!(g.get_max_num_running_seqs(), 1);

        let mut g2 = make_group(1);
        g2.sampling_params.best_of = 4;
        assert_eq!(g2.get_max_num_running_seqs(), 4);

        let mut g3 = make_group(1);
        g3.seqs[0]
            .set_status(SequenceStatus::FinishedAborted)
            .unwrap();
        assert_eq!(g3.get_max_num_running_seqs(), 0);
    }

    #[test]
    fn get_seqs_mut() {
        let mut g = make_group(2);
        g.get_seqs_mut()[0].append_token(99, -0.1);
        assert_eq!(g.seqs[0].get_output_len(), 1);
    }
}
