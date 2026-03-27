//! Beam search decoding: maintain K beams, expand each by top-K tokens,
//! prune to the K best candidates by cumulative log-probability.
//!
//! Designed to integrate with the engine's step loop and the block manager's
//! `fork()` / CoW mechanism for KV cache sharing between beams.

use std::collections::HashMap;

use tracing::debug;

use rvllm_core::prelude::{
    CompletionOutput, FinishReason, LogProb, RequestId, RequestOutput, SequenceId, TokenId,
};

use crate::output::SequenceOutputState;

/// A single beam hypothesis being tracked during search.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Sequence id this beam corresponds to in the engine.
    pub seq_id: SequenceId,
    /// Accumulated token ids (output only, not prompt).
    pub token_ids: Vec<TokenId>,
    /// Decoded text so far.
    pub text: String,
    /// Sum of log-probabilities of generated tokens.
    pub cumulative_logprob: LogProb,
    /// Per-position top logprobs if requested.
    pub logprobs: Vec<Vec<(TokenId, LogProb)>>,
    /// Whether this beam has terminated (EOS or stop condition).
    pub finished: bool,
    /// Reason for finishing, if applicable.
    pub finish_reason: Option<FinishReason>,
}

impl BeamHypothesis {
    /// Create a new beam from its initial sequence id.
    pub fn new(seq_id: SequenceId) -> Self {
        Self {
            seq_id,
            token_ids: Vec::new(),
            text: String::new(),
            cumulative_logprob: 0.0,
            logprobs: Vec::new(),
            finished: false,
            finish_reason: None,
        }
    }

    /// Length-normalized score for ranking beams.
    ///
    /// Uses the Wu et al. (2016) length penalty:
    /// `score = logprob / ((5 + len) / 6)^alpha`
    pub fn score(&self, length_penalty: f32) -> f32 {
        let len = self.token_ids.len().max(1) as f32;
        let penalty = ((5.0 + len) / 6.0).powf(length_penalty);
        self.cumulative_logprob / penalty
    }
}

/// A candidate expansion: extending a beam by one token.
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    /// Index of the parent beam in the current beam set.
    pub parent_beam_idx: usize,
    /// The token to append.
    pub token_id: TokenId,
    /// Log-probability of this token.
    pub logprob: LogProb,
    /// Cumulative logprob including this token.
    pub cumulative_logprob: LogProb,
    /// Decoded text for this token.
    pub decoded_text: String,
    /// Top logprobs at this position (if requested).
    pub top_logprobs: Option<Vec<(TokenId, LogProb)>>,
    /// Whether this token terminates the beam.
    pub is_terminal: bool,
    /// Finish reason if terminal.
    pub finish_reason: Option<FinishReason>,
}

/// Manages beam search state for a single request.
#[derive(Debug)]
pub struct BeamSearchState {
    /// Request this search belongs to.
    pub request_id: RequestId,
    /// Number of beams (K).
    pub num_beams: usize,
    /// Length penalty alpha for scoring.
    pub length_penalty: f32,
    /// Whether to stop early when `num_beams` finished hypotheses have scores
    /// better than the best active beam can achieve.
    pub early_stopping: bool,
    /// Active beams being expanded.
    pub active_beams: Vec<BeamHypothesis>,
    /// Completed hypotheses (finished beams).
    pub completed: Vec<BeamHypothesis>,
    /// Maximum output tokens.
    pub max_tokens: usize,
}

impl BeamSearchState {
    /// Create a new beam search with K beams.
    pub fn new(
        request_id: RequestId,
        num_beams: usize,
        max_tokens: usize,
        length_penalty: f32,
        early_stopping: bool,
        initial_seq_ids: &[SequenceId],
    ) -> Self {
        let active_beams = initial_seq_ids
            .iter()
            .map(|&sid| BeamHypothesis::new(sid))
            .collect();

        Self {
            request_id,
            num_beams,
            length_penalty,
            early_stopping,
            active_beams,
            completed: Vec::new(),
            max_tokens,
        }
    }

    /// Whether beam search is complete: all beams finished or early stop.
    pub fn is_finished(&self) -> bool {
        if self.active_beams.is_empty() {
            return true;
        }
        if self.early_stopping && self.completed.len() >= self.num_beams {
            // Check if the worst completed beam is better than the best
            // possible score from any active beam.
            let worst_completed = self
                .completed
                .iter()
                .map(|h| h.score(self.length_penalty))
                .fold(f32::INFINITY, f32::min);
            let best_active = self
                .active_beams
                .iter()
                .map(|h| h.cumulative_logprob)
                .fold(f32::NEG_INFINITY, f32::max);
            worst_completed >= best_active
        } else {
            false
        }
    }

    /// Process one step of beam search given per-sequence top-K token expansions.
    ///
    /// `expansions` maps `seq_id -> Vec<(token_id, logprob, decoded_text, is_eos)>`.
    /// Each vec should contain the top-K tokens for that beam.
    ///
    /// Returns a `BeamStepResult` describing which beams to keep, which
    /// parent sequences to fork from, and which sequences to free.
    pub fn step(
        &mut self,
        expansions: &HashMap<SequenceId, Vec<(TokenId, LogProb, String, bool)>>,
    ) -> BeamStepResult {
        let mut candidates: Vec<BeamCandidate> = Vec::new();

        for (beam_idx, beam) in self.active_beams.iter().enumerate() {
            if beam.finished {
                continue;
            }

            let tokens = match expansions.get(&beam.seq_id) {
                Some(t) => t,
                None => continue,
            };

            for &(token_id, logprob, ref decoded, is_eos) in tokens {
                let cum_lp = beam.cumulative_logprob + logprob;
                let new_len = beam.token_ids.len() + 1;
                let is_terminal =
                    is_eos || new_len >= self.max_tokens;
                let finish_reason = if is_eos {
                    Some(FinishReason::Stop)
                } else if new_len >= self.max_tokens {
                    Some(FinishReason::Length)
                } else {
                    None
                };

                candidates.push(BeamCandidate {
                    parent_beam_idx: beam_idx,
                    token_id,
                    logprob,
                    cumulative_logprob: cum_lp,
                    decoded_text: decoded.clone(),
                    top_logprobs: None,
                    is_terminal,
                    finish_reason,
                });
            }
        }

        // Sort candidates by cumulative logprob descending.
        candidates.sort_by(|a, b| {
            b.cumulative_logprob
                .partial_cmp(&a.cumulative_logprob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_beams: Vec<BeamHypothesis> = Vec::new();
        let mut fork_ops: Vec<ForkOp> = Vec::new();
        let mut seqs_to_free: Vec<SequenceId> = Vec::new();

        // Track which old beams are reused (at most once directly, rest are forks).
        let mut old_beam_used: Vec<bool> = vec![false; self.active_beams.len()];

        for candidate in &candidates {
            if new_beams.len() >= self.num_beams {
                break;
            }

            let parent = &self.active_beams[candidate.parent_beam_idx];

            // Build the new hypothesis.
            let mut hyp = BeamHypothesis {
                seq_id: parent.seq_id, // Will be updated if forked.
                token_ids: parent.token_ids.clone(),
                text: parent.text.clone(),
                cumulative_logprob: candidate.cumulative_logprob,
                logprobs: parent.logprobs.clone(),
                finished: candidate.is_terminal,
                finish_reason: candidate.finish_reason,
            };
            hyp.token_ids.push(candidate.token_id);
            hyp.text.push_str(&candidate.decoded_text);
            if let Some(ref top) = candidate.top_logprobs {
                hyp.logprobs.push(top.clone());
            }

            if candidate.is_terminal {
                self.completed.push(hyp);
            } else if !old_beam_used[candidate.parent_beam_idx] {
                // Reuse the parent's sequence directly (no fork needed).
                old_beam_used[candidate.parent_beam_idx] = true;
                new_beams.push(hyp);
            } else {
                // Need to fork from parent.
                fork_ops.push(ForkOp {
                    parent_seq_id: parent.seq_id,
                    new_beam_idx: new_beams.len(),
                });
                // seq_id will be filled in by the caller after fork.
                hyp.seq_id = SequenceId(u64::MAX); // placeholder
                new_beams.push(hyp);
            }
        }

        // Free old beams that are not reused.
        for (i, beam) in self.active_beams.iter().enumerate() {
            if !old_beam_used[i] {
                seqs_to_free.push(beam.seq_id);
            }
        }

        let prev_active = self.active_beams.len();
        self.active_beams = new_beams;

        debug!(
            request_id = %self.request_id,
            prev_active,
            new_active = self.active_beams.len(),
            completed = self.completed.len(),
            forks = fork_ops.len(),
            freed = seqs_to_free.len(),
            "beam search step"
        );

        BeamStepResult {
            fork_ops,
            seqs_to_free,
        }
    }

    /// Update a beam's sequence id after a fork operation.
    pub fn set_beam_seq_id(&mut self, beam_idx: usize, seq_id: SequenceId) {
        if let Some(beam) = self.active_beams.get_mut(beam_idx) {
            beam.seq_id = seq_id;
        }
    }

    /// Build the final `RequestOutput` from the best completed hypothesis.
    ///
    /// If `n_best` > 1, returns the top N completed hypotheses sorted by score.
    pub fn build_output(
        &self,
        prompt: &str,
        prompt_token_ids: &[TokenId],
        n_best: usize,
    ) -> RequestOutput {
        let mut sorted: Vec<&BeamHypothesis> = self.completed.iter().collect();
        sorted.sort_by(|a, b| {
            let sa = a.score(self.length_penalty);
            let sb = b.score(self.length_penalty);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let take = n_best.min(sorted.len()).max(1);
        let outputs: Vec<CompletionOutput> = sorted[..take]
            .iter()
            .enumerate()
            .map(|(i, hyp)| CompletionOutput {
                index: i,
                text: hyp.text.clone(),
                token_ids: hyp.token_ids.clone(),
                cumulative_logprob: hyp.cumulative_logprob,
                logprobs: if hyp.logprobs.is_empty() {
                    None
                } else {
                    Some(hyp.logprobs.clone())
                },
                finish_reason: hyp.finish_reason,
            })
            .collect();

        // If no completed beams, fall back to best active beam.
        let outputs = if outputs.is_empty() && !self.active_beams.is_empty() {
            let best = self
                .active_beams
                .iter()
                .max_by(|a, b| {
                    a.cumulative_logprob
                        .partial_cmp(&b.cumulative_logprob)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            vec![CompletionOutput {
                index: 0,
                text: best.text.clone(),
                token_ids: best.token_ids.clone(),
                cumulative_logprob: best.cumulative_logprob,
                logprobs: None,
                finish_reason: Some(FinishReason::Length),
            }]
        } else {
            outputs
        };

        RequestOutput {
            request_id: self.request_id,
            prompt: prompt.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            prompt_logprobs: None,
            outputs,
            finished: self.is_finished() || !self.completed.is_empty(),
        }
    }
}

/// Instructions from a beam search step about sequence management.
#[derive(Debug)]
pub struct BeamStepResult {
    /// Fork operations: create child sequences from parents.
    pub fork_ops: Vec<ForkOp>,
    /// Sequences that should be freed (pruned beams).
    pub seqs_to_free: Vec<SequenceId>,
}

/// A fork operation: create a new sequence from an existing parent.
#[derive(Debug)]
pub struct ForkOp {
    /// The parent sequence to fork from (shares KV cache via CoW).
    pub parent_seq_id: SequenceId,
    /// Index in the new active_beams vec to update with the new seq_id.
    pub new_beam_idx: usize,
}

/// Convenience: extract the top-K tokens from logprobs for beam expansion.
///
/// Given per-sequence logprobs (from the sampling layer), returns the top K
/// (token_id, logprob) pairs sorted by logprob descending.
pub fn top_k_from_logprobs(
    logprobs: &[(TokenId, LogProb)],
    k: usize,
) -> Vec<(TokenId, LogProb)> {
    let mut sorted: Vec<(TokenId, LogProb)> = logprobs.to_vec();
    sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.truncate(k);
    sorted
}

/// Build SequenceOutputState entries from completed beam hypotheses
/// (for compatibility with the output processor).
pub fn beam_to_output_states(state: &BeamSearchState) -> Vec<SequenceOutputState> {
    let mut sorted: Vec<&BeamHypothesis> = state.completed.iter().collect();
    sorted.sort_by(|a, b| {
        let sa = a.score(state.length_penalty);
        let sb = b.score(state.length_penalty);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    sorted
        .iter()
        .map(|hyp| SequenceOutputState {
            text: hyp.text.clone(),
            token_ids: hyp.token_ids.clone(),
            cumulative_logprob: hyp.cumulative_logprob,
            logprobs: hyp.logprobs.clone(),
            finish_reason: hyp.finish_reason,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(num_beams: usize) -> BeamSearchState {
        let seq_ids: Vec<SequenceId> = (0..num_beams)
            .map(|i| SequenceId(i as u64))
            .collect();
        BeamSearchState::new(
            RequestId(1),
            num_beams,
            10,
            1.0,
            false,
            &seq_ids,
        )
    }

    #[test]
    fn beam_hypothesis_score() {
        let mut h = BeamHypothesis::new(SequenceId(0));
        h.cumulative_logprob = -6.0;
        h.token_ids = vec![1, 2, 3];
        // len=3, penalty = ((5+3)/6)^1.0 = 8/6 = 1.333
        let score = h.score(1.0);
        assert!((score - (-6.0 / (8.0 / 6.0))).abs() < 1e-4);
    }

    #[test]
    fn beam_hypothesis_score_zero_penalty() {
        let mut h = BeamHypothesis::new(SequenceId(0));
        h.cumulative_logprob = -3.0;
        h.token_ids = vec![1, 2];
        let score = h.score(0.0);
        // penalty = ((5+2)/6)^0 = 1.0
        assert!((score - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn new_state_has_active_beams() {
        let state = make_state(3);
        assert_eq!(state.active_beams.len(), 3);
        assert!(state.completed.is_empty());
        assert!(!state.is_finished());
    }

    #[test]
    fn step_prunes_to_num_beams() {
        let mut state = make_state(2);

        let mut expansions = HashMap::new();
        // Beam 0: two expansions
        expansions.insert(
            SequenceId(0),
            vec![
                (10u32, -0.5f32, "a".to_string(), false),
                (11, -1.0, "b".to_string(), false),
            ],
        );
        // Beam 1: two expansions
        expansions.insert(
            SequenceId(1),
            vec![
                (20, -0.3, "c".to_string(), false),
                (21, -2.0, "d".to_string(), false),
            ],
        );

        let result = state.step(&expansions);

        // Should keep only 2 beams (the best 2 out of 4 candidates).
        assert_eq!(state.active_beams.len(), 2);
        // Best candidates: beam1+token20 (-0.3), beam0+token10 (-0.5)
        assert_eq!(
            state.active_beams[0].cumulative_logprob, -0.3,
        );
        assert_eq!(
            state.active_beams[1].cumulative_logprob, -0.5,
        );

        // Both parents were reused exactly once -- no forks or frees needed.
        assert!(result.seqs_to_free.is_empty());
        assert!(result.fork_ops.is_empty());
    }

    #[test]
    fn step_forks_when_same_parent_reused() {
        let mut state = make_state(2);

        let mut expansions = HashMap::new();
        // Beam 0: both top-2 candidates come from here
        expansions.insert(
            SequenceId(0),
            vec![
                (10u32, -0.1f32, "a".to_string(), false),
                (11, -0.2, "b".to_string(), false),
            ],
        );
        // Beam 1: worse candidates
        expansions.insert(
            SequenceId(1),
            vec![
                (20, -5.0, "c".to_string(), false),
                (21, -6.0, "d".to_string(), false),
            ],
        );

        let result = state.step(&expansions);

        assert_eq!(state.active_beams.len(), 2);
        // Both top beams come from parent 0, so one fork is needed.
        assert_eq!(result.fork_ops.len(), 1);
        assert_eq!(result.fork_ops[0].parent_seq_id, SequenceId(0));
        // Beam 1 is freed since it wasn't reused.
        assert!(result.seqs_to_free.contains(&SequenceId(1)));
    }

    #[test]
    fn step_moves_terminal_to_completed() {
        let mut state = make_state(2);

        let mut expansions = HashMap::new();
        expansions.insert(
            SequenceId(0),
            vec![
                (10u32, -0.5f32, "a".to_string(), true), // EOS
                (11, -1.0, "b".to_string(), false),
            ],
        );
        expansions.insert(
            SequenceId(1),
            vec![
                (20, -0.3, "c".to_string(), false),
                (21, -2.0, "d".to_string(), false),
            ],
        );

        state.step(&expansions);

        // The EOS candidate should be in completed.
        assert!(!state.completed.is_empty());
        assert_eq!(state.completed[0].token_ids, vec![10]);
        assert_eq!(
            state.completed[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn finished_when_no_active_beams() {
        let mut state = make_state(1);

        // All beams produce EOS.
        let mut expansions = HashMap::new();
        expansions.insert(
            SequenceId(0),
            vec![(10u32, -0.5f32, "done".to_string(), true)],
        );

        state.step(&expansions);
        assert!(state.active_beams.is_empty());
        assert!(state.is_finished());
    }

    #[test]
    fn build_output_returns_best() {
        let mut state = make_state(2);

        // Simulate two completed beams.
        state.completed.push(BeamHypothesis {
            seq_id: SequenceId(0),
            token_ids: vec![1, 2, 3],
            text: "worst".to_string(),
            cumulative_logprob: -10.0,
            logprobs: vec![],
            finished: true,
            finish_reason: Some(FinishReason::Stop),
        });
        state.completed.push(BeamHypothesis {
            seq_id: SequenceId(1),
            token_ids: vec![4, 5],
            text: "best".to_string(),
            cumulative_logprob: -1.0,
            logprobs: vec![],
            finished: true,
            finish_reason: Some(FinishReason::Stop),
        });

        let output = state.build_output("prompt", &[100], 1);
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].text, "best");
        assert!(output.finished);
    }

    #[test]
    fn build_output_n_best() {
        let mut state = make_state(3);

        for i in 0..3 {
            state.completed.push(BeamHypothesis {
                seq_id: SequenceId(i),
                token_ids: vec![i as u32],
                text: format!("hyp_{i}"),
                cumulative_logprob: -(i as f32),
                logprobs: vec![],
                finished: true,
                finish_reason: Some(FinishReason::Stop),
            });
        }

        let output = state.build_output("p", &[1], 2);
        assert_eq!(output.outputs.len(), 2);
        // Best two: hyp_0 (-0.0) and hyp_1 (-1.0)
        assert_eq!(output.outputs[0].text, "hyp_0");
        assert_eq!(output.outputs[1].text, "hyp_1");
    }

    #[test]
    fn top_k_from_logprobs_works() {
        let lps = vec![(1u32, -3.0f32), (2, -1.0), (3, -2.0), (4, -0.5)];
        let top2 = top_k_from_logprobs(&lps, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 4); // -0.5
        assert_eq!(top2[1].0, 2); // -1.0
    }

    #[test]
    fn top_k_less_than_available() {
        let lps = vec![(1u32, -1.0f32)];
        let top3 = top_k_from_logprobs(&lps, 3);
        assert_eq!(top3.len(), 1);
    }

    #[test]
    fn beam_to_output_states_ordered() {
        let mut state = make_state(2);
        state.completed.push(BeamHypothesis {
            seq_id: SequenceId(0),
            token_ids: vec![1],
            text: "bad".into(),
            cumulative_logprob: -10.0,
            logprobs: vec![],
            finished: true,
            finish_reason: Some(FinishReason::Stop),
        });
        state.completed.push(BeamHypothesis {
            seq_id: SequenceId(1),
            token_ids: vec![2],
            text: "good".into(),
            cumulative_logprob: -1.0,
            logprobs: vec![],
            finished: true,
            finish_reason: Some(FinishReason::Stop),
        });

        let states = beam_to_output_states(&state);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].text, "good");
        assert_eq!(states[1].text, "bad");
    }

    #[test]
    fn set_beam_seq_id_updates() {
        let mut state = make_state(2);
        state.set_beam_seq_id(0, SequenceId(99));
        assert_eq!(state.active_beams[0].seq_id, SequenceId(99));
    }

    #[test]
    fn max_tokens_terminates_beam() {
        let mut state = BeamSearchState::new(
            RequestId(1),
            1,
            2, // max 2 tokens
            1.0,
            false,
            &[SequenceId(0)],
        );

        // Step 1: not terminal
        let mut exp = HashMap::new();
        exp.insert(
            SequenceId(0),
            vec![(10u32, -0.5f32, "a".to_string(), false)],
        );
        state.step(&exp);
        assert_eq!(state.active_beams.len(), 1);
        assert!(state.completed.is_empty());

        // Step 2: reaches max_tokens
        let sid = state.active_beams[0].seq_id;
        let mut exp2 = HashMap::new();
        exp2.insert(
            sid,
            vec![(11u32, -0.3f32, "b".to_string(), false)],
        );
        state.step(&exp2);
        // Should be finished via max_tokens.
        assert!(state.active_beams.is_empty());
        assert_eq!(state.completed.len(), 1);
        assert_eq!(
            state.completed[0].finish_reason,
            Some(FinishReason::Length)
        );
    }
}
