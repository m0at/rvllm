//! Best-of-N sampling: run N independent completions, return the one with
//! the highest cumulative log-probability.
//!
//! The engine already spawns `best_of` sequences per request in
//! `insert_request` / `add_request`. This module provides the selection
//! logic that picks the winning completion once all N sequences finish.

use rvllm_core::prelude::{CompletionOutput, RequestOutput};

use crate::output::SequenceOutputState;

/// Select the single best completion from `best_of` independent samples.
///
/// Returns a new `RequestOutput` whose `outputs` vec contains only the
/// winning completion (re-indexed to 0). If no sequence is finished the
/// input is returned unchanged.
pub fn select_best_of_n(mut output: RequestOutput) -> RequestOutput {
    // Nothing to select if there is only one output or none are finished.
    if output.outputs.len() <= 1 {
        return output;
    }

    let all_finished = output.outputs.iter().all(|o| o.finish_reason.is_some());
    if !all_finished {
        return output;
    }

    // Pick the completion with the highest cumulative logprob.
    let best_idx = output
        .outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.cumulative_logprob
                .partial_cmp(&b.cumulative_logprob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut best = output.outputs.swap_remove(best_idx);
    best.index = 0;
    output.outputs = vec![best];
    output
}

/// Select the best completion from raw `SequenceOutputState` slices.
///
/// Returns the index of the state with the highest cumulative logprob
/// among those that are finished. Returns `None` if no state is finished.
pub fn best_of_n_index(states: &[SequenceOutputState]) -> Option<usize> {
    states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finished())
        .max_by(|(_, a), (_, b)| {
            a.cumulative_logprob
                .partial_cmp(&b.cumulative_logprob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Build a `RequestOutput` containing only the best-of-N winner from
/// per-sequence states.
///
/// If not all states are finished, returns `None` (caller should keep
/// running the request).
pub fn build_best_of_n_output(
    output: RequestOutput,
    states: &[SequenceOutputState],
) -> RequestOutput {
    if states.len() <= 1 || !states.iter().all(|s| s.is_finished()) {
        return output;
    }

    let best_idx = best_of_n_index(states).unwrap_or(0);
    let best = &states[best_idx];

    let completion = CompletionOutput {
        index: 0,
        text: best.text.clone(),
        token_ids: best.token_ids.clone(),
        cumulative_logprob: best.cumulative_logprob,
        logprobs: if best.logprobs.is_empty() {
            None
        } else {
            Some(best.logprobs.clone())
        },
        finish_reason: best.finish_reason,
    };

    RequestOutput {
        outputs: vec![completion],
        ..output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::prelude::{FinishReason, RequestId};

    fn make_output(logprobs: &[f32]) -> RequestOutput {
        let outputs: Vec<CompletionOutput> = logprobs
            .iter()
            .enumerate()
            .map(|(i, &lp)| CompletionOutput {
                index: i,
                text: format!("text_{i}"),
                token_ids: vec![i as u32],
                cumulative_logprob: lp,
                logprobs: None,
                finish_reason: Some(FinishReason::Stop),
            })
            .collect();

        RequestOutput {
            request_id: RequestId(1),
            prompt: "test".into(),
            prompt_token_ids: vec![1],
            prompt_logprobs: None,
            outputs,
            finished: true,
        }
    }

    #[test]
    fn selects_highest_logprob() {
        let output = make_output(&[-3.0, -1.0, -2.0]);
        let result = select_best_of_n(output);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].text, "text_1");
        assert_eq!(result.outputs[0].index, 0);
    }

    #[test]
    fn single_output_unchanged() {
        let output = make_output(&[-1.0]);
        let result = select_best_of_n(output);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].text, "text_0");
    }

    #[test]
    fn not_all_finished_unchanged() {
        let mut output = make_output(&[-1.0, -2.0]);
        output.outputs[1].finish_reason = None;
        let result = select_best_of_n(output);
        assert_eq!(result.outputs.len(), 2);
    }

    #[test]
    fn best_of_n_index_picks_best() {
        let states = vec![
            SequenceOutputState {
                text: "a".into(),
                token_ids: vec![1],
                cumulative_logprob: -5.0,
                logprobs: vec![],
                finish_reason: Some(FinishReason::Stop),
            },
            SequenceOutputState {
                text: "b".into(),
                token_ids: vec![2],
                cumulative_logprob: -1.0,
                logprobs: vec![],
                finish_reason: Some(FinishReason::Length),
            },
            SequenceOutputState {
                text: "c".into(),
                token_ids: vec![3],
                cumulative_logprob: -3.0,
                logprobs: vec![],
                finish_reason: Some(FinishReason::Stop),
            },
        ];
        assert_eq!(best_of_n_index(&states), Some(1));
    }

    #[test]
    fn best_of_n_index_none_when_unfinished() {
        let states = vec![SequenceOutputState {
            text: String::new(),
            token_ids: vec![],
            cumulative_logprob: 0.0,
            logprobs: vec![],
            finish_reason: None,
        }];
        assert_eq!(best_of_n_index(&states), None);
    }

    #[test]
    fn build_best_of_n_output_selects_winner() {
        let states = vec![
            SequenceOutputState {
                text: "loser".into(),
                token_ids: vec![1],
                cumulative_logprob: -10.0,
                logprobs: vec![],
                finish_reason: Some(FinishReason::Stop),
            },
            SequenceOutputState {
                text: "winner".into(),
                token_ids: vec![2],
                cumulative_logprob: -0.5,
                logprobs: vec![],
                finish_reason: Some(FinishReason::Stop),
            },
        ];
        let output = make_output(&[-10.0, -0.5]);
        let result = build_best_of_n_output(output, &states);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].text, "winner");
    }
}
