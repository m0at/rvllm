//! Output processing: detokenization, logprob formatting, RequestOutput construction.

use rvllm_core::prelude::{
    CompletionOutput, FinishReason, LogProb, RequestId, RequestOutput, SamplingParams, TokenId,
};

use crate::stop_checker::StopChecker;

/// Per-sequence accumulator tracking generated tokens and decoded text.
#[derive(Debug, Clone)]
pub struct SequenceOutputState {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub cumulative_logprob: LogProb,
    pub logprobs: Vec<Vec<(TokenId, LogProb)>>,
    pub finish_reason: Option<FinishReason>,
}

impl SequenceOutputState {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            token_ids: Vec::new(),
            cumulative_logprob: 0.0,
            logprobs: Vec::new(),
            finish_reason: None,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

impl Default for SequenceOutputState {
    fn default() -> Self {
        Self::new()
    }
}

/// Processes raw sampler outputs into structured `RequestOutput`.
pub struct OutputProcessor;

impl OutputProcessor {
    /// Append a newly sampled token to the output state, decode incrementally,
    /// and check stop conditions.
    pub fn process_token(
        state: &mut SequenceOutputState,
        token_id: TokenId,
        logprob: LogProb,
        top_logprobs: Option<Vec<(TokenId, LogProb)>>,
        decoded_text: &str,
        params: &SamplingParams,
        eos_token_id: Option<TokenId>,
    ) {
        state.token_ids.push(token_id);
        state.cumulative_logprob += logprob;
        state.text.push_str(decoded_text);

        if let Some(top) = top_logprobs {
            state.logprobs.push(top);
        }

        // Check stop conditions
        if let Some(reason) =
            StopChecker::check_stop(&state.text, &state.token_ids, params, eos_token_id)
        {
            state.finish_reason = Some(reason);

            // Truncate at stop string if applicable
            if reason == FinishReason::Stop && !params.stop_strings.is_empty() {
                let (truncated, _) =
                    StopChecker::truncate_at_stop(&state.text, &params.stop_strings);
                state.text = truncated;
            }
        }
    }

    /// Build a `CompletionOutput` from accumulated state.
    pub fn build_completion(state: &SequenceOutputState, index: usize) -> CompletionOutput {
        CompletionOutput {
            index,
            text: state.text.clone(),
            token_ids: state.token_ids.clone(),
            cumulative_logprob: state.cumulative_logprob,
            logprobs: if state.logprobs.is_empty() {
                None
            } else {
                Some(state.logprobs.clone())
            },
            finish_reason: state.finish_reason,
        }
    }

    /// Build a full `RequestOutput` from the request metadata and sequence states.
    pub fn build_request_output(
        request_id: RequestId,
        prompt: &str,
        prompt_token_ids: &[TokenId],
        seq_states: &[SequenceOutputState],
    ) -> RequestOutput {
        Self::build_request_output_with_prompt_logprobs(
            request_id, prompt, prompt_token_ids, seq_states, None,
        )
    }

    /// Build a full `RequestOutput` with optional prompt logprobs (for echo mode).
    pub fn build_request_output_with_prompt_logprobs(
        request_id: RequestId,
        prompt: &str,
        prompt_token_ids: &[TokenId],
        seq_states: &[SequenceOutputState],
        prompt_logprobs: Option<Vec<Vec<(TokenId, LogProb)>>>,
    ) -> RequestOutput {
        let finished = seq_states.iter().all(|s| s.is_finished());
        let outputs: Vec<CompletionOutput> = seq_states
            .iter()
            .enumerate()
            .map(|(i, s)| Self::build_completion(s, i))
            .collect();

        RequestOutput {
            request_id,
            prompt: prompt.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            prompt_logprobs,
            outputs,
            finished,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn sequence_output_state_default() {
        let s = SequenceOutputState::new();
        assert!(s.text.is_empty());
        assert!(s.token_ids.is_empty());
        assert!(!s.is_finished());
        assert_eq!(s.cumulative_logprob, 0.0);
    }

    #[test]
    fn process_token_accumulates() {
        let mut state = SequenceOutputState::new();
        let params = default_params();

        OutputProcessor::process_token(&mut state, 10, -0.5, None, "hello", &params, None);
        assert_eq!(state.token_ids, vec![10]);
        assert_eq!(state.text, "hello");
        assert!((state.cumulative_logprob - (-0.5)).abs() < 1e-6);
        assert!(!state.is_finished());

        OutputProcessor::process_token(&mut state, 20, -1.0, None, " world", &params, None);
        assert_eq!(state.token_ids, vec![10, 20]);
        assert_eq!(state.text, "hello world");
        assert!((state.cumulative_logprob - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn process_token_stops_on_eos() {
        let mut state = SequenceOutputState::new();
        let params = default_params();

        OutputProcessor::process_token(&mut state, 99, -0.1, None, "done", &params, Some(99));
        assert!(state.is_finished());
        assert_eq!(state.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn process_token_stops_on_max_tokens() {
        let mut state = SequenceOutputState::new();
        let mut params = default_params();
        params.max_tokens = 2;

        OutputProcessor::process_token(&mut state, 1, -0.1, None, "a", &params, None);
        assert!(!state.is_finished());
        OutputProcessor::process_token(&mut state, 2, -0.2, None, "b", &params, None);
        assert!(state.is_finished());
        assert_eq!(state.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn process_token_stops_on_stop_string() {
        let mut state = SequenceOutputState::new();
        let mut params = default_params();
        params.stop_strings = vec!["<end>".to_string()];

        OutputProcessor::process_token(&mut state, 1, -0.1, None, "text<end>more", &params, None);
        assert!(state.is_finished());
        assert_eq!(state.finish_reason, Some(FinishReason::Stop));
        // Text should be truncated at the stop string
        assert_eq!(state.text, "text");
    }

    #[test]
    fn process_token_with_logprobs() {
        let mut state = SequenceOutputState::new();
        let params = default_params();
        let top = vec![(10, -0.5_f32), (20, -1.0)];

        OutputProcessor::process_token(&mut state, 10, -0.5, Some(top.clone()), "a", &params, None);
        assert_eq!(state.logprobs.len(), 1);
        assert_eq!(state.logprobs[0], top);
    }

    #[test]
    fn build_completion_output() {
        let mut state = SequenceOutputState::new();
        state.text = "hello".into();
        state.token_ids = vec![1, 2];
        state.cumulative_logprob = -1.5;
        state.finish_reason = Some(FinishReason::Stop);

        let co = OutputProcessor::build_completion(&state, 0);
        assert_eq!(co.index, 0);
        assert_eq!(co.text, "hello");
        assert_eq!(co.token_ids, vec![1, 2]);
        assert_eq!(co.finish_reason, Some(FinishReason::Stop));
        assert!(co.logprobs.is_none());
    }

    #[test]
    fn build_completion_with_logprobs() {
        let mut state = SequenceOutputState::new();
        state.text = "a".into();
        state.token_ids = vec![1];
        state.logprobs = vec![vec![(1, -0.5)]];

        let co = OutputProcessor::build_completion(&state, 0);
        assert!(co.logprobs.is_some());
        assert_eq!(co.logprobs.unwrap().len(), 1);
    }

    #[test]
    fn build_request_output_not_finished() {
        let s1 = SequenceOutputState::new();
        let ro = OutputProcessor::build_request_output(
            RequestId(1),
            "prompt",
            &[10, 20],
            &[s1],
        );
        assert_eq!(ro.request_id, RequestId(1));
        assert_eq!(ro.prompt, "prompt");
        assert_eq!(ro.prompt_token_ids, vec![10, 20]);
        assert!(!ro.finished);
        assert_eq!(ro.outputs.len(), 1);
    }

    #[test]
    fn build_request_output_finished() {
        let mut s1 = SequenceOutputState::new();
        s1.finish_reason = Some(FinishReason::Stop);
        s1.text = "done".into();

        let ro = OutputProcessor::build_request_output(
            RequestId(2),
            "test",
            &[5],
            &[s1],
        );
        assert!(ro.finished);
    }

    #[test]
    fn build_request_output_multiple_seqs() {
        let mut s1 = SequenceOutputState::new();
        s1.finish_reason = Some(FinishReason::Stop);
        let s2 = SequenceOutputState::new(); // not finished

        let ro = OutputProcessor::build_request_output(
            RequestId(3),
            "multi",
            &[1],
            &[s1, s2],
        );
        assert!(!ro.finished); // one is not finished
        assert_eq!(ro.outputs.len(), 2);
        assert_eq!(ro.outputs[0].index, 0);
        assert_eq!(ro.outputs[1].index, 1);
    }
}
