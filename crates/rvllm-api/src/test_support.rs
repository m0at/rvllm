use std::sync::{Arc, Mutex};

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer as HfTokenizer;

use crate::server::InferenceEngine;

#[derive(Clone, Debug)]
pub struct RecordedGenerateCall {
    pub prompt: String,
    pub params: rvllm_core::prelude::SamplingParams,
}

pub struct RecordingEngine {
    supports_beam_search: bool,
    response: rvllm_core::prelude::RequestOutput,
    calls: Mutex<Vec<RecordedGenerateCall>>,
}

impl RecordingEngine {
    pub fn new(
        response: rvllm_core::prelude::RequestOutput,
        supports_beam_search: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            supports_beam_search,
            response,
            calls: Mutex::new(Vec::new()),
        })
    }

    pub fn calls(&self) -> Vec<RecordedGenerateCall> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl InferenceEngine for RecordingEngine {
    async fn generate(
        &self,
        prompt: String,
        params: rvllm_core::prelude::SamplingParams,
    ) -> rvllm_core::prelude::Result<(
        rvllm_core::prelude::RequestId,
        tokio_stream::wrappers::ReceiverStream<rvllm_core::prelude::RequestOutput>,
    )> {
        self.calls
            .lock()
            .unwrap()
            .push(RecordedGenerateCall { prompt, params });

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(self.response.clone()).await.unwrap();
        Ok((
            self.response.request_id,
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    fn supports_beam_search(&self) -> bool {
        self.supports_beam_search
    }
}

pub fn make_finished_output(texts: &[&str], finished: bool) -> rvllm_core::prelude::RequestOutput {
    rvllm_core::prelude::RequestOutput {
        request_id: rvllm_core::prelude::RequestId(1),
        prompt: "prompt".into(),
        prompt_token_ids: vec![1, 2, 3],
        prompt_logprobs: None,
        outputs: texts
            .iter()
            .enumerate()
            .map(|(index, text)| rvllm_core::prelude::CompletionOutput {
                index,
                text: (*text).to_string(),
                token_ids: vec![index as u32 + 10],
                cumulative_logprob: -0.1 - index as f32,
                logprobs: None,
                finish_reason: Some(rvllm_core::prelude::FinishReason::Stop),
            })
            .collect(),
        finished,
    }
}

pub fn make_test_tokenizer() -> rvllm_tokenizer::Tokenizer {
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert(" ".to_string(), 2);
    vocab.insert("!".to_string(), 3);
    vocab.insert("[UNK]".to_string(), 4);

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();

    let mut hf = HfTokenizer::new(bpe);
    hf.with_pre_tokenizer(Some(Whitespace {}));

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    hf.save(&path, false).unwrap();
    rvllm_tokenizer::Tokenizer::from_file(&path).unwrap()
}
