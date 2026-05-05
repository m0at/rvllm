use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelListResponse {
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamChoice {
    delta: Delta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Delta {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamChunk {
    #[allow(dead_code)]
    id: Option<String>,
    choices: Vec<StreamChoice>,
}

/// Shape rvllm-serve uses when an SSE stream errors mid-flight.
/// The handler emits one `data: {"error": {...}}` line and then
/// `data: [DONE]`. Without parsing this we'd silently treat the
/// failed generation as a successful empty completion.
#[derive(Debug, Clone, Deserialize)]
struct StreamErrorEnvelope {
    error: StreamErrorBody,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamErrorBody {
    message: String,
    #[serde(default)]
    #[allow(dead_code)]
    code: Option<String>,
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    kind: Option<String>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Done { tokens: u32, elapsed_secs: f64 },
    Error(String),
    TokPerSec(f64),
}

pub fn fetch_models(endpoint: String, tx: mpsc::Sender<Vec<String>>) {
    let url = models_url(&endpoint);
    tokio::spawn(async move {
        let result = async {
            let resp = reqwest::get(&url).await?;
            let body: ModelListResponse = resp.json().await?;
            Ok::<_, reqwest::Error>(body.data.into_iter().map(|m| m.id).collect::<Vec<_>>())
        }
        .await;
        match result {
            Ok(models) => { let _ = tx.send(models); }
            Err(_) => { let _ = tx.send(vec![]); }
        }
    });
}

pub fn stream_chat(
    endpoint: String,
    request: ChatRequest,
    tx: mpsc::Sender<StreamEvent>,
) {
    tokio::spawn(async move {
        let result = stream_inner(endpoint, request, tx.clone()).await;
        if let Err(e) = result {
            let _ = tx.send(StreamEvent::Error(e.to_string()));
        }
    });
}

async fn stream_inner(
    endpoint: String,
    request: ChatRequest,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let resp = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body).into());
    }

    let mut first_token_time: Option<Instant> = None;
    let mut token_count: u32 = 0;
    let mut last_tps_update = Instant::now();
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Process complete SSE lines from buffer
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if data == "[DONE]" {
                    let elapsed = first_token_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    let _ = tx.send(StreamEvent::Done {
                        tokens: token_count,
                        elapsed_secs: elapsed,
                    });
                    return Ok(());
                }

                // Server-side SSE error envelope. Try this BEFORE
                // StreamChunk so a CUDA / tokenizer / timeout failure
                // surfaces as Error in the UI instead of vanishing
                // (StreamChunk would silently fail to deserialize and
                // the loop would emit nothing until [DONE] flagged
                // success).
                if let Ok(env) = serde_json::from_str::<StreamErrorEnvelope>(data) {
                    let _ = tx.send(StreamEvent::Error(env.error.message));
                    return Ok(());
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    for choice in &chunk.choices {
                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                if first_token_time.is_none() {
                                    first_token_time = Some(Instant::now());
                                }
                                token_count += 1;
                                let _ = tx.send(StreamEvent::Token(content.clone()));

                                // Update tok/s at most every 100ms, measured from first token
                                let now = Instant::now();
                                if now.duration_since(last_tps_update).as_millis() >= 100 {
                                    if let Some(ft) = first_token_time {
                                        let elapsed = ft.elapsed().as_secs_f64();
                                        if elapsed > 0.0 {
                                            let tps = token_count as f64 / elapsed;
                                            let _ = tx.send(StreamEvent::TokPerSec(tps));
                                        }
                                    }
                                    last_tps_update = now;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Stream ended without [DONE]
    let elapsed = first_token_time
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let _ = tx.send(StreamEvent::Done {
        tokens: token_count,
        elapsed_secs: elapsed,
    });
    Ok(())
}

fn models_url(chat_endpoint: &str) -> String {
    // Given .../v1/chat/completions, derive .../v1/models
    if let Some(pos) = chat_endpoint.find("/v1/") {
        format!("{}/v1/models", &chat_endpoint[..pos])
    } else {
        // Fallback: just replace the path
        chat_endpoint
            .rsplit_once('/')
            .map(|(base, _)| format!("{}/models", base))
            .unwrap_or_else(|| format!("{}/models", chat_endpoint))
    }
}
