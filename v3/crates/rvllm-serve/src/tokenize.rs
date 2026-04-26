//! Tokenizer + chat-template wrapper.
//!
//! Loads:
//!   * `tokenizer.json` via the `tokenizers` crate — BPE model,
//!     added-tokens map, pre-tokenizer config.
//!   * `tokenizer_config.json` — harvest `bos_token` / `eos_token` ids
//!     + `add_bos_token` default.
//!   * `chat_template.jinja` (preferred) or `chat_template` key inside
//!     `tokenizer_config.json` — rendered per request via `minijinja`.
//!
//! Decode is split between:
//!   * [`TokenizerHandle::decode`] — batch-decode a complete token
//!     list; used by the non-streaming completion path.
//!   * [`TokenizerHandle::stream_decoder`] — returns a
//!     [`StreamDecoder`] that buffers partial UTF-8 between tokens
//!     (BPE can emit a byte-level token mid-codepoint). Phase 5 SSE
//!     path uses this so clients never see replacement chars.

use std::path::Path;
use std::sync::Arc;

use minijinja::{context, Environment};
use serde::Serialize;

use crate::error::ApiError;
use crate::openai::chat::ChatMessage;
use crate::openai::types::Role;

/// Shared tokenizer handle. Cloneable (`Arc` inside).
#[derive(Clone)]
pub struct TokenizerHandle {
    inner: Arc<TokenizerInner>,
}

struct TokenizerInner {
    tokenizer: tokenizers::Tokenizer,
    /// Rendered into the first turn on every request.
    chat_template: String,
    /// Literal `<bos>` style token the template references via
    /// `bos_token`. Rendered as a STRING by the template; the whole
    /// rendered string is then re-tokenized so the BOS id pops out
    /// naturally. Stored for diagnostics only.
    bos_token_str: Option<String>,
    /// Resolved `<bos>` id. `None` if the tokenizer has no BOS.
    /// Exposed so the raw-prompt encode path can prepend it (Gemma 4
    /// ships an empty `TemplateProcessing` so `add_special_tokens`
    /// does not add BOS by itself).
    bos_token_id: Option<u32>,
    /// Tokens that should end generation. Populated from
    /// `tokenizer_config.json` (`eos_token`) plus any `<end_of_turn>`
    /// / `<turn|>` style markers we resolve against the vocab.
    eos_token_ids: Vec<u32>,
}

impl TokenizerHandle {
    /// Load everything from a HF model dir.
    pub fn load(model_dir: &Path) -> Result<Self, ApiError> {
        // 1. tokenizer.json
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| {
            ApiError::Internal(format!(
                "tokenizer.json load from {}: {e}",
                tok_path.display()
            ))
        })?;

        // 2. tokenizer_config.json — harvest special token strings.
        let cfg_path = model_dir.join("tokenizer_config.json");
        let cfg: serde_json::Value = std::fs::read_to_string(&cfg_path)
            .map_err(|e| ApiError::Internal(format!("read {}: {e}", cfg_path.display())))
            .and_then(|body| {
                serde_json::from_str(&body).map_err(|e| {
                    ApiError::Internal(format!("parse {}: {e}", cfg_path.display()))
                })
            })?;

        let bos_token_str = extract_token_str(&cfg, "bos_token");
        let eos_token_str = extract_token_str(&cfg, "eos_token");
        let bos_token_id = bos_token_str
            .as_deref()
            .and_then(|s| tokenizer.token_to_id(s));

        // 3. Resolve EOS (+ companions) against the vocab.
        let mut eos_token_ids: Vec<u32> = Vec::new();
        if let Some(eos) = &eos_token_str {
            if let Some(id) = tokenizer.token_to_id(eos) {
                eos_token_ids.push(id);
            }
        }
        // Gemma 4 marks turn boundaries with `<turn|>` / `<end_of_turn>`.
        // Either may or may not be in the vocab; add whichever resolves.
        for extra in ["<end_of_turn>", "<turn|>"] {
            if let Some(id) = tokenizer.token_to_id(extra) {
                if !eos_token_ids.contains(&id) {
                    eos_token_ids.push(id);
                }
            }
        }
        if eos_token_ids.is_empty() {
            return Err(ApiError::Internal(
                "tokenizer_config.json has no resolvable eos_token — refusing to load"
                    .into(),
            ));
        }

        // 4. chat_template.jinja (preferred) or inline in config.
        let tpl_path = model_dir.join("chat_template.jinja");
        let chat_template = if tpl_path.is_file() {
            std::fs::read_to_string(&tpl_path).map_err(|e| {
                ApiError::Internal(format!("read {}: {e}", tpl_path.display()))
            })?
        } else {
            match cfg.get("chat_template") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(arr)) => {
                    // HF supports a list of {name, template} entries.
                    // Pick the "default" if named, else the first.
                    let pick = arr
                        .iter()
                        .find(|e| {
                            e.get("name").and_then(|n| n.as_str())
                                == Some("default")
                        })
                        .or_else(|| arr.first())
                        .and_then(|e| {
                            e.get("template").and_then(|t| t.as_str())
                        })
                        .map(String::from);
                    pick.ok_or_else(|| {
                        ApiError::Internal(
                            "chat_template array lacks a string `template` field"
                                .into(),
                        )
                    })?
                }
                _ => {
                    return Err(ApiError::Internal(format!(
                        "no chat_template.jinja at {} and no `chat_template` in {}",
                        tpl_path.display(),
                        cfg_path.display(),
                    )))
                }
            }
        };

        Ok(Self {
            inner: Arc::new(TokenizerInner {
                tokenizer,
                chat_template,
                bos_token_str,
                bos_token_id,
                eos_token_ids,
            }),
        })
    }

    /// Encode raw text. Prepends `<bos>` if the tokenizer's
    /// `post_processor` doesn't add it (Gemma 4 ships an empty
    /// TemplateProcessing, so `add_special_tokens=true` is a no-op
    /// for this model). Without BOS, Gemma 4 greedy-decodes a
    /// degenerate all-`<pad>` sequence — confirmed on GB10 by
    /// logging the raw generated ids. The [`ChatMessage`] path
    /// doesn't use this — its template emits `<bos>` explicitly at
    /// the start of the rendered string.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ApiError> {
        let enc = self
            .inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))?;
        let mut ids = enc.get_ids().to_vec();
        if let Some(bos_id) = self.inner.bos_token_id {
            if ids.first().copied() != Some(bos_id) {
                ids.insert(0, bos_id);
            }
        }
        Ok(ids)
    }

    /// Token id of `<bos>` (if the tokenizer knows one).
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id
    }

    /// Render the chat template around `messages` and return prompt
    /// token IDs (BOS already included via the template).
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<Vec<u32>, ApiError> {
        let mut env = Environment::new();
        // HF chat templates use Python-style `dict.get(key)` /
        // `.items()` which jinja2 supports natively but minijinja
        // does not. Register the pycompat shim so method calls on
        // maps / strings resolve to the Python semantics.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.inner.chat_template)
            .map_err(|e| ApiError::Tokenize(format!("chat template parse: {e}")))?;
        let tpl = env
            .get_template("chat")
            .map_err(|e| ApiError::Internal(format!("get template: {e}")))?;

        // minijinja consumes serde-serializable values. Project each
        // typed `ChatMessage` to an HF-shaped dict that carries every
        // optional field the Gemma 4 template actually reads
        // (content, tool_calls, tool_call_id, name). Missing keys are
        // skipped so `message.get('tool_calls')` returns falsy exactly
        // like the HF reference path.
        // Pre-parse `tool_calls[i].function.arguments` from its OpenAI
        // string form into a JSON object where possible. Gemma 4's
        // chat template has two branches:
        //   * `arguments is mapping` → iterates the dict, renders
        //     `{key:value,key:value}` without extra braces.
        //   * `arguments is string` → inlines the string verbatim
        //     between the `NAME{` prefix and the `}` suffix.
        // With OpenAI-spec strings like `{"city":"Bern"}` the string
        // branch would emit `NAME{{"city":"Bern"}}` — double braces,
        // off-distribution for the model. It then ignores the tool
        // result on replay and reissues the same call every turn,
        // which zeroclaw's circuit breaker rightly aborts after 5
        // identical invocations. Converting the string to a mapping
        // routes through the first branch, producing the training
        // shape `NAME{city:"Bern"}`.
        let normalized_tool_calls: Vec<Option<serde_json::Value>> = messages
            .iter()
            .map(|m| m.tool_calls.as_ref().map(normalize_tool_calls))
            .collect();
        let serde_msgs: Vec<TemplateMsg> = messages
            .iter()
            .zip(normalized_tool_calls.iter())
            .map(|(m, tc)| TemplateMsg {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                },
                content: m.content.as_deref().unwrap_or(""),
                tool_calls: tc.as_ref(),
                tool_call_id: m.tool_call_id.as_deref(),
                name: m.name.as_deref(),
            })
            .collect();

        let rendered = tpl
            .render(context! {
                messages => serde_msgs,
                bos_token => self.inner.bos_token_str.as_deref().unwrap_or(""),
                add_generation_prompt => true,
                // Threaded through so Gemma 4's chat template emits its native
                // `<|tool_call>call:NAME{...}<tool_call|>` tool-calling block.
                // The existing `tool_parser` extracts that back on response.
                // Without this the model improvises Python-style `brain(...)`
                // calls as plain text — zeroclaw saw exactly that on prefill.
                tools => tools,
                enable_thinking => false,
            })
            .map_err(|e| ApiError::Tokenize(format!("chat template render: {e}")))?;

        // Re-tokenize the rendered string. Special tokens inside the
        // string (e.g. `<bos>`, `<|turn>`) are recognised by the
        // tokenizer's added-tokens map when `add_special_tokens=true`.
        let enc = self
            .inner
            .tokenizer
            .encode(rendered, true)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a complete token run into text. Strips special tokens
    /// so user-visible output is clean.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ApiError> {
        self.inner
            .tokenizer
            .decode(ids, /*skip_special_tokens*/ true)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))
    }

    /// Decode while preserving special tokens (`<|tool_call>`,
    /// `<|channel>`, …). The `tool_parser` needs these markers to
    /// reliably extract calls + strip thought channels from content;
    /// with `skip_special_tokens=true` they vanish and the raw prose
    /// inside a `<|channel>thought` block leaks straight into the
    /// user-visible reply.
    pub fn decode_raw(&self, ids: &[u32]) -> Result<String, ApiError> {
        self.inner
            .tokenizer
            .decode(ids, /*skip_special_tokens*/ false)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))
    }

    /// Build a new streaming decoder. Each call to
    /// [`StreamDecoder::step`] returns the UTF-8 text emitted by the
    /// next token; incomplete codepoints are buffered until a
    /// subsequent token completes them, so SSE clients never receive
    /// `U+FFFD` replacement chars.
    pub fn stream_decoder(&self) -> StreamDecoder {
        StreamDecoder {
            tokenizer: self.inner.clone(),
            ids: Vec::new(),
            emitted_chars: 0,
            keep_special: false,
        }
    }

    /// Streaming decoder that PRESERVES special tokens (`<|tool_call>`,
    /// `<|channel>`, …). Used by the chat-completions SSE path so the
    /// caller can run `tool_parser::strip_tool_markup` on the raw view
    /// before deciding what to emit as user-visible `delta.content`.
    /// Without this, the streaming path leaked thought-block prose and
    /// orphan tool-call markers — non-streaming preserved them via
    /// `decode_raw` then stripped structurally; streaming was dropping
    /// the very markers it needed to recognise the spans.
    pub fn stream_decoder_raw(&self) -> StreamDecoder {
        StreamDecoder {
            tokenizer: self.inner.clone(),
            ids: Vec::new(),
            emitted_chars: 0,
            keep_special: true,
        }
    }

    /// Token ids that terminate generation.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.inner.eos_token_ids
    }
}

/// Walk each call's `function.arguments` and, when it's a JSON-object
/// literal in string form, replace it with the parsed mapping. Non-JSON
/// or non-object strings are left untouched (the template's string
/// branch is still the right choice for those — the model's own
/// training set occasionally emits bare `{key:value}` rather than full
/// JSON). Clones only the touched path; the rest of the value is reused.
fn normalize_tool_calls(calls: &serde_json::Value) -> serde_json::Value {
    let mut cloned = calls.clone();
    if let Some(arr) = cloned.as_array_mut() {
        for call in arr.iter_mut() {
            let Some(function) = call.get_mut("function") else { continue };
            let Some(args) = function.get_mut("arguments") else { continue };
            if let Some(s) = args.as_str() {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                    if parsed.is_object() {
                        *args = parsed;
                    }
                }
            }
        }
    }
    cloned
}

/// Shape handed to the jinja template per message. All tool-specific
/// fields are `skip_serializing_if = "Option::is_none"` so the jinja
/// `message.get('tool_calls')` / `message.get('tool_call_id')` probes
/// return falsy when absent, matching HF's reference rendering.
#[derive(Serialize)]
struct TemplateMsg<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

fn extract_token_str(cfg: &serde_json::Value, key: &str) -> Option<String> {
    match cfg.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        // HF sometimes stores `{"content": "<eos>", ...}`.
        serde_json::Value::Object(o) => o
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from),
        _ => None,
    }
}

// ─── Streaming decoder ───────────────────────────────────────────────

/// Stateful per-request decoder. Tracks the accumulated token ids and
/// emits **only the newly-visible UTF-8 delta** on each `step` call,
/// handling:
///   * tokens that emit multi-byte UTF-8 characters in multiple pieces
///     (BPE byte-level fallback), and
///   * the tokenizer's whitespace-merging behaviour, where the string
///     for `[a, b]` may differ from `"a" + "b"` (stripped leading space
///     on b).
pub struct StreamDecoder {
    tokenizer: Arc<TokenizerInner>,
    ids: Vec<u32>,
    /// Number of char-boundary-aligned bytes already emitted to the
    /// client. Tracked in bytes (not chars) so we can slice the
    /// decoded string without re-scanning.
    emitted_chars: usize,
    /// When true, `decode(..., skip_special_tokens=false)` so callers
    /// see `<|tool_call>` / `<|channel>` markers verbatim. Set via
    /// `TokenizerHandle::stream_decoder_raw()`.
    keep_special: bool,
}

impl StreamDecoder {
    /// Feed one newly-generated token. Returns the incremental UTF-8
    /// text that the client should append (may be empty if the token
    /// completed a partial codepoint started by the previous token —
    /// we hold text until it lands on a char boundary).
    pub fn step(&mut self, id: u32) -> Result<String, ApiError> {
        self.ids.push(id);
        // Decode the whole accumulated run each step. The HF rust
        // tokenizer is fast enough at kilotoken scales that this
        // beats managing a BPE-aware incremental decoder by hand.
        let full = self
            .tokenizer
            .tokenizer
            .decode(&self.ids, /*skip_special_tokens*/ !self.keep_special)
            .map_err(|e| ApiError::Tokenize(format!("{e}")))?;

        // Emit bytes past the last high-water mark, but only up to
        // the most recent valid UTF-8 boundary — BPE can split a
        // 4-byte codepoint across tokens.
        if full.len() <= self.emitted_chars {
            return Ok(String::new());
        }
        let new_slice = &full[self.emitted_chars..];
        // `str` slicing is already boundary-safe because `full` is a
        // `String` (validated UTF-8) — the slice from a valid start
        // index to the end is always valid.
        self.emitted_chars = full.len();
        Ok(new_slice.to_string())
    }

    /// Flush any buffered tail after the last `step`. Currently a
    /// no-op (the boundary check above never withholds valid UTF-8),
    /// kept as a hook for future decoders that batch text until
    /// newline / sentence boundary.
    pub fn finish(self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests that don't need a real model dir. The live tokenizer-
    // backed tests are gated on `RVLLM_TEST_MODEL_DIR` env var so CI
    // can opt in without bundling a tokenizer.

    #[test]
    fn extract_token_handles_string_and_object() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"bos_token":"<bos>","eos_token":{"content":"<eos>"}}"#,
        )
        .expect("parse");
        assert_eq!(extract_token_str(&v, "bos_token").as_deref(), Some("<bos>"));
        assert_eq!(extract_token_str(&v, "eos_token").as_deref(), Some("<eos>"));
        assert_eq!(extract_token_str(&v, "missing"), None);
    }

    #[test]
    fn load_roundtrips_against_live_tokenizer() {
        let Some(dir) = std::env::var_os("RVLLM_TEST_MODEL_DIR") else {
            eprintln!("skip: RVLLM_TEST_MODEL_DIR not set");
            return;
        };
        let t = TokenizerHandle::load(std::path::Path::new(&dir)).expect("load");
        assert!(!t.eos_token_ids().is_empty());
        let ids = t.encode("hello world").expect("encode");
        let back = t.decode(&ids).expect("decode");
        assert!(back.contains("hello"));
    }

    #[test]
    fn render_chat_against_live_tokenizer() {
        let Some(dir) = std::env::var_os("RVLLM_TEST_MODEL_DIR") else {
            eprintln!("skip: RVLLM_TEST_MODEL_DIR not set");
            return;
        };
        let t = TokenizerHandle::load(std::path::Path::new(&dir)).expect("load");
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: Some("Say hi.".into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let ids = t.render_chat(&msgs, None).expect("render");
        assert!(!ids.is_empty(), "empty prompt — template render produced nothing");
        // Should start with BOS — Gemma 4 bos id is 2.
        eprintln!("first 5 ids: {:?}", &ids[..ids.len().min(5)]);
    }

    /// `normalize_tool_calls` routes the Gemma 4 chat template's
    /// `function['arguments'] is mapping` branch, which emits training-
    /// shape `NAME{k:"v"}`. If we left the OpenAI-canonical string
    /// through, the string branch would wrap `{"city":"Bern"}` in the
    /// template's extra `{…}` and produce `NAME{{"city":"Bern"}}` —
    /// which is what sent the model into the weather-loop in prod.
    #[test]
    fn normalize_tool_calls_converts_arguments_string_to_object() {
        let calls = serde_json::json!([{
            "id": "c1",
            "type": "function",
            "function": { "name": "weather", "arguments": "{\"city\":\"Bern\"}" }
        }]);
        let normalized = normalize_tool_calls(&calls);
        let args = &normalized[0]["function"]["arguments"];
        assert!(args.is_object(), "expected mapping, got {args}");
        assert_eq!(args["city"], "Bern");
    }

    /// Non-JSON / non-object argument strings must be left untouched —
    /// the template's string branch is the right choice for them.
    #[test]
    fn normalize_tool_calls_leaves_non_object_arguments_alone() {
        let calls = serde_json::json!([{
            "id": "c1",
            "type": "function",
            "function": { "name": "ping", "arguments": "not-json" }
        }]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized[0]["function"]["arguments"], "not-json");
    }

    /// Already-a-mapping arguments stay as is (zeroclaw's native-provider
    /// path sends them this way).
    #[test]
    fn normalize_tool_calls_preserves_mapping_arguments() {
        let calls = serde_json::json!([{
            "id": "c1",
            "type": "function",
            "function": { "name": "w", "arguments": {"k": "v"} }
        }]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized[0]["function"]["arguments"], serde_json::json!({"k":"v"}));
    }

    /// Multi-call: each call's arguments is normalised independently.
    #[test]
    fn normalize_tool_calls_handles_multiple_entries() {
        let calls = serde_json::json!([
            { "id": "a", "type": "function", "function": { "name": "a", "arguments": "{\"x\":1}" } },
            { "id": "b", "type": "function", "function": { "name": "b", "arguments": "{\"y\":2}" } }
        ]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized[0]["function"]["arguments"]["x"], 1);
        assert_eq!(normalized[1]["function"]["arguments"]["y"], 2);
    }

    /// End-to-end template render against the live Gemma 4 tokenizer for
    /// a three-turn tool trajectory. The rendered prompt must:
    ///   * carry the training-shape `call:weather{city:"Bern"}`, **not**
    ///     the double-braced `{{"city":"Bern"}}` variant.
    ///   * carry the tool result as a `<|tool_response>response:weather`
    ///     block.
    /// This is the prompt the model sees on turn 2; if either is wrong
    /// the model ignores the tool result and re-emits the call.
    #[test]
    fn render_multi_turn_tool_trajectory_uses_training_shape() {
        let Some(dir) = std::env::var_os("RVLLM_TEST_MODEL_DIR") else {
            eprintln!("skip: RVLLM_TEST_MODEL_DIR not set");
            return;
        };
        let t = TokenizerHandle::load(std::path::Path::new(&dir)).expect("load");
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        }]);
        let msgs = vec![
            ChatMessage {
                role: Role::User, content: Some("wetter in bern?".into()),
                name: None, tool_calls: None, tool_call_id: None,
            },
            ChatMessage {
                role: Role::Assistant, content: None, name: None, tool_call_id: None,
                tool_calls: Some(serde_json::json!([{
                    "id": "c1", "type": "function",
                    "function": { "name": "weather", "arguments": "{\"city\":\"Bern\"}" }
                }])),
            },
            ChatMessage {
                role: Role::Tool, content: Some("{\"temp_c\":14,\"desc\":\"bewölkt\"}".into()),
                name: None, tool_calls: None, tool_call_id: Some("c1".into()),
            },
        ];
        let ids = t.render_chat(&msgs, Some(&tools)).expect("render");
        let back = t.decode_raw(&ids).expect("decode_raw");
        assert!(
            back.contains("call:weather{city:"),
            "expected training-shape tool call, got prompt:\n{back}",
        );
        assert!(
            !back.contains("call:weather{{"),
            "double-brace regression — would make the model loop:\n{back}",
        );
        assert!(
            back.contains("response:weather"),
            "tool response block must be rendered so the model reads the result:\n{back}",
        );
    }

    // === Cycle 36 pure unit tests for normalize_tool_calls edge cases ===

    #[test]
    fn normalize_tool_calls_handles_empty_array() {
        let calls = serde_json::json!([]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized, serde_json::json!([]));
    }

    #[test]
    fn normalize_tool_calls_skips_call_without_function() {
        let calls = serde_json::json!([{"id": "c1", "type": "function"}]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized, calls); // unchanged
    }

    #[test]
    fn normalize_tool_calls_skips_call_without_arguments() {
        let calls = serde_json::json!([{"function": {"name": "foo"}}]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized, calls); // unchanged
    }

    #[test]
    fn normalize_tool_calls_handles_nested_object_args() {
        // When args is a JSON-object string like `{"options":{"deep":1}}`,
        // the parsed object should preserve the nested structure.
        let calls = serde_json::json!([{
            "function": { "name": "foo", "arguments": "{\"options\":{\"deep\":1}}" }
        }]);
        let normalized = normalize_tool_calls(&calls);
        let args = &normalized[0]["function"]["arguments"];
        assert!(args.is_object());
        assert_eq!(args["options"]["deep"], serde_json::json!(1));
    }

    #[test]
    fn normalize_tool_calls_leaves_array_arguments_alone() {
        // If args is a JSON ARRAY string (not object), the existing
        // policy keeps it as a string. Defensive: array form is rare.
        let calls = serde_json::json!([{
            "function": { "name": "foo", "arguments": "[1,2,3]" }
        }]);
        let normalized = normalize_tool_calls(&calls);
        let args = &normalized[0]["function"]["arguments"];
        // The current policy: only OBJECT is parsed. Array string stays string.
        assert!(args.is_string(), "expected string, got {args:?}");
    }

    #[test]
    fn normalize_tool_calls_leaves_invalid_json_alone() {
        let calls = serde_json::json!([{
            "function": { "name": "foo", "arguments": "{bad json,," }
        }]);
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized[0]["function"]["arguments"], serde_json::json!("{bad json,,"));
    }

    #[test]
    fn normalize_tool_calls_top_level_non_array_is_passthrough() {
        // Defensive: should not panic if calls is somehow not an array.
        let calls = serde_json::json!({"not": "an array"});
        let normalized = normalize_tool_calls(&calls);
        assert_eq!(normalized, calls);
    }

    #[test]
    fn stream_decoder_incrementally_emits() {
        let Some(dir) = std::env::var_os("RVLLM_TEST_MODEL_DIR") else {
            eprintln!("skip: RVLLM_TEST_MODEL_DIR not set");
            return;
        };
        let t = TokenizerHandle::load(std::path::Path::new(&dir)).expect("load");
        let ids = t.encode("Hello, world!").expect("encode");
        let mut dec = t.stream_decoder();
        let mut pieces = Vec::new();
        for id in ids {
            pieces.push(dec.step(id).expect("step"));
        }
        let joined: String = pieces.concat();
        assert!(joined.contains("Hello"));
    }
}
