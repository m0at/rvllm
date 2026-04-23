//! Gemma 4 tool-call extraction.
//!
//! Gemma 4 emits tool calls as plain text inside the assistant reply,
//! wrapped in special-token strings: `<|tool_call>call:NAME{ARGS}<tool_call|>`
//! (some variants close with `<turn|>` instead). If we leave that in
//! `content`, the client sees the raw markup as a normal assistant
//! message — which is what just bit zeroclaw. This module extracts
//! the calls so the handler can hoist them into the OpenAI-shaped
//! `tool_calls` array and set `finish_reason = "tool_calls"`.
//!
//! Ported from vLLM's `vllm/tool_parsers/gemma4_utils.py`. Two tiers:
//!   * **tier-1** — strict `<|tool_call>call:NAME{ARGS}<tool_call|>`.
//!     Matches what the model emits with the special tokens intact.
//!   * **tier-2** — bare `call:NAME{ARGS}` anchored at start-of-string
//!     or after whitespace. Fires when the decoder strips the
//!     `<|tool_call>` special tokens (our `TokenizerHandle::decode`
//!     runs with `skip_special_tokens=true`, which is how this path
//!     first surfaced: zeroclaw saw `call:get_weather{location:"..."}`
//!     land in `content` instead of as a structured tool call).

use serde::Serialize;
use serde_json::{Map, Value};

/// One extracted call. `arguments` is the JSON-string form expected by
/// OpenAI's `tool_calls[].function.arguments` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

const START: &str = "<|tool_call>";
const END_A: &str = "<tool_call|>";
const END_B: &str = "<turn|>";
const ESCAPE: &str = "<|\"|>";
const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";
// Gemma 4 wraps its pre-answer reasoning in multiple block shapes:
//   <|channel>thought\n...<channel|>
//   <|tool_response>thought\n...<channel|>
//   <thought\n...<channel|>     (model hallucination without the leading `|`)
// They all close with `<channel|>` and must be dropped from user content
// wholesale — the inner prose is the model's "what do I think the answer is"
// draft, which should never be shown to the user.
const THOUGHT_BLOCK_OPENERS: &[&str] = &[
    "<|channel>",
    "<|tool_response>",
    "<thought",
];

/// Extract all Gemma 4 tool calls from decoded text.
///
/// Returns an empty vec when no markup is present — callers should
/// treat that as a plain text response.
pub fn parse_gemma4_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut out = parse_tier1(text);
    if out.is_empty() {
        out = parse_tier2_bare(text);
    }
    out
}

fn parse_tier1(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let bytes = text.as_bytes();

    while cursor < bytes.len() {
        let Some(rel) = text[cursor..].find(START) else { break };
        let after_start = cursor + rel + START.len();

        // After `<|tool_call>` the format is `call:NAME{ARGS}END`.
        let rest = &text[after_start..];
        let Some(stripped) = rest.strip_prefix("call:") else {
            cursor = after_start;
            continue;
        };
        // Name = [A-Za-z0-9_]+ up to `{`.
        let brace = match stripped.find('{') {
            Some(i) => i,
            None => break,
        };
        let name = &stripped[..brace];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            cursor = after_start;
            continue;
        }
        let args_start = brace + 1; // past `{`
        let args_region = &stripped[args_start..];

        // Find the terminator. Args may contain `}` inside a JSON value,
        // so we search for the closing tag first and then trim one `}`.
        let (args_end_in_region, tag_len) = match (args_region.find(END_A), args_region.find(END_B)) {
            (Some(a), Some(b)) if a < b => (a, END_A.len()),
            (Some(_a), Some(b)) => (b, END_B.len()),
            (Some(a), None) => (a, END_A.len()),
            (None, Some(b)) => (b, END_B.len()),
            (None, None) => break,
        };
        let raw_args = &args_region[..args_end_in_region];
        let trimmed = raw_args.strip_suffix('}').unwrap_or(raw_args);

        out.push(ParsedToolCall {
            name: name.to_string(),
            arguments: arguments_to_json_string(trimmed),
        });

        cursor = after_start + "call:".len() + brace + 1 + args_end_in_region + tag_len;
    }

    out
}

/// Tier-2: bare `call:NAME{ARGS}` at start-of-string or after
/// whitespace. Matches what remains when a `skip_special_tokens=true`
/// decoder strips `<|tool_call>` / `<tool_call|>`.
///
/// UTF-8 note: the markers (`call:`, `{`, `}`) are ASCII, so marker
/// matching is byte-oriented — but `i` must land on a char boundary
/// before we can slice `&text[..]`. Output from Gemma 4 routinely
/// contains non-ASCII (German prose, emoji) that would otherwise
/// panic `str::is_char_boundary`.
fn parse_tier2_bare(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 5 <= bytes.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        // Require start-of-string or whitespace immediately before `call:`.
        let anchored = i == 0 || (bytes[i - 1] as char).is_whitespace();
        if !anchored || &bytes[i..i + 5] != b"call:" {
            i += 1;
            continue;
        }
        let name_start = i + 5;
        let mut j = name_start;
        while j < bytes.len() && {
            let c = bytes[j] as char;
            c.is_ascii_alphanumeric() || c == '_'
        } {
            j += 1;
        }
        if j == name_start || j >= bytes.len() || bytes[j] != b'{' {
            i += 1;
            continue;
        }
        let name = &text[name_start..j];
        let args_start = j + 1;
        // First `}` after args_start terminates. Tier-2 doesn't handle
        // braces inside string values; Gemma 4's bare-call output does
        // not nest in practice.
        let Some(args_end_rel) = text[args_start..].find('}') else {
            break;
        };
        let args = &text[args_start..args_start + args_end_rel];
        out.push(ParsedToolCall {
            name: name.to_string(),
            arguments: arguments_to_json_string(args),
        });
        i = args_start + args_end_rel + 1;
    }
    out
}

/// Render the inner args region into a JSON object string.
/// Prefers direct JSON parse (Gemma 4 escape-token handled). On
/// failure, returns `{}` — bubbling up a malformed call is worse than
/// letting the client decide.
fn arguments_to_json_string(args: &str) -> String {
    let cleaned = args.replace(ESCAPE, "\"");
    let wrapped = format!("{{{}}}", cleaned);
    if let Ok(v) = serde_json::from_str::<Value>(&wrapped) {
        return serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string());
    }

    // Fallback: harvest `key: "value"` pairs with a tiny hand-rolled scanner
    // rather than pulling in a regex dep for one edge case.
    let mut map = Map::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && {
            let c = bytes[i] as char;
            c.is_ascii_alphanumeric() || c == '_'
        } {
            i += 1;
        }
        if i == key_start {
            break;
        }
        let key = &cleaned[key_start..i];
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            break;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'"' {
            break;
        }
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let val = &cleaned[val_start..i];
        map.insert(key.to_string(), Value::String(val.to_string()));
        i += 1; // closing quote
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

/// Strip any Gemma 4 tool-call markup + leading/trailing whitespace
/// so the plain-text path has a clean payload when no calls are emitted
/// (or, for mixed output, for the prefix before the first call).
/// Handles both tier-1 `<|tool_call>...<tool_call|>` wrappers and
/// tier-2 bare `call:NAME{...}` patterns.
pub fn strip_tool_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Markers are ASCII and therefore always start at a char boundary;
        // checking `is_char_boundary` first makes the subsequent `text[i..]`
        // slice safe when non-ASCII prose (e.g. German umlauts) sits in the
        // middle of the stream.
        if text.is_char_boundary(i) {
            // Drop entire thought / tool_response / hallucinated-thought blocks —
            // each closes with `<channel|>` and carries Gemma's internal
            // reasoning, never user-facing prose.
            let mut matched_thought = false;
            for opener in THOUGHT_BLOCK_OPENERS {
                if text[i..].starts_with(opener) {
                    // Prefer `<channel|>` as the closer; if Gemma terminated the
                    // block with `<turn|>` (end-of-turn) or just let it trail
                    // off, match that too rather than eating the rest of the
                    // reply. No closer found → only strip the opener itself so
                    // real content after it survives.
                    let rest = &text[i + opener.len()..];
                    let close = match (rest.find(CHANNEL_CLOSE), rest.find("<turn|>")) {
                        (Some(a), Some(b)) if a <= b => Some((a, CHANNEL_CLOSE.len())),
                        (Some(_), Some(b)) => Some((b, "<turn|>".len())),
                        (Some(a), None) => Some((a, CHANNEL_CLOSE.len())),
                        (None, Some(b)) => Some((b, "<turn|>".len())),
                        (None, None) => None,
                    };
                    let skip = match close {
                        Some((rel, close_len)) => opener.len() + rel + close_len,
                        None => opener.len(),
                    };
                    i += skip;
                    matched_thought = true;
                    break;
                }
            }
            if matched_thought {
                continue;
            }
            if text[i..].starts_with(START) {
                let rest = &text[i + START.len()..];
                let skip = match (rest.find(END_A), rest.find(END_B)) {
                    (Some(a), Some(b)) if a <= b => START.len() + a + END_A.len(),
                    (Some(_a), Some(b)) => START.len() + b + END_B.len(),
                    (Some(a), None) => START.len() + a + END_A.len(),
                    (None, Some(b)) => START.len() + b + END_B.len(),
                    (None, None) => bytes.len() - i,
                };
                i += skip;
                continue;
            }
            // Tier-2 bare `call:NAME{...}` — strip when anchored.
            let anchored = i == 0
                || out.chars().last().map(|c| c.is_whitespace()).unwrap_or(false);
            if anchored && text[i..].starts_with("call:") {
                let after = i + "call:".len();
                let mut j = after;
                while j < bytes.len() && {
                    let c = bytes[j] as char;
                    c.is_ascii_alphanumeric() || c == '_'
                } {
                    j += 1;
                }
                if j > after && j < bytes.len() && bytes[j] == b'{' {
                    if let Some(end_rel) = text[j + 1..].find('}') {
                        i = j + 1 + end_rel + 1;
                        continue;
                    }
                }
            }
        }
        // Copy one whole UTF-8 scalar. If `i` is mid-codepoint (can happen
        // after a `skip` jump landed mid-sequence on malformed input), step
        // one byte to resynchronise — the lost byte will be replaced by the
        // next `chars()` decode.
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    // Final sweep: Gemma 4's raw output sprinkles other single-token control
    // markers (`<|tool_response>`, `<turn|>`, `<|turn>`, literal `<thought`
    // fragments the model hallucinates) that the named passes above don't
    // enumerate. They all share a common shape (`<|…>` or `<…|>`), so a
    // conservative token sweep is enough — and any false positive would
    // have to be Gemma control-token-looking text in the middle of a
    // legitimate reply, which is vanishingly rare.
    strip_stray_control_markers(&out).trim().to_string()
}

fn strip_stray_control_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        if text.is_char_boundary(i) {
            // `<|…>` — opening-style control marker.
            if text[i..].starts_with("<|") {
                if let Some(rel) = text[i + 2..].find('>') {
                    i += 2 + rel + 1;
                    continue;
                }
            }
            // `<…|>` — closing-style control marker.
            if let Some(ch) = text[i..].chars().next() {
                if ch == '<' {
                    if let Some(rel) = text[i + 1..].find("|>") {
                        let inside = &text[i + 1..i + 1 + rel];
                        // Guard against eating a legitimate `<` in prose: only
                        // strip when the body is a bare token name (letters /
                        // digits / underscore), matching Gemma's control shape.
                        if !inside.is_empty()
                            && inside.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                        {
                            i += 1 + rel + 2;
                            continue;
                        }
                    }
                }
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_call_json_args() {
        let s = r#"<|tool_call>call:get_weather{"city":"Zurich"}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Zurich"}"#);
    }

    #[test]
    fn extracts_call_with_turn_terminator() {
        let s = r#"<|tool_call>call:ping{}<turn|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn handles_escape_token() {
        let s = "<|tool_call>call:foo{q:<|\"|>hello<|\"|>}<tool_call|>";
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"q":"hello"}"#);
    }

    #[test]
    fn extracts_multiple_calls() {
        let s = r#"<|tool_call>call:a{"x":1}<tool_call|> noise <|tool_call>call:b{"y":2}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn plain_text_yields_nothing() {
        let calls = parse_gemma4_tool_calls("Paris is the capital of France.");
        assert!(calls.is_empty());
    }

    #[test]
    fn tier2_bare_call_extracted() {
        // What Gemma 4 actually emits to rvllm-serve once special
        // tokens are stripped — this is the exact payload that
        // tripped zeroclaw in prod.
        let s = r#"call:get_weather{location: "Zurich"}thought"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"location":"Zurich"}"#);
    }

    #[test]
    fn tier2_only_triggers_when_anchored() {
        // "recall:foo{..}" must not match tier-2.
        let calls = parse_gemma4_tool_calls("the recall:foo{x:1} is fine");
        assert!(calls.is_empty());
    }

    #[test]
    fn strip_sweeps_stray_control_markers() {
        // `<|tool_response>…<turn|>` is a reasoning block — its body (here
        // "ok") is the model's pre-answer draft and must stay hidden.
        // Trailing stray `<|turn>` is a control marker the sweep removes.
        let s = "<|tool_response>ok<turn|> Das ist in Ordnung.<|turn> ";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "Das ist in Ordnung.");
    }

    #[test]
    fn strip_drops_hallucinated_thought_fragment() {
        // Regression for the live failure: Gemma replied with the leaked
        // pattern `<|tool_response>thought\n<channel|><thought\n<channel|>
        // Das Wetter ...`. Every bracketed marker should vanish; only the
        // real answer survives.
        let s = "<|tool_response>thought\n<channel|>\
                 <thought\n<channel|>\
                 Das Wetter in Bern ist heute bewölkt.<turn|>";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "Das Wetter in Bern ist heute bewölkt.");
    }

    #[test]
    fn strip_keeps_math_inequality() {
        // Regression guard — the stray-sweep must not chew up `<something|>`
        // shapes that aren't Gemma tokens. A sentence like "5<3|>" is weird
        // but the body "3" is numeric → alphanumeric → would strip. That's
        // accepted; the same body with punctuation (`"3, 4"`) must NOT.
        let s = "a<3, 4|>b";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "a<3, 4|>b");
    }

    #[test]
    fn strip_drops_thought_channel() {
        // Gemma 4 writes its internal reasoning inside `<|channel>thought...<channel|>`.
        // Without explicit stripping the tokenizer's skip-specials pass drops
        // the markers but leaves the reasoning prose in user-visible content.
        let s = "<|channel>thought\nDas wird wohl 14°C sein.<channel|>\
                 <|tool_call>call:weather{city:\"Bern\"}<tool_call|>";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "", "thought + tool_call markup should leave empty content, got {stripped:?}");
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Bern"}"#);
    }

    #[test]
    fn strip_handles_utf8_prose() {
        // Regression: the parser used to panic with
        // `byte index is not a char boundary; it is inside 'ü'` when
        // Gemma 4 emitted a call followed by German prose.
        let s = "call:weather{city:Bern} Es regnet in Zürich und München.";
        let stripped = strip_tool_markup(s);
        assert!(stripped.contains("Zürich"));
        assert!(!stripped.contains("call:"));
    }

    #[test]
    fn parse_tier2_handles_utf8_prose() {
        // Same regression for the tier-2 matcher — must walk past
        // non-ASCII bytes without panicking.
        let s = "Die Antwort: ü call:ping{x:1}";
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
    }

    #[test]
    fn strip_removes_markup_keeps_prose() {
        let s = r#"Let me check. <|tool_call>call:foo{"a":1}<tool_call|> done."#;
        assert_eq!(strip_tool_markup(s), "Let me check.  done.");
    }
}
