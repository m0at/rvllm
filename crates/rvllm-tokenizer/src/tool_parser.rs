//! Parse tool/function call JSON from model output text.
//!
//! Supports two common tool-call formats emitted by instruction-tuned models:
//!
//! 1. **Hermes / ChatML style** -- delimited by `<tool_call>...</tool_call>` tags
//! 2. **Inline JSON** -- a bare JSON object (or array of objects) containing
//!    `"name"` and `"arguments"` keys
//!
//! The parser is intentionally lenient: it extracts the first plausible tool
//! call(s) from free-form model text, ignoring surrounding prose.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A single parsed tool call extracted from model output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParsedToolCall {
    /// Stable identifier for this call (assigned by caller, not parsed).
    pub id: String,
    /// Function/tool name.
    pub name: String,
    /// Raw JSON string of the arguments object.
    pub arguments: String,
}

/// Result of attempting to parse tool calls from model text.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolParseResult {
    /// Model produced plain text, no tool calls detected.
    PlainText(String),
    /// One or more tool calls were extracted.
    ToolCalls {
        /// Any text that preceded the first tool call.
        prefix_text: String,
        /// Parsed tool calls in order of appearance.
        calls: Vec<ParsedToolCall>,
    },
}

/// Format style for injecting tool definitions into the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPromptStyle {
    /// Hermes/ChatML: wrap definitions in `<tools>` tags, calls in `<tool_call>` tags.
    Hermes,
    /// Generic JSON: embed tool schemas as a JSON array in the system prompt.
    GenericJson,
}

/// Definition of a single tool parameter property.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParameterProperty {
    /// JSON Schema type (e.g. "string", "number", "integer", "boolean", "array", "object").
    #[serde(rename = "type")]
    pub param_type: String,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enum variants, if applicable.
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Parameters schema for a tool, following JSON Schema conventions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: std::collections::HashMap<String, ToolParameterProperty>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// An OpenAI-compatible tool definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Always "function" for function-calling tools.
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// Function metadata within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<ToolParameters>,
}

// ---------------------------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------------------------

/// Format tool definitions into a string suitable for injection into the
/// system prompt, according to the chosen style.
pub fn format_tool_definitions(tools: &[ToolDefinition], style: ToolPromptStyle) -> String {
    match style {
        ToolPromptStyle::Hermes => format_hermes_tools(tools),
        ToolPromptStyle::GenericJson => format_generic_json_tools(tools),
    }
}

fn format_hermes_tools(tools: &[ToolDefinition]) -> String {
    let mut out = String::from("You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:\n<tools>\n");
    for tool in tools {
        if let Ok(json) = serde_json::to_string(&tool) {
            out.push_str(&json);
            out.push('\n');
        }
    }
    out.push_str("</tools>\n\nFor each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    out
}

fn format_generic_json_tools(tools: &[ToolDefinition]) -> String {
    let mut out = String::from(
        "You have access to the following tools. To call a tool, respond with a JSON object with \"name\" and \"arguments\" keys.\n\nAvailable tools:\n",
    );
    if let Ok(json) = serde_json::to_string_pretty(tools) {
        out.push_str(&json);
    }
    out
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Parse model output text for tool calls.
///
/// Tries Hermes-style `<tool_call>` tags first, then falls back to scanning
/// for bare JSON objects with the expected shape.
pub fn parse_tool_calls(text: &str, call_id_prefix: &str) -> ToolParseResult {
    // Try Hermes-style tags first
    let hermes = parse_hermes_tool_calls(text, call_id_prefix);
    if let ToolParseResult::ToolCalls { .. } = &hermes {
        return hermes;
    }

    // Fallback: scan for bare JSON
    let json_result = parse_json_tool_calls(text, call_id_prefix);
    if let ToolParseResult::ToolCalls { .. } = &json_result {
        return json_result;
    }

    ToolParseResult::PlainText(text.to_string())
}

/// Parse `<tool_call>{ ... }</tool_call>` blocks.
fn parse_hermes_tool_calls(text: &str, call_id_prefix: &str) -> ToolParseResult {
    const OPEN_TAG: &str = "<tool_call>";
    const CLOSE_TAG: &str = "</tool_call>";

    let mut calls = Vec::new();
    let mut prefix_text = String::new();
    let mut search_start = 0;
    let mut found_first = false;

    while let Some(open) = text[search_start..].find(OPEN_TAG) {
        let abs_open = search_start + open;

        if !found_first {
            prefix_text = text[..abs_open].trim().to_string();
            found_first = true;
        }

        let content_start = abs_open + OPEN_TAG.len();
        if let Some(close) = text[content_start..].find(CLOSE_TAG) {
            let content = text[content_start..content_start + close].trim();
            if let Some(parsed) = try_parse_single_call(content) {
                let id = format!("call_{}{}", call_id_prefix, calls.len());
                calls.push(ParsedToolCall {
                    id,
                    name: parsed.0,
                    arguments: parsed.1,
                });
            }
            search_start = content_start + close + CLOSE_TAG.len();
        } else {
            // Unclosed tag -- try to parse the rest
            let content = text[content_start..].trim();
            if let Some(parsed) = try_parse_single_call(content) {
                let id = format!("call_{}{}", call_id_prefix, calls.len());
                calls.push(ParsedToolCall {
                    id,
                    name: parsed.0,
                    arguments: parsed.1,
                });
            }
            break;
        }
    }

    if calls.is_empty() {
        ToolParseResult::PlainText(text.to_string())
    } else {
        debug!(count = calls.len(), "parsed hermes-style tool calls");
        ToolParseResult::ToolCalls {
            prefix_text,
            calls,
        }
    }
}

/// Scan for bare JSON objects containing `"name"` and `"arguments"`.
fn parse_json_tool_calls(text: &str, call_id_prefix: &str) -> ToolParseResult {
    let mut calls = Vec::new();
    let mut prefix_text = String::new();
    let mut found_first = false;

    // Try to find JSON array first
    if let Some(start) = text.find('[') {
        if let Some(end) = rfind_matching_bracket(text, start) {
            let slice = &text[start..=end];
            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(slice) {
                if !found_first {
                    prefix_text = text[..start].trim().to_string();
                    found_first = true;
                }
                for item in &arr {
                    if let Some(parsed) = extract_name_arguments(item) {
                        let id = format!("call_{}{}", call_id_prefix, calls.len());
                        calls.push(ParsedToolCall {
                            id,
                            name: parsed.0,
                            arguments: parsed.1,
                        });
                    }
                }
            }
        }
    }

    if calls.is_empty() {
        // Try individual JSON objects
        let mut pos = 0;
        while pos < text.len() {
            if let Some(brace) = text[pos..].find('{') {
                let abs = pos + brace;
                if let Some(end) = rfind_matching_bracket(text, abs) {
                    let slice = &text[abs..=end];
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(slice) {
                        if let Some(parsed) = extract_name_arguments(&val) {
                            if !found_first {
                                prefix_text = text[..abs].trim().to_string();
                                found_first = true;
                            }
                            let id = format!("call_{}{}", call_id_prefix, calls.len());
                            calls.push(ParsedToolCall {
                                id,
                                name: parsed.0,
                                arguments: parsed.1,
                            });
                            pos = end + 1;
                            continue;
                        }
                    }
                    pos = abs + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    if calls.is_empty() {
        ToolParseResult::PlainText(text.to_string())
    } else {
        debug!(count = calls.len(), "parsed json-style tool calls");
        ToolParseResult::ToolCalls {
            prefix_text,
            calls,
        }
    }
}

/// Try to parse a JSON string as a tool call with `name` and `arguments`.
fn try_parse_single_call(json_str: &str) -> Option<(String, String)> {
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    extract_name_arguments(&val)
}

/// Extract `(name, arguments_json)` from a JSON value.
fn extract_name_arguments(val: &serde_json::Value) -> Option<(String, String)> {
    let obj = val.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj.get("arguments")?;
    let args_str = if args.is_string() {
        args.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(args).ok()?
    };
    Some((name, args_str))
}

/// Find the matching closing bracket/brace for an opening one at `start`.
fn rfind_matching_bracket(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let (open, close) = match bytes[start] {
        b'{' => (b'{', b'}'),
        b'[' => (b'[', b']'),
        _ => return None,
    };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(start + i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hermes_single_tool_call() {
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>"#;
        let result = parse_tool_calls(text, "test_");
        match result {
            ToolParseResult::ToolCalls { calls, prefix_text } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert!(calls[0].arguments.contains("San Francisco"));
                assert_eq!(calls[0].id, "call_test_0");
                assert!(prefix_text.is_empty());
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_hermes_parallel_tool_calls() {
        let text = r#"I'll check both for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "LA"}}
</tool_call>"#;
        let result = parse_tool_calls(text, "p_");
        match result {
            ToolParseResult::ToolCalls { calls, prefix_text } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[1].name, "get_weather");
                assert!(calls[0].arguments.contains("NYC"));
                assert!(calls[1].arguments.contains("LA"));
                assert_eq!(prefix_text, "I'll check both for you.");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_bare_json_tool_call() {
        let text = r#"Sure, let me look that up.
{"name": "search", "arguments": {"query": "rust programming"}}"#;
        let result = parse_tool_calls(text, "j_");
        match result {
            ToolParseResult::ToolCalls { calls, prefix_text } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(prefix_text, "Sure, let me look that up.");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_json_array_tool_calls() {
        let text = r#"[{"name": "fn_a", "arguments": {"x": 1}}, {"name": "fn_b", "arguments": {"y": 2}}]"#;
        let result = parse_tool_calls(text, "a_");
        match result {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "fn_a");
                assert_eq!(calls[1].name, "fn_b");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_plain_text_no_tools() {
        let text = "The weather in San Francisco is currently 65 degrees.";
        let result = parse_tool_calls(text, "x_");
        match result {
            ToolParseResult::PlainText(t) => assert_eq!(t, text),
            _ => panic!("expected PlainText"),
        }
    }

    #[test]
    fn parse_arguments_as_string() {
        let text = r#"<tool_call>
{"name": "calc", "arguments": "{\"expr\": \"2+2\"}"}
</tool_call>"#;
        let result = parse_tool_calls(text, "s_");
        match result {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "calc");
                // arguments was a JSON string, should be kept as-is
                assert!(calls[0].arguments.contains("2+2"));
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn format_hermes_tools_output() {
        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                parameters: Some(ToolParameters {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut m = std::collections::HashMap::new();
                        m.insert(
                            "location".to_string(),
                            ToolParameterProperty {
                                param_type: "string".to_string(),
                                description: Some("City name".to_string()),
                                enum_values: None,
                            },
                        );
                        m
                    },
                    required: vec!["location".to_string()],
                }),
            },
        }];
        let formatted = format_tool_definitions(&tools, ToolPromptStyle::Hermes);
        assert!(formatted.contains("<tools>"));
        assert!(formatted.contains("</tools>"));
        assert!(formatted.contains("get_weather"));
        assert!(formatted.contains("<tool_call>"));
    }

    #[test]
    fn format_generic_json_tools_output() {
        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let formatted = format_tool_definitions(&tools, ToolPromptStyle::GenericJson);
        assert!(formatted.contains("search"));
        assert!(formatted.contains("Available tools"));
    }

    #[test]
    fn rfind_matching_bracket_nested() {
        let s = r#"{"a": {"b": [1, 2]}, "c": "}"}"#;
        let end = rfind_matching_bracket(s, 0).unwrap();
        // Should match the outermost closing brace
        let parsed: serde_json::Value = serde_json::from_str(&s[..=end]).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn rfind_matching_bracket_unclosed() {
        assert!(rfind_matching_bracket("{abc", 0).is_none());
    }

    #[test]
    fn tool_definition_serde_roundtrip() {
        let td = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "test_fn".to_string(),
                description: Some("A test".to_string()),
                parameters: None,
            },
        };
        let json = serde_json::to_string(&td).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back, td);
    }

    #[test]
    fn parsed_tool_call_serde_roundtrip() {
        let tc = ParsedToolCall {
            id: "call_0".to_string(),
            name: "foo".to_string(),
            arguments: r#"{"bar": 42}"#.to_string(),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let back: ParsedToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back, tc);
    }

    #[test]
    fn hermes_unclosed_tag_still_parses() {
        let text = r#"<tool_call>
{"name": "ping", "arguments": {}}"#;
        let result = parse_tool_calls(text, "u_");
        match result {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "ping");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn nested_json_in_arguments() {
        let text = r#"<tool_call>
{"name": "create_file", "arguments": {"path": "/tmp/test", "content": "{\"key\": \"value\"}"}}
</tool_call>"#;
        let result = parse_tool_calls(text, "n_");
        match result {
            ToolParseResult::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "create_file");
            }
            _ => panic!("expected ToolCalls"),
        }
    }
}
