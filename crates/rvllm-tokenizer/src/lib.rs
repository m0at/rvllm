#![forbid(unsafe_code)]
//! Tokenization and detokenization for vllm-rs.
//!
//! Wraps the HuggingFace `tokenizers` crate with streaming decode,
//! chat template support, and special token detection.

pub mod chat;
pub mod incremental;
pub mod tokenizer;
pub mod tool_parser;

pub use chat::{ChatMessage, ChatRole};
pub use incremental::IncrementalDecoder;
pub use tokenizer::Tokenizer;
pub use tool_parser::{
    FunctionDefinition, ParsedToolCall, ToolDefinition, ToolParameterProperty, ToolParameters,
    ToolParseResult, ToolPromptStyle, format_tool_definitions, parse_tool_calls,
};
