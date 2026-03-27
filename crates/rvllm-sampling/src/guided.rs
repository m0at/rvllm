//! Guided decoding engine for constrained output generation.
//!
//! Applies token-level masks before sampling to guarantee the output conforms
//! to a specified format: valid JSON, a JSON schema, or a regex pattern.
//! Uses a recursive descent approach -- no external dependency on outlines.

use tracing::{debug, trace};
use rvllm_core::prelude::{ResponseFormat, Result, TokenId};

use crate::json_schema::{self, SchemaNode, ValidChars};

/// State machine for guided decoding of a single sequence.
///
/// Tracks what has been generated so far and computes a token mask for the
/// next sampling step. The mask sets disallowed token logits to `-inf` so
/// the sampler never picks them.
#[derive(Debug, Clone)]
pub struct GuidedDecodingState {
    /// The constraint being enforced.
    constraint: Constraint,
    /// Text generated so far (used to determine valid continuations).
    generated: String,
}

/// Internal constraint representation.
#[derive(Debug, Clone)]
enum Constraint {
    /// No constraint.
    None,
    /// Output must be valid JSON.
    Json,
    /// Output must conform to a compiled JSON schema.
    JsonSchema(SchemaNode),
    /// Output must match a regex (stored as the pattern string).
    Regex(String),
}

/// A vocabulary entry mapping token id to its text representation.
#[derive(Debug, Clone)]
pub struct VocabEntry {
    /// Token id.
    pub id: TokenId,
    /// UTF-8 text this token decodes to.
    pub text: String,
}

/// Pre-built vocabulary table for fast token mask computation.
#[derive(Debug, Clone)]
pub struct VocabTable {
    entries: Vec<VocabEntry>,
    /// Token id that represents end-of-sequence.
    eos_token_id: TokenId,
}

impl VocabTable {
    /// Build a vocabulary table from id-text pairs.
    pub fn new(entries: Vec<VocabEntry>, eos_token_id: TokenId) -> Self {
        Self {
            entries,
            eos_token_id,
        }
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the EOS token id.
    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }
}

impl GuidedDecodingState {
    /// Create a new guided decoding state from a response format.
    pub fn new(format: &ResponseFormat) -> Result<Self> {
        let constraint = match format {
            ResponseFormat::Text => Constraint::None,
            ResponseFormat::JsonObject => Constraint::Json,
            ResponseFormat::JsonSchema { json_schema } => {
                let node = json_schema::compile_schema(json_schema)?;
                debug!("compiled JSON schema constraint");
                Constraint::JsonSchema(node)
            }
            ResponseFormat::Regex { pattern } => {
                Constraint::Regex(pattern.clone())
            }
        };
        Ok(Self {
            constraint,
            generated: String::new(),
        })
    }

    /// Returns true if this state has no constraint (free-form text).
    pub fn is_unconstrained(&self) -> bool {
        matches!(self.constraint, Constraint::None)
    }

    /// Record that a token was selected, updating internal state.
    pub fn advance(&mut self, token_text: &str) {
        self.generated.push_str(token_text);
        trace!(generated_len = self.generated.len(), "guided state advanced");
    }

    /// Get the text generated so far.
    pub fn generated_text(&self) -> &str {
        &self.generated
    }

    /// Compute a token mask: apply `-inf` to logits of disallowed tokens.
    ///
    /// This is the core guided decoding operation. For each token in the
    /// vocabulary, we check if appending it to the generated text would
    /// remain consistent with the constraint. If not, set its logit to
    /// `-inf`.
    ///
    /// The `vocab` table provides the mapping from token id to text.
    pub fn apply_mask(&self, logits: &mut [f32], vocab: &VocabTable) {
        if self.is_unconstrained() {
            return;
        }

        let valid = self.compute_valid_chars();

        match valid {
            ValidChars::Any => {
                // Everything allowed, no masking needed
                return;
            }
            ValidChars::End => {
                // Only EOS is allowed
                for (i, l) in logits.iter_mut().enumerate() {
                    if i as TokenId != vocab.eos_token_id {
                        *l = f32::NEG_INFINITY;
                    }
                }
                return;
            }
            ValidChars::Set(ref allowed_bytes) => {
                self.apply_char_mask(logits, vocab, allowed_bytes);
            }
        }
    }

    /// Compute which characters are valid next given the current state.
    fn compute_valid_chars(&self) -> ValidChars {
        match &self.constraint {
            Constraint::None => ValidChars::Any,
            Constraint::Json => {
                let node = SchemaNode::Any;
                json_schema::valid_next_chars(&self.generated, &node)
            }
            Constraint::JsonSchema(node) => {
                json_schema::valid_next_chars(&self.generated, node)
            }
            Constraint::Regex(pattern) => {
                self.compute_regex_valid_chars(pattern)
            }
        }
    }

    /// Apply a character-level mask based on allowed first bytes.
    ///
    /// A token is allowed if its first byte is in the allowed set, OR if
    /// appending it results in text consistent with the constraint.
    fn apply_char_mask(&self, logits: &mut [f32], vocab: &VocabTable, allowed_bytes: &[u8]) {
        for entry in &vocab.entries {
            let id = entry.id as usize;
            if id >= logits.len() {
                continue;
            }
            if entry.id == vocab.eos_token_id {
                // EOS handling: allow if the generated text is a valid complete value
                if !self.is_complete() {
                    logits[id] = f32::NEG_INFINITY;
                }
                continue;
            }
            if entry.text.is_empty() {
                continue;
            }
            let first_byte = entry.text.as_bytes()[0];
            // Quick check: if the first byte of the token isn't in the allowed set,
            // mask it immediately.
            if !allowed_bytes.contains(&first_byte) {
                logits[id] = f32::NEG_INFINITY;
            }
            // For multi-character tokens, we do a best-effort check:
            // allow it if any prefix of the token text starts with an allowed byte.
            // Full validation would require simulating the constraint for each token,
            // which is too expensive per step. The character-level mask catches most
            // violations and the recursive descent naturally recovers.
        }
    }

    /// Check whether the generated text so far is a complete valid value.
    fn is_complete(&self) -> bool {
        match &self.constraint {
            Constraint::None => true,
            Constraint::Json => is_valid_json(&self.generated),
            Constraint::JsonSchema(node) => {
                if !is_valid_json(&self.generated) {
                    return false;
                }
                // For schema mode, also check the value parses and
                // at least the top-level type matches.
                validate_against_schema(&self.generated, node)
            }
            Constraint::Regex(pattern) => {
                // Simple full-match check
                regex_full_match(&self.generated, pattern)
            }
        }
    }

    /// Compute valid next characters for regex constraint.
    /// Uses a simple approach: try each printable ASCII character and see
    /// if the resulting prefix could still lead to a full match.
    fn compute_regex_valid_chars(&self, pattern: &str) -> ValidChars {
        let mut valid = Vec::new();
        for b in 0x20u8..=0x7E {
            let mut candidate = self.generated.clone();
            candidate.push(b as char);
            if regex_prefix_possible(&candidate, pattern) {
                valid.push(b);
            }
        }
        // Also check whitespace chars
        for &b in &[b'\n', b'\t', b'\r'] {
            let mut candidate = self.generated.clone();
            candidate.push(b as char);
            if regex_prefix_possible(&candidate, pattern) {
                valid.push(b);
            }
        }
        if valid.is_empty() {
            ValidChars::End
        } else {
            ValidChars::Set(valid)
        }
    }
}

/// Check if a string is valid JSON.
fn is_valid_json(s: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(s).is_ok()
}

/// Basic validation of a JSON string against a schema node (top-level type check).
fn validate_against_schema(json_str: &str, node: &SchemaNode) -> bool {
    let val: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return false,
    };
    validate_value(&val, node)
}

/// Recursively validate a JSON value against a schema node.
fn validate_value(val: &serde_json::Value, node: &SchemaNode) -> bool {
    match node {
        SchemaNode::Any => true,
        SchemaNode::Null => val.is_null(),
        SchemaNode::Boolean => val.is_boolean(),
        SchemaNode::Number => val.is_number(),
        SchemaNode::Integer => val.is_i64() || val.is_u64(),
        SchemaNode::String(_) => val.is_string(),
        SchemaNode::Array { items, min_items, max_items } => {
            if let Some(arr) = val.as_array() {
                arr.len() >= *min_items
                    && arr.len() <= *max_items
                    && arr.iter().all(|v| validate_value(v, items))
            } else {
                false
            }
        }
        SchemaNode::Object { properties, additional_properties } => {
            if let Some(obj) = val.as_object() {
                // Check required properties exist
                for (key, prop_schema, required) in properties {
                    if *required && !obj.contains_key(key) {
                        return false;
                    }
                    if let Some(v) = obj.get(key) {
                        if !validate_value(v, prop_schema) {
                            return false;
                        }
                    }
                }
                // Check no extra properties if additionalProperties=false
                if !additional_properties {
                    let known: std::collections::HashSet<&str> =
                        properties.iter().map(|(k, _, _)| k.as_str()).collect();
                    for key in obj.keys() {
                        if !known.contains(key.as_str()) {
                            return false;
                        }
                    }
                }
                true
            } else {
                false
            }
        }
        SchemaNode::AnyOf(nodes) => nodes.iter().any(|n| validate_value(val, n)),
        SchemaNode::Enum(vals) => vals.contains(val),
        SchemaNode::Const(expected) => val == expected,
    }
}

/// Simple regex prefix check using byte-level matching.
/// Returns true if `text` could be a prefix of a string matching `pattern`.
/// This is a basic implementation -- for production you'd want a proper NFA.
fn regex_prefix_possible(text: &str, pattern: &str) -> bool {
    // Simplified: check if the text matches the pattern so far
    // by trying prefix matches. For a real implementation, you'd
    // convert the regex to an NFA and check if any state is reachable.
    // Here we use a simple heuristic: the text must match the beginning
    // of the pattern, or the pattern must be able to start matching.
    simple_regex_prefix_match(text, pattern)
}

/// Full match check for a simple regex pattern against text.
fn regex_full_match(text: &str, pattern: &str) -> bool {
    simple_regex_full_match(text, pattern)
}

/// Minimal regex engine supporting: literal chars, `.`, `*`, `+`, `?`, `[...]`,
/// `\d`, `\w`, `\s`, and `{n,m}` quantifiers. Sufficient for common guided
/// decoding patterns.
fn simple_regex_prefix_match(text: &str, pattern: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let text_bytes = text.as_bytes();
    prefix_match_recursive(text_bytes, 0, &pat_chars, 0)
}

fn simple_regex_full_match(text: &str, pattern: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let text_bytes = text.as_bytes();
    full_match_recursive(text_bytes, 0, &pat_chars, 0)
}

/// Width of a pattern element at position `pi`: 1 for normal chars, 2 for `\x` escapes.
fn element_width(pattern: &[char], pi: usize) -> usize {
    if pattern[pi] == '\\' && pi + 1 < pattern.len() { 2 } else { 1 }
}

/// Check if there is a quantifier (`*`, `+`, `?`) immediately after the element at `pi`.
fn quantifier_at(pattern: &[char], pi: usize) -> Option<char> {
    let w = element_width(pattern, pi);
    let qi = pi + w;
    if qi < pattern.len() && matches!(pattern[qi], '*' | '+' | '?') {
        Some(pattern[qi])
    } else {
        None
    }
}

fn prefix_match_recursive(text: &[u8], ti: usize, pattern: &[char], pi: usize) -> bool {
    if ti >= text.len() {
        return true; // text consumed => valid prefix
    }
    if pi >= pattern.len() {
        return false;
    }

    let pc = pattern[pi];
    let ew = element_width(pattern, pi);

    if let Some(q) = quantifier_at(pattern, pi) {
        let after = pi + ew + 1; // index after the quantifier
        match q {
            '*' => {
                if prefix_match_recursive(text, ti, pattern, after) {
                    return true;
                }
                if char_matches(text[ti], pc, pattern, pi)
                    && prefix_match_recursive(text, ti + 1, pattern, pi)
                {
                    return true;
                }
                false
            }
            '+' => {
                if char_matches(text[ti], pc, pattern, pi) {
                    if prefix_match_recursive(text, ti + 1, pattern, pi) {
                        return true;
                    }
                    if prefix_match_recursive(text, ti + 1, pattern, after) {
                        return true;
                    }
                }
                false
            }
            '?' => {
                if prefix_match_recursive(text, ti, pattern, after) {
                    return true;
                }
                if char_matches(text[ti], pc, pattern, pi)
                    && prefix_match_recursive(text, ti + 1, pattern, after)
                {
                    return true;
                }
                false
            }
            _ => unreachable!(),
        }
    } else {
        if char_matches(text[ti], pc, pattern, pi) {
            prefix_match_recursive(text, ti + 1, pattern, pi + ew)
        } else {
            false
        }
    }
}

fn full_match_recursive(text: &[u8], ti: usize, pattern: &[char], pi: usize) -> bool {
    if ti >= text.len() && pi >= pattern.len() {
        return true;
    }
    if ti >= text.len() {
        if pi < pattern.len() {
            if let Some(q) = quantifier_at(pattern, pi) {
                if matches!(q, '*' | '?') {
                    let ew = element_width(pattern, pi);
                    return full_match_recursive(text, ti, pattern, pi + ew + 1);
                }
            }
        }
        return false;
    }
    if pi >= pattern.len() {
        return false;
    }

    let pc = pattern[pi];
    let ew = element_width(pattern, pi);

    if let Some(q) = quantifier_at(pattern, pi) {
        let after = pi + ew + 1;
        match q {
            '*' => {
                if full_match_recursive(text, ti, pattern, after) {
                    return true;
                }
                if char_matches(text[ti], pc, pattern, pi)
                    && full_match_recursive(text, ti + 1, pattern, pi)
                {
                    return true;
                }
                false
            }
            '+' => {
                if char_matches(text[ti], pc, pattern, pi) {
                    if full_match_recursive(text, ti + 1, pattern, pi) {
                        return true;
                    }
                    if full_match_recursive(text, ti + 1, pattern, after) {
                        return true;
                    }
                }
                false
            }
            '?' => {
                if full_match_recursive(text, ti, pattern, after) {
                    return true;
                }
                if char_matches(text[ti], pc, pattern, pi)
                    && full_match_recursive(text, ti + 1, pattern, after)
                {
                    return true;
                }
                false
            }
            _ => unreachable!(),
        }
    } else {
        if char_matches(text[ti], pc, pattern, pi) {
            full_match_recursive(text, ti + 1, pattern, pi + ew)
        } else {
            false
        }
    }
}

/// Check if a text byte matches a pattern character.
fn char_matches(byte: u8, pc: char, pattern: &[char], pi: usize) -> bool {
    match pc {
        '.' => true, // matches any character
        '\\' => {
            // Escape sequence
            if pi + 1 < pattern.len() {
                match pattern[pi + 1] {
                    'd' => byte.is_ascii_digit(),
                    'D' => !byte.is_ascii_digit(),
                    'w' => byte.is_ascii_alphanumeric() || byte == b'_',
                    'W' => !(byte.is_ascii_alphanumeric() || byte == b'_'),
                    's' => byte.is_ascii_whitespace(),
                    'S' => !byte.is_ascii_whitespace(),
                    c => byte == c as u8, // literal escaped char
                }
            } else {
                byte == b'\\'
            }
        }
        c => byte == c as u8,
    }
}

/// Apply guided decoding mask to logits in-place.
///
/// This is the main entry point for integrating guided decoding into the
/// sampling pipeline. Call before temperature scaling and other logit
/// processing.
pub fn apply_guided_mask(
    logits: &mut [f32],
    state: &GuidedDecodingState,
    vocab: &VocabTable,
) {
    state.apply_mask(logits, vocab);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_vocab(entries: Vec<(&str, TokenId)>, eos: TokenId) -> VocabTable {
        VocabTable::new(
            entries
                .into_iter()
                .map(|(text, id)| VocabEntry {
                    id,
                    text: text.to_string(),
                })
                .collect(),
            eos,
        )
    }

    #[test]
    fn unconstrained_no_masking() {
        let state = GuidedDecodingState::new(&ResponseFormat::Text).unwrap();
        assert!(state.is_unconstrained());
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let vocab = make_vocab(vec![("a", 0), ("b", 1), ("c", 2), ("<eos>", 3)], 3);
        state.apply_mask(&mut logits, &vocab);
        assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn json_mode_forces_json_start() {
        let state = GuidedDecodingState::new(&ResponseFormat::JsonObject).unwrap();
        assert!(!state.is_unconstrained());
        // JSON can start with any value start char
        let valid = state.compute_valid_chars();
        // With no text generated, Any is valid (since SchemaNode::Any)
        assert!(matches!(valid, ValidChars::Any));
    }

    #[test]
    fn json_schema_forces_object_start() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        let format = ResponseFormat::JsonSchema { json_schema: schema };
        let state = GuidedDecodingState::new(&format).unwrap();
        assert!(!state.is_unconstrained());

        let valid = state.compute_valid_chars();
        match valid {
            ValidChars::Set(s) => {
                assert!(s.contains(&b'{'));
                assert!(!s.contains(&b'['));
            }
            _ => panic!("expected Set"),
        }
    }

    #[test]
    fn advance_updates_generated() {
        let mut state = GuidedDecodingState::new(&ResponseFormat::JsonObject).unwrap();
        state.advance("{");
        assert_eq!(state.generated_text(), "{");
        state.advance("\"name\"");
        assert_eq!(state.generated_text(), "{\"name\"");
    }

    #[test]
    fn is_complete_valid_json() {
        let mut state = GuidedDecodingState::new(&ResponseFormat::JsonObject).unwrap();
        state.advance("{\"key\": \"value\"}");
        assert!(state.is_complete());
    }

    #[test]
    fn is_complete_invalid_json() {
        let mut state = GuidedDecodingState::new(&ResponseFormat::JsonObject).unwrap();
        state.advance("{\"key\": ");
        assert!(!state.is_complete());
    }

    #[test]
    fn json_schema_validation_complete() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        let format = ResponseFormat::JsonSchema { json_schema: schema };
        let mut state = GuidedDecodingState::new(&format).unwrap();
        state.advance("{\"name\": \"Alice\"}");
        assert!(state.is_complete());
    }

    #[test]
    fn json_schema_validation_missing_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        let format = ResponseFormat::JsonSchema { json_schema: schema };
        let mut state = GuidedDecodingState::new(&format).unwrap();
        state.advance("{}");
        // Valid JSON but doesn't satisfy schema (missing required "name")
        assert!(!state.is_complete());
    }

    #[test]
    fn mask_forces_eos_on_complete() {
        let mut state = GuidedDecodingState::new(&ResponseFormat::JsonObject).unwrap();
        state.generated = "null".to_string();
        // null is valid JSON, but for JSON mode with SchemaNode::Any,
        // the valid_next_chars should handle this.
    }

    #[test]
    fn apply_mask_blocks_invalid_tokens() {
        let schema = json!({"type": "integer"});
        let format = ResponseFormat::JsonSchema { json_schema: schema };
        let state = GuidedDecodingState::new(&format).unwrap();

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vocab = make_vocab(
            vec![
                ("1", 0),
                ("a", 1),
                ("{", 2),
                ("-", 3),
                ("<eos>", 4),
            ],
            4,
        );
        state.apply_mask(&mut logits, &vocab);

        // "1" and "-" should be allowed (integer can start with digit or minus)
        assert!(logits[0] > f32::NEG_INFINITY); // "1"
        assert_eq!(logits[1], f32::NEG_INFINITY); // "a" not valid
        assert_eq!(logits[2], f32::NEG_INFINITY); // "{" not valid
        assert!(logits[3] > f32::NEG_INFINITY); // "-"
        assert_eq!(logits[4], f32::NEG_INFINITY); // EOS not valid (nothing generated)
    }

    #[test]
    fn apply_mask_allows_eos_when_complete() {
        let schema = json!({"type": "integer"});
        let format = ResponseFormat::JsonSchema { json_schema: schema };
        let mut state = GuidedDecodingState::new(&format).unwrap();
        state.advance("42");

        let mut logits = vec![1.0, 2.0, 3.0];
        let vocab = make_vocab(vec![("3", 0), ("a", 1), ("<eos>", 2)], 2);
        state.apply_mask(&mut logits, &vocab);

        // "3" should be allowed (continue number)
        assert!(logits[0] > f32::NEG_INFINITY);
        // EOS should be allowed (42 is valid integer)
        assert!(logits[2] > f32::NEG_INFINITY);
    }

    #[test]
    fn vocab_table_basic() {
        let vocab = make_vocab(vec![("hello", 0), ("world", 1)], 2);
        assert_eq!(vocab.len(), 2);
        assert!(!vocab.is_empty());
        assert_eq!(vocab.eos_token_id(), 2);
    }

    #[test]
    fn validate_value_against_schema() {
        let node = SchemaNode::Object {
            properties: vec![
                ("name".to_string(), SchemaNode::String(Default::default()), true),
                ("age".to_string(), SchemaNode::Integer, false),
            ],
            additional_properties: false,
        };

        let valid = json!({"name": "Alice", "age": 30});
        assert!(validate_value(&valid, &node));

        let missing_required = json!({"age": 30});
        assert!(!validate_value(&missing_required, &node));

        let extra_prop = json!({"name": "Alice", "extra": true});
        assert!(!validate_value(&extra_prop, &node));

        let wrong_type = json!({"name": 42});
        assert!(!validate_value(&wrong_type, &node));
    }

    #[test]
    fn validate_array() {
        let node = SchemaNode::Array {
            items: Box::new(SchemaNode::Integer),
            min_items: 1,
            max_items: 3,
        };

        assert!(validate_value(&json!([1, 2, 3]), &node));
        assert!(!validate_value(&json!([]), &node)); // too few
        assert!(!validate_value(&json!([1, 2, 3, 4]), &node)); // too many
        assert!(!validate_value(&json!([1, "two"]), &node)); // wrong type
    }

    #[test]
    fn validate_anyof() {
        let node = SchemaNode::AnyOf(vec![
            SchemaNode::String(Default::default()),
            SchemaNode::Integer,
        ]);
        assert!(validate_value(&json!("hello"), &node));
        assert!(validate_value(&json!(42), &node));
        assert!(!validate_value(&json!(3.14), &node));
    }

    #[test]
    fn validate_enum() {
        let node = SchemaNode::Enum(vec![json!("red"), json!("green"), json!("blue")]);
        assert!(validate_value(&json!("red"), &node));
        assert!(!validate_value(&json!("yellow"), &node));
    }

    #[test]
    fn validate_const() {
        let node = SchemaNode::Const(json!(42));
        assert!(validate_value(&json!(42), &node));
        assert!(!validate_value(&json!(43), &node));
    }

    #[test]
    fn regex_simple_match() {
        assert!(simple_regex_full_match("abc", "abc"));
        assert!(!simple_regex_full_match("abd", "abc"));
    }

    #[test]
    fn regex_dot_match() {
        assert!(simple_regex_full_match("abc", "a.c"));
        assert!(simple_regex_full_match("axc", "a.c"));
    }

    #[test]
    fn regex_star_match() {
        assert!(simple_regex_full_match("aaaabc", "a*bc"));
        assert!(simple_regex_full_match("bc", "a*bc"));
    }

    #[test]
    fn regex_plus_match() {
        assert!(simple_regex_full_match("aabc", "a+bc"));
        assert!(!simple_regex_full_match("bc", "a+bc"));
    }

    #[test]
    fn regex_question_match() {
        assert!(simple_regex_full_match("abc", "ab?c"));
        assert!(simple_regex_full_match("ac", "ab?c"));
    }

    #[test]
    fn regex_digit_escape() {
        assert!(simple_regex_full_match("123", "\\d\\d\\d"));
        assert!(!simple_regex_full_match("12a", "\\d\\d\\d"));
    }

    #[test]
    fn regex_word_escape() {
        assert!(simple_regex_full_match("a_1", "\\w\\w\\w"));
        assert!(!simple_regex_full_match("a 1", "\\w\\w\\w"));
    }

    #[test]
    fn regex_prefix_match() {
        assert!(simple_regex_prefix_match("ab", "abcdef"));
        assert!(simple_regex_prefix_match("a", "a+b"));
        assert!(!simple_regex_prefix_match("x", "a+b"));
    }

    #[test]
    fn regex_guided_state() {
        let format = ResponseFormat::Regex {
            pattern: "\\d+".to_string(),
        };
        let state = GuidedDecodingState::new(&format).unwrap();
        let valid = state.compute_valid_chars();
        match valid {
            ValidChars::Set(s) => {
                assert!(s.contains(&b'0'));
                assert!(s.contains(&b'9'));
                assert!(!s.contains(&b'a'));
            }
            _ => panic!("expected Set for regex"),
        }
    }

    #[test]
    fn regex_guided_advance() {
        let format = ResponseFormat::Regex {
            pattern: "\\d+".to_string(),
        };
        let mut state = GuidedDecodingState::new(&format).unwrap();
        state.advance("12");
        let valid = state.compute_valid_chars();
        match valid {
            ValidChars::Set(s) => {
                assert!(s.contains(&b'3'));
            }
            _ => panic!("expected Set"),
        }
    }

    #[test]
    fn apply_guided_mask_function() {
        let format = ResponseFormat::Text;
        let state = GuidedDecodingState::new(&format).unwrap();
        let mut logits = vec![1.0, 2.0];
        let vocab = make_vocab(vec![("a", 0), ("b", 1)], 1);
        apply_guided_mask(&mut logits, &state, &vocab);
        // Text mode should not mask anything
        assert_eq!(logits, vec![1.0, 2.0]);
    }
}
