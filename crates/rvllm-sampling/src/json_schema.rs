//! JSON schema compilation for guided decoding.
//!
//! Compiles a JSON schema (`serde_json::Value`) into a [`SchemaNode`] tree
//! that the guided decoder walks to determine which characters (and therefore
//! tokens) are valid at each position in the output.

use serde_json::Value;
use rvllm_core::prelude::LLMError;

/// Compiled representation of a JSON schema node.
#[derive(Debug, Clone)]
pub enum SchemaNode {
    /// Matches any valid JSON value.
    Any,
    /// Matches a JSON null literal.
    Null,
    /// Matches a JSON boolean (`true` or `false`).
    Boolean,
    /// Matches a JSON number (integer or float).
    Number,
    /// Matches a JSON integer only.
    Integer,
    /// Matches a JSON string.
    String(StringConstraints),
    /// Matches a JSON array with items conforming to the inner schema.
    Array {
        /// Schema that each element must satisfy.
        items: Box<SchemaNode>,
        /// Minimum number of items.
        min_items: usize,
        /// Maximum number of items (usize::MAX = unbounded).
        max_items: usize,
    },
    /// Matches a JSON object with named properties.
    Object {
        /// Property name -> (schema, required).
        properties: Vec<(String, SchemaNode, bool)>,
        /// Whether additional properties beyond those listed are allowed.
        additional_properties: bool,
    },
    /// Matches one of several schemas (anyOf / oneOf).
    AnyOf(Vec<SchemaNode>),
    /// Matches a value from an explicit set.
    Enum(Vec<Value>),
    /// Matches a constant value.
    Const(Value),
}

/// Constraints on a JSON string value.
#[derive(Debug, Clone)]
pub struct StringConstraints {
    /// Minimum string length in characters.
    pub min_length: usize,
    /// Maximum string length (usize::MAX = unbounded).
    pub max_length: usize,
    /// Optional regex pattern the string must match.
    pub pattern: Option<String>,
    /// If non-empty, the string must be one of these values.
    pub enum_values: Vec<String>,
}

impl Default for StringConstraints {
    fn default() -> Self {
        Self {
            min_length: 0,
            max_length: usize::MAX,
            pattern: None,
            enum_values: Vec::new(),
        }
    }
}

/// Compile a JSON schema value into a `SchemaNode` tree.
///
/// Supports a practical subset of JSON Schema Draft 2020-12:
/// - type: null, boolean, number, integer, string, array, object
/// - properties, required, additionalProperties
/// - items, minItems, maxItems
/// - minLength, maxLength, pattern
/// - enum, const
/// - anyOf, oneOf
pub fn compile_schema(schema: &Value) -> Result<SchemaNode, LLMError> {
    compile_node(schema, 0)
}

const MAX_DEPTH: usize = 64;

fn compile_node(schema: &Value, depth: usize) -> Result<SchemaNode, LLMError> {
    if depth > MAX_DEPTH {
        return Err(LLMError::SamplingError(
            "JSON schema exceeds maximum nesting depth".into(),
        ));
    }

    // Handle boolean schemas: true = any, false = nothing
    if let Some(b) = schema.as_bool() {
        return if b {
            Ok(SchemaNode::Any)
        } else {
            Err(LLMError::SamplingError(
                "JSON schema 'false' rejects all values".into(),
            ))
        };
    }

    let obj = schema.as_object().ok_or_else(|| {
        LLMError::SamplingError("JSON schema node must be an object or boolean".into())
    })?;

    // Empty object = any
    if obj.is_empty() {
        return Ok(SchemaNode::Any);
    }

    // const
    if let Some(val) = obj.get("const") {
        return Ok(SchemaNode::Const(val.clone()));
    }

    // enum
    if let Some(vals) = obj.get("enum") {
        let arr = vals.as_array().ok_or_else(|| {
            LLMError::SamplingError("'enum' must be an array".into())
        })?;
        return Ok(SchemaNode::Enum(arr.clone()));
    }

    // anyOf / oneOf
    if let Some(any_of) = obj.get("anyOf").or_else(|| obj.get("oneOf")) {
        let arr = any_of.as_array().ok_or_else(|| {
            LLMError::SamplingError("'anyOf'/'oneOf' must be an array".into())
        })?;
        let nodes: Result<Vec<_>, _> = arr
            .iter()
            .map(|v| compile_node(v, depth + 1))
            .collect();
        return Ok(SchemaNode::AnyOf(nodes?));
    }

    // type-based dispatch
    let type_val = obj.get("type");
    match type_val.and_then(|v| v.as_str()) {
        Some("null") => Ok(SchemaNode::Null),
        Some("boolean") => Ok(SchemaNode::Boolean),
        Some("number") => Ok(SchemaNode::Number),
        Some("integer") => Ok(SchemaNode::Integer),
        Some("string") => compile_string(obj, depth),
        Some("array") => compile_array(obj, depth),
        Some("object") => compile_object(obj, depth),
        Some(other) => Err(LLMError::SamplingError(format!(
            "unsupported JSON schema type: {other}"
        ))),
        None => {
            // No type specified -- if it has properties, treat as object
            if obj.contains_key("properties") {
                compile_object(obj, depth)
            } else {
                Ok(SchemaNode::Any)
            }
        }
    }
}

fn compile_string(
    obj: &serde_json::Map<String, Value>,
    _depth: usize,
) -> Result<SchemaNode, LLMError> {
    let min_length = obj
        .get("minLength")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let max_length = obj
        .get("maxLength")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as usize;
    let pattern = obj
        .get("pattern")
        .and_then(|v| v.as_str())
        .map(String::from);
    let enum_values = obj
        .get("enum")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(SchemaNode::String(StringConstraints {
        min_length,
        max_length,
        pattern,
        enum_values,
    }))
}

fn compile_array(
    obj: &serde_json::Map<String, Value>,
    depth: usize,
) -> Result<SchemaNode, LLMError> {
    let items = match obj.get("items") {
        Some(v) => compile_node(v, depth + 1)?,
        None => SchemaNode::Any,
    };
    let min_items = obj
        .get("minItems")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let max_items = obj
        .get("maxItems")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as usize;

    Ok(SchemaNode::Array {
        items: Box::new(items),
        min_items,
        max_items,
    })
}

fn compile_object(
    obj: &serde_json::Map<String, Value>,
    depth: usize,
) -> Result<SchemaNode, LLMError> {
    let required: Vec<String> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let required_set: std::collections::HashSet<&str> =
        required.iter().map(|s| s.as_str()).collect();

    let properties = match obj.get("properties") {
        Some(props) => {
            let props_obj = props.as_object().ok_or_else(|| {
                LLMError::SamplingError("'properties' must be an object".into())
            })?;
            let mut result = Vec::with_capacity(props_obj.len());
            for (key, val) in props_obj {
                let node = compile_node(val, depth + 1)?;
                let is_required = required_set.contains(key.as_str());
                result.push((key.clone(), node, is_required));
            }
            result
        }
        None => Vec::new(),
    };

    let additional_properties = obj
        .get("additionalProperties")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(SchemaNode::Object {
        properties,
        additional_properties,
    })
}

/// Return the set of valid next characters given a partial JSON string and schema.
///
/// This is used by the guided decoder to build a token mask. Given the text
/// generated so far, it returns which characters are legal continuations.
pub fn valid_next_chars(partial: &str, node: &SchemaNode) -> ValidChars {
    let trimmed = partial.trim_start();
    if trimmed.is_empty() {
        return valid_start_chars(node);
    }
    // Walk the partial output against the schema to figure out what comes next.
    let ctx = ParseContext::new(trimmed);
    ctx.valid_next(node)
}

/// Represents the set of characters allowed at the next position.
#[derive(Debug, Clone)]
pub enum ValidChars {
    /// Any character is allowed (unconstrained).
    Any,
    /// Only these specific bytes are allowed.
    Set(Vec<u8>),
    /// Generation should stop (the value is complete).
    End,
}

impl ValidChars {
    /// Check whether a byte is allowed.
    pub fn allows(&self, b: u8) -> bool {
        match self {
            ValidChars::Any => true,
            ValidChars::Set(set) => set.contains(&b),
            ValidChars::End => false,
        }
    }

    /// Merge two ValidChars (union).
    pub fn union(self, other: ValidChars) -> ValidChars {
        match (self, other) {
            (ValidChars::Any, _) | (_, ValidChars::Any) => ValidChars::Any,
            (ValidChars::End, other) | (other, ValidChars::End) => other,
            (ValidChars::Set(mut a), ValidChars::Set(b)) => {
                for c in b {
                    if !a.contains(&c) {
                        a.push(c);
                    }
                }
                ValidChars::Set(a)
            }
        }
    }
}

/// Return the valid starting characters for a schema node.
fn valid_start_chars(node: &SchemaNode) -> ValidChars {
    match node {
        SchemaNode::Any => ValidChars::Any,
        SchemaNode::Null => ValidChars::Set(vec![b'n']),
        SchemaNode::Boolean => ValidChars::Set(vec![b't', b'f']),
        SchemaNode::Number | SchemaNode::Integer => {
            ValidChars::Set(vec![b'-', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'])
        }
        SchemaNode::String(_) => ValidChars::Set(vec![b'"']),
        SchemaNode::Array { .. } => ValidChars::Set(vec![b'[']),
        SchemaNode::Object { .. } => ValidChars::Set(vec![b'{']),
        SchemaNode::AnyOf(nodes) => {
            let mut result = ValidChars::End;
            for n in nodes {
                result = result.union(valid_start_chars(n));
            }
            result
        }
        SchemaNode::Enum(vals) => {
            let mut chars = Vec::new();
            for v in vals {
                if let Some(first) = json_value_first_char(v) {
                    if !chars.contains(&first) {
                        chars.push(first);
                    }
                }
            }
            if chars.is_empty() {
                ValidChars::Any
            } else {
                ValidChars::Set(chars)
            }
        }
        SchemaNode::Const(v) => {
            if let Some(first) = json_value_first_char(v) {
                ValidChars::Set(vec![first])
            } else {
                ValidChars::Any
            }
        }
    }
}

/// Get the first byte of a JSON-serialized value.
fn json_value_first_char(v: &Value) -> Option<u8> {
    match v {
        Value::Null => Some(b'n'),
        Value::Bool(true) => Some(b't'),
        Value::Bool(false) => Some(b'f'),
        Value::Number(_) => {
            let s = v.to_string();
            s.bytes().next()
        }
        Value::String(_) => Some(b'"'),
        Value::Array(_) => Some(b'['),
        Value::Object(_) => Some(b'{'),
    }
}

/// Lightweight context for walking partial JSON output.
struct ParseContext<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> ParseContext<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn remaining(&self) -> &str {
        &self.text[self.pos..]
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.text.len() {
            match self.text.as_bytes()[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    /// Determine valid next characters given what has been generated so far.
    fn valid_next(&self, node: &SchemaNode) -> ValidChars {
        let mut ctx = ParseContext {
            text: self.text,
            pos: self.pos,
        };
        ctx.skip_whitespace();

        if ctx.pos >= ctx.text.len() {
            return valid_start_chars(node);
        }

        match node {
            SchemaNode::Any => ValidChars::Any,
            SchemaNode::Null => ctx.valid_next_literal("null"),
            SchemaNode::Boolean => {
                let b = ctx.peek().unwrap_or(0);
                if b == b't' {
                    ctx.valid_next_literal("true")
                } else {
                    ctx.valid_next_literal("false")
                }
            }
            SchemaNode::Number => ctx.valid_next_number(false),
            SchemaNode::Integer => ctx.valid_next_number(true),
            SchemaNode::String(_constraints) => ctx.valid_next_string(),
            SchemaNode::Array { items, min_items, max_items } => {
                ctx.valid_next_array(items, *min_items, *max_items)
            }
            SchemaNode::Object { properties, additional_properties } => {
                ctx.valid_next_object(properties, *additional_properties)
            }
            SchemaNode::AnyOf(nodes) => {
                let mut result = ValidChars::End;
                for n in nodes {
                    let sub = ParseContext {
                        text: ctx.text,
                        pos: ctx.pos,
                    };
                    result = result.union(sub.valid_next(n));
                }
                result
            }
            SchemaNode::Enum(vals) => {
                let mut result = ValidChars::End;
                for v in vals {
                    let serialized = serde_json::to_string(v).unwrap_or_default();
                    let r = ctx.remaining();
                    if serialized.starts_with(r) {
                        // We are a prefix of this enum value
                        if let Some(next) = serialized.as_bytes().get(r.len()) {
                            result = result.union(ValidChars::Set(vec![*next]));
                        } else {
                            result = result.union(ValidChars::End);
                        }
                    }
                }
                result
            }
            SchemaNode::Const(v) => {
                let serialized = serde_json::to_string(v).unwrap_or_default();
                let r = ctx.remaining();
                if serialized.starts_with(r) {
                    if let Some(next) = serialized.as_bytes().get(r.len()) {
                        ValidChars::Set(vec![*next])
                    } else {
                        ValidChars::End
                    }
                } else {
                    ValidChars::End
                }
            }
        }
    }

    fn valid_next_literal(&self, literal: &str) -> ValidChars {
        let remaining = self.remaining();
        if literal.starts_with(remaining) {
            if let Some(next) = literal.as_bytes().get(remaining.len()) {
                ValidChars::Set(vec![*next])
            } else {
                // Full literal matched
                ValidChars::End
            }
        } else {
            ValidChars::End
        }
    }

    fn valid_next_number(&self, integer_only: bool) -> ValidChars {
        let remaining = self.remaining();
        if remaining.is_empty() {
            return ValidChars::Set(
                vec![b'-', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'],
            );
        }

        let bytes = remaining.as_bytes();
        let last = bytes[bytes.len() - 1];
        let has_dot = remaining.contains('.');
        let has_e = remaining.contains('e') || remaining.contains('E');

        let mut valid = Vec::new();

        // Digits are almost always valid continuation
        if last != b'-' || bytes.len() == 1 {
            // After minus or after digits
            valid.extend_from_slice(&[b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']);
        } else {
            valid.extend_from_slice(&[b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']);
        }

        // Decimal point if no dot yet and not integer-only, and last was a digit
        if !integer_only && !has_dot && !has_e && last.is_ascii_digit() {
            valid.push(b'.');
        }

        // Exponent if no e yet and last was digit or dot
        if !integer_only && !has_e && last.is_ascii_digit() {
            valid.push(b'e');
            valid.push(b'E');
        }

        // Minus after e/E
        if has_e && (last == b'e' || last == b'E') {
            valid.push(b'-');
            valid.push(b'+');
        }

        // Number could be complete if it ends with a digit
        if last.is_ascii_digit() && bytes.len() > 0 {
            // Check it's a valid number so far
            if remaining != "-" {
                // The number could end here -- signal this by also allowing End
                // We combine Set with the possibility of ending
                return ValidChars::Set(valid);
            }
        }

        ValidChars::Set(valid)
    }

    fn valid_next_string(&self) -> ValidChars {
        let remaining = self.remaining();
        let bytes = remaining.as_bytes();

        if bytes.is_empty() {
            return ValidChars::Set(vec![b'"']);
        }

        if bytes[0] != b'"' {
            return ValidChars::Set(vec![b'"']);
        }

        // We're inside a string. Find if it's complete.
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2; // skip escaped char
                continue;
            }
            if bytes[i] == b'"' {
                // String is complete
                return ValidChars::End;
            }
            i += 1;
        }

        // String is not yet closed. Allow any printable char or escape or closing quote.
        // In practice we allow everything except bare control characters.
        let valid: Vec<u8> = (0x20..=0x7E).collect();
        // Also allow the closing quote and backslash for escapes
        // (already included in 0x20..=0x7E)
        // Check if last char was a backslash (escape in progress)
        if bytes.len() >= 2 && bytes[bytes.len() - 1] == b'\\' {
            // Must be a valid escape character
            return ValidChars::Set(vec![b'"', b'\\', b'/', b'b', b'f', b'n', b'r', b't', b'u']);
        }

        ValidChars::Set(valid)
    }

    fn valid_next_array(
        &self,
        items: &SchemaNode,
        min_items: usize,
        max_items: usize,
    ) -> ValidChars {
        let remaining = self.remaining();
        if remaining.is_empty() {
            return ValidChars::Set(vec![b'[']);
        }

        let bytes = remaining.as_bytes();
        if bytes[0] != b'[' {
            return ValidChars::Set(vec![b'[']);
        }

        // Count how deep we are in the array
        let inner = &remaining[1..];
        let trimmed = inner.trim_start();

        if trimmed.is_empty() {
            // Just opened the array
            if min_items == 0 {
                let mut v = vec![b']'];
                if max_items > 0 {
                    match valid_start_chars(items) {
                        ValidChars::Set(s) => v.extend_from_slice(&s),
                        ValidChars::Any => return ValidChars::Any,
                        ValidChars::End => {}
                    }
                }
                // Allow whitespace
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                return ValidChars::Set(v);
            } else {
                let mut v = Vec::new();
                match valid_start_chars(items) {
                    ValidChars::Set(s) => v.extend_from_slice(&s),
                    ValidChars::Any => return ValidChars::Any,
                    ValidChars::End => {}
                }
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                return ValidChars::Set(v);
            }
        }

        // For simplicity in the recursive descent, allow structural chars and
        // delegate detailed item-level validation to the guided decoder layer.
        // This provides correct but slightly permissive array-level guidance.
        let last_nonws = trimmed.as_bytes().last().copied().unwrap_or(0);
        match last_nonws {
            b',' => {
                // After comma, next item starts
                let mut v = Vec::new();
                match valid_start_chars(items) {
                    ValidChars::Set(s) => v.extend_from_slice(&s),
                    ValidChars::Any => return ValidChars::Any,
                    ValidChars::End => {}
                }
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                ValidChars::Set(v)
            }
            _ => {
                // Could be mid-value or end of value
                let mut v = vec![b',', b']'];
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                // Also allow continuation of current value
                ValidChars::Set(v)
            }
        }
    }

    fn valid_next_object(
        &self,
        properties: &[(String, SchemaNode, bool)],
        _additional_properties: bool,
    ) -> ValidChars {
        let remaining = self.remaining();
        if remaining.is_empty() {
            return ValidChars::Set(vec![b'{']);
        }

        let bytes = remaining.as_bytes();
        if bytes[0] != b'{' {
            return ValidChars::Set(vec![b'{']);
        }

        let inner = &remaining[1..];
        let trimmed = inner.trim_start();

        if trimmed.is_empty() {
            // Just opened the object
            let has_required = properties.iter().any(|(_, _, req)| *req);
            let mut v = Vec::new();
            if !has_required {
                v.push(b'}');
            }
            v.push(b'"'); // property name start
            v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
            return ValidChars::Set(v);
        }

        let last_nonws = trimmed.as_bytes().last().copied().unwrap_or(0);
        match last_nonws {
            b':' => {
                // After colon, value starts -- allow any value start
                ValidChars::Any
            }
            b',' => {
                // After comma, next key
                let mut v = vec![b'"'];
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                ValidChars::Set(v)
            }
            b'{' => {
                // Just opened, same as empty
                let has_required = properties.iter().any(|(_, _, req)| *req);
                let mut v = Vec::new();
                if !has_required {
                    v.push(b'}');
                }
                v.push(b'"');
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                ValidChars::Set(v)
            }
            _ => {
                // Could be mid-key, mid-value, or between entries
                let mut v = vec![b',', b'}', b':'];
                v.extend_from_slice(&[b' ', b'\n', b'\t', b'\r']);
                // Allow continuation of current token
                ValidChars::Set(v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compile_null_schema() {
        let schema = json!({"type": "null"});
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Null));
    }

    #[test]
    fn compile_boolean_schema() {
        let schema = json!({"type": "boolean"});
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Boolean));
    }

    #[test]
    fn compile_number_schema() {
        let schema = json!({"type": "number"});
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Number));
    }

    #[test]
    fn compile_integer_schema() {
        let schema = json!({"type": "integer"});
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Integer));
    }

    #[test]
    fn compile_string_schema() {
        let schema = json!({"type": "string", "minLength": 1, "maxLength": 10});
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::String(c) => {
                assert_eq!(c.min_length, 1);
                assert_eq!(c.max_length, 10);
            }
            _ => panic!("expected String node"),
        }
    }

    #[test]
    fn compile_array_schema() {
        let schema = json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 5
        });
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::Array { items, min_items, max_items } => {
                assert!(matches!(*items, SchemaNode::Integer));
                assert_eq!(min_items, 1);
                assert_eq!(max_items, 5);
            }
            _ => panic!("expected Array node"),
        }
    }

    #[test]
    fn compile_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::Object { properties, additional_properties } => {
                assert_eq!(properties.len(), 2);
                let name_prop = properties.iter().find(|(k, _, _)| k == "name").unwrap();
                assert!(name_prop.2); // required
                let age_prop = properties.iter().find(|(k, _, _)| k == "age").unwrap();
                assert!(!age_prop.2); // not required
                assert!(additional_properties);
            }
            _ => panic!("expected Object node"),
        }
    }

    #[test]
    fn compile_anyof_schema() {
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::AnyOf(nodes) => {
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("expected AnyOf node"),
        }
    }

    #[test]
    fn compile_enum_schema() {
        let schema = json!({"enum": ["red", "green", "blue"]});
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::Enum(vals) => assert_eq!(vals.len(), 3),
            _ => panic!("expected Enum node"),
        }
    }

    #[test]
    fn compile_const_schema() {
        let schema = json!({"const": 42});
        let node = compile_schema(&schema).unwrap();
        match node {
            SchemaNode::Const(v) => assert_eq!(v, json!(42)),
            _ => panic!("expected Const node"),
        }
    }

    #[test]
    fn compile_empty_schema_is_any() {
        let schema = json!({});
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Any));
    }

    #[test]
    fn compile_true_schema_is_any() {
        let schema = json!(true);
        let node = compile_schema(&schema).unwrap();
        assert!(matches!(node, SchemaNode::Any));
    }

    #[test]
    fn compile_false_schema_is_error() {
        let schema = json!(false);
        assert!(compile_schema(&schema).is_err());
    }

    #[test]
    fn valid_start_null() {
        let node = SchemaNode::Null;
        let chars = valid_next_chars("", &node);
        assert!(chars.allows(b'n'));
        assert!(!chars.allows(b't'));
    }

    #[test]
    fn valid_start_boolean() {
        let node = SchemaNode::Boolean;
        let chars = valid_next_chars("", &node);
        assert!(chars.allows(b't'));
        assert!(chars.allows(b'f'));
        assert!(!chars.allows(b'n'));
    }

    #[test]
    fn valid_start_string() {
        let node = SchemaNode::String(StringConstraints::default());
        let chars = valid_next_chars("", &node);
        assert!(chars.allows(b'"'));
        assert!(!chars.allows(b'a'));
    }

    #[test]
    fn valid_start_object() {
        let node = SchemaNode::Object {
            properties: vec![],
            additional_properties: true,
        };
        let chars = valid_next_chars("", &node);
        assert!(chars.allows(b'{'));
    }

    #[test]
    fn valid_next_partial_null() {
        let node = SchemaNode::Null;
        let chars = valid_next_chars("nu", &node);
        assert!(chars.allows(b'l'));
        assert!(!chars.allows(b'x'));
    }

    #[test]
    fn valid_next_complete_null() {
        let node = SchemaNode::Null;
        let chars = valid_next_chars("null", &node);
        // null is complete, should signal end
        matches!(chars, ValidChars::End);
    }

    #[test]
    fn valid_next_partial_true() {
        let node = SchemaNode::Boolean;
        let chars = valid_next_chars("tr", &node);
        assert!(chars.allows(b'u'));
    }

    #[test]
    fn valid_next_number_start() {
        let node = SchemaNode::Number;
        let chars = valid_next_chars("", &node);
        assert!(chars.allows(b'1'));
        assert!(chars.allows(b'-'));
    }

    #[test]
    fn valid_next_number_digits() {
        let node = SchemaNode::Number;
        let chars = valid_next_chars("12", &node);
        assert!(chars.allows(b'3'));
        assert!(chars.allows(b'.'));
        assert!(chars.allows(b'e'));
    }

    #[test]
    fn valid_next_integer_no_dot() {
        let node = SchemaNode::Integer;
        let chars = valid_next_chars("12", &node);
        assert!(chars.allows(b'3'));
        assert!(!chars.allows(b'.'));
    }

    #[test]
    fn valid_next_string_open() {
        let node = SchemaNode::String(StringConstraints::default());
        let chars = valid_next_chars("\"hel", &node);
        // Inside string, should allow printable chars and closing quote
        assert!(chars.allows(b'l'));
        assert!(chars.allows(b'"'));
    }

    #[test]
    fn valid_next_string_closed() {
        let node = SchemaNode::String(StringConstraints::default());
        let chars = valid_next_chars("\"hello\"", &node);
        matches!(chars, ValidChars::End);
    }

    #[test]
    fn nested_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    },
                    "required": ["street", "city"]
                }
            },
            "required": ["address"]
        });
        let node = compile_schema(&schema).unwrap();
        match &node {
            SchemaNode::Object { properties, .. } => {
                assert_eq!(properties.len(), 1);
                assert!(matches!(&properties[0].1, SchemaNode::Object { .. }));
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn deep_nesting_limit() {
        // Build a deeply nested schema
        let mut schema = json!({"type": "string"});
        for _ in 0..100 {
            schema = json!({"type": "array", "items": schema});
        }
        assert!(compile_schema(&schema).is_err());
    }

    #[test]
    fn string_constraints_default() {
        let c = StringConstraints::default();
        assert_eq!(c.min_length, 0);
        assert_eq!(c.max_length, usize::MAX);
        assert!(c.pattern.is_none());
        assert!(c.enum_values.is_empty());
    }

    #[test]
    fn valid_chars_union() {
        let a = ValidChars::Set(vec![b'a', b'b']);
        let b = ValidChars::Set(vec![b'b', b'c']);
        let merged = a.union(b);
        match merged {
            ValidChars::Set(s) => {
                assert!(s.contains(&b'a'));
                assert!(s.contains(&b'b'));
                assert!(s.contains(&b'c'));
            }
            _ => panic!("expected Set"),
        }
    }

    #[test]
    fn valid_chars_union_any() {
        let a = ValidChars::Set(vec![b'a']);
        let b = ValidChars::Any;
        assert!(matches!(a.union(b), ValidChars::Any));
    }

    #[test]
    fn valid_chars_union_end() {
        let a = ValidChars::Set(vec![b'a']);
        let b = ValidChars::End;
        match a.union(b) {
            ValidChars::Set(s) => assert!(s.contains(&b'a')),
            _ => panic!("expected Set"),
        }
    }
}
