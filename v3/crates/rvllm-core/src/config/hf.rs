//! HF `config.json` parsing helpers.
//!
//! Strict: no `serde(default)`, no `unwrap_or`. Missing field = named error.

use std::path::Path;

use crate::error::{ConfigError, Result, RvllmError};

pub(super) fn usize_field(
    v: &serde_json::Value,
    field: &'static str,
    file: &Path,
) -> Result<usize> {
    match v.get(field) {
        Some(x) if x.is_u64() => Ok(x.as_u64().unwrap_or(0) as usize),
        Some(_) => Err(RvllmError::config(
            ConfigError::HfTypeMismatch {
                name: field,
                expected: "non-negative integer",
            },
            field,
        )),
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

pub(super) fn f32_field(v: &serde_json::Value, field: &'static str, file: &Path) -> Result<f32> {
    match v.get(field).and_then(|x| x.as_f64()) {
        Some(x) => Ok(x as f32),
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

pub(super) fn bool_field_opt(v: &serde_json::Value, field: &'static str) -> Option<bool> {
    v.get(field).and_then(|x| x.as_bool())
}

/// String field supporting dotted paths like `architectures.0`.
pub(super) fn str_field(
    v: &serde_json::Value,
    dotted: &'static str,
    file: &Path,
) -> Result<String> {
    let mut cur = v;
    for part in dotted.split('.') {
        let next = if let Ok(idx) = part.parse::<usize>() {
            cur.get(idx)
        } else {
            cur.get(part)
        };
        cur = match next {
            Some(x) => x,
            None => {
                return Err(RvllmError::config(
                    ConfigError::MissingHfField {
                        name: dotted,
                        file: file.to_path_buf(),
                    },
                    dotted,
                ));
            }
        };
    }
    match cur.as_str() {
        Some(s) => Ok(s.to_string()),
        None => Err(RvllmError::config(
            ConfigError::HfTypeMismatch {
                name: dotted,
                expected: "string",
            },
            dotted,
        )),
    }
}
