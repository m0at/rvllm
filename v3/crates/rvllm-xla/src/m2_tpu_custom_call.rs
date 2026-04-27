use rvllm_core::{ConfigError, Result, RvllmError};

pub const TPU_CUSTOM_CALL_TARGET: &str = "tpu_custom_call";
pub const TPU_MOSAIC_SERIALIZATION_FORMAT: u32 = 1;
pub const TPU_MOSAIC_BYTECODE_VERSION: u32 = 0;
pub const TPU_MOSAIC_SERDE_PASS: &str = "mosaic-serde{serialize=true target-version=3?}";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpuMosaicSerializedBody {
    body_b64: String,
    byte_len: usize,
}

impl TpuMosaicSerializedBody {
    pub fn from_serialized_bytecode(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(invalid("mosaic_body", "serialized bytecode is empty"));
        }
        let trimmed = bytes
            .iter()
            .copied()
            .skip_while(|b| b.is_ascii_whitespace())
            .take(8)
            .collect::<Vec<_>>();
        if trimmed.starts_with(b"module") {
            return Err(invalid(
                "mosaic_body",
                "got textual MLIR; expected Mosaic serde bytecode",
            ));
        }
        Ok(Self {
            body_b64: base64_encode(bytes),
            byte_len: bytes.len(),
        })
    }

    pub fn from_base64(body_b64: String, byte_len: usize) -> Result<Self> {
        if body_b64.is_empty() {
            return Err(invalid("mosaic_body", "base64 body is empty"));
        }
        Ok(Self { body_b64, byte_len })
    }

    pub fn body_b64(&self) -> &str {
        &self.body_b64
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

pub fn tpu_custom_call_backend_config(body_b64: &str) -> String {
    let json = format!(
        r#"{{"custom_call_config": {{"body": "{body_b64}", "serialization_format": {TPU_MOSAIC_SERIALIZATION_FORMAT}, "needs_layout_passes": true}}, "implicit_sharding": {{"type": "MANUAL"}}}}"#
    );
    mlir_string_escape(&json)
}

pub fn tpu_custom_call_backend_config_for_body(body: &TpuMosaicSerializedBody) -> String {
    tpu_custom_call_backend_config(body.body_b64())
}

pub fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn mlir_string_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'"' => out.push_str("\\22"),
            b'\\' => out.push_str("\\5C"),
            b'\n' => out.push_str("\\0A"),
            b'\r' => out.push_str("\\0D"),
            b'\t' => out.push_str("\\09"),
            0x20..=0x7e => out.push(b as char),
            _ => {
                out.push('\\');
                out.push(hex((b >> 4) & 0xf));
                out.push(hex(b & 0xf));
            }
        }
    }
    out
}

fn hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + (n - 10)) as char,
        _ => unreachable!(),
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_tpu_custom_call",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_config_matches_tpu_custom_call_json_contract() {
        let cfg = tpu_custom_call_backend_config("abcd");
        assert!(cfg.contains("\\22custom_call_config\\22"));
        assert!(cfg.contains("\\22body\\22: \\22abcd\\22"));
        assert!(cfg.contains("\\22serialization_format\\22: 1"));
        assert!(cfg.contains("\\22needs_layout_passes\\22: true"));
        assert!(cfg.contains("\\22implicit_sharding\\22"));
    }

    #[test]
    fn base64_encoder_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
    }

    #[test]
    fn serialized_body_rejects_textual_mlir() {
        assert!(TpuMosaicSerializedBody::from_serialized_bytecode(b"").is_err());
        assert!(TpuMosaicSerializedBody::from_serialized_bytecode(b"module {}").is_err());
        let body = TpuMosaicSerializedBody::from_serialized_bytecode(&[0x4d, 0x4c, 0xef, 0x52])
            .unwrap();
        assert_eq!(body.byte_len(), 4);
        assert_eq!(body.body_b64(), "TUzvUg==");
    }
}
