pub const TPU_CUSTOM_CALL_TARGET: &str = "tpu_custom_call";

pub fn tpu_custom_call_backend_config(body_b64: &str) -> String {
    let json = format!(
        r#"{{"custom_call_config": {{"body": "{body_b64}", "serialization_format": 1, "needs_layout_passes": true}}, "implicit_sharding": {{"type": "MANUAL"}}}}"#
    );
    mlir_string_escape(&json)
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
}
