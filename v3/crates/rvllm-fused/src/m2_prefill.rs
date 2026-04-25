use rvllm_core::{ConfigError, Result, RvllmError};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum M2PrefillKvDType {
    Bf16,
    Int8,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct M2PrefillScanShape {
    pub batch: usize,
    pub prompt_len: usize,
    pub hidden: usize,
    pub ctx: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dtype: M2PrefillKvDType,
}

impl M2PrefillScanShape {
    pub fn validate(self) -> Result<()> {
        if self.batch == 0
            || self.prompt_len == 0
            || self.hidden == 0
            || self.ctx == 0
            || self.num_layers == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(invalid("shape", "all dimensions must be > 0"));
        }
        if self.prompt_len > self.ctx {
            return Err(invalid("prompt_len", "must be <= ctx"));
        }
        Ok(())
    }

    pub const fn total_tokens(self) -> usize {
        self.batch * self.prompt_len
    }

    pub const fn kv_elem_bytes(self) -> usize {
        match self.kv_dtype {
            M2PrefillKvDType::Bf16 => 2,
            M2PrefillKvDType::Int8 => 1,
        }
    }

    pub const fn kv_cache_bytes(self) -> usize {
        2 * self.num_layers
            * self.batch
            * self.ctx
            * self.num_kv_heads
            * self.head_dim
            * self.kv_elem_bytes()
    }

    pub fn mlir(self, kernel_name: &str) -> Result<String> {
        self.validate()?;
        if !is_mlir_symbol(kernel_name) {
            return Err(invalid("kernel_name", "must be an MLIR symbol"));
        }
        let kv_dtype = match self.kv_dtype {
            M2PrefillKvDType::Bf16 => "bf16",
            M2PrefillKvDType::Int8 => "i8",
        };
        Ok(format!(
            r#"module attributes {{rvllm.kind = "m2_prefill_scan"}} {{
  func.func @{kernel_name}(
      %token_ids: memref<{batch}x{prompt_len}xi32>,
      %positions: memref<{tokens}xi32>,
      %slot_mapping: memref<{tokens}xi32>,
      %cu_seqlens_q: memref<{cu_len}xi32>,
      %context_lens: memref<{batch}xi32>,
      %kv_cache: memref<{kv_bytes}xi8>,
      %last_hidden: memref<{batch}x{hidden}xbf16>) attributes {{
        rvllm.signature = "tokens,positions,slots,cu_seqlens,context_lens,kv_cache,last_hidden",
        rvllm.prefill = "single_compiled_scan",
        rvllm.batch = {batch} : i64,
        rvllm.prompt_len = {prompt_len} : i64,
        rvllm.total_tokens = {tokens} : i64,
        rvllm.ctx = {ctx} : i64,
        rvllm.num_layers = {num_layers} : i64,
        rvllm.kv_shape = "layers={num_layers},B={batch},ctx={ctx},kv_heads={num_kv_heads},head_dim={head_dim},dtype={kv_dtype}",
        rvllm.kv_cache_bytes = {kv_bytes} : i64,
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "scan prompt positions once, write every K/V slot, return last prompt hidden for LM head"
      }} {{
    // Lowering outline:
    //   flat_tokens = reshape token_ids[B, T] -> [B*T]
    //   residual = embedding(flat_tokens)
    //   for layer in 0..num_layers:
    //     run M2 layer with num_tokens=B*T and phase=Prefill
    //     attention consumes cu_seqlens_q/context_lens and writes K/V using slot_mapping
    //   copy hidden rows at each sequence's final prompt position to last_hidden[B, H]
    return
  }}
}}
"#,
            kernel_name = kernel_name,
            batch = self.batch,
            prompt_len = self.prompt_len,
            tokens = self.total_tokens(),
            cu_len = self.batch + 1,
            hidden = self.hidden,
            ctx = self.ctx,
            num_layers = self.num_layers,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            kv_dtype = kv_dtype,
            kv_bytes = self.kv_cache_bytes(),
        ))
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_string(),
        },
        "m2_prefill",
    )
}

fn is_mlir_symbol(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_kv_cache_bytes_match_m2_b8_t128() {
        let shape = M2PrefillScanShape {
            batch: 8,
            prompt_len: 128,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        shape.validate().unwrap();
        assert_eq!(shape.total_tokens(), 1024);
        assert_eq!(shape.kv_elem_bytes(), 1);
        assert_eq!(shape.kv_cache_bytes(), 2_080_374_784);
    }

    #[test]
    fn emits_single_compiled_scan_contract() {
        let shape = M2PrefillScanShape {
            batch: 1,
            prompt_len: 20,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Int8,
        };
        let mlir = shape.mlir("rvllm_m2_prefill_scan").unwrap();
        assert!(mlir.contains("memref<1x20xi32>"));
        assert!(mlir.contains("rvllm.prefill = \"single_compiled_scan\""));
        assert!(mlir.contains("rvllm.total_tokens = 20 : i64"));
        assert!(mlir.contains("dtype=i8"));
        assert!(mlir.contains("write every K/V slot"));
    }

    #[test]
    fn rejects_prompt_longer_than_ctx() {
        let shape = M2PrefillScanShape {
            batch: 1,
            prompt_len: 4096,
            hidden: 3072,
            ctx: 2048,
            num_layers: 62,
            num_kv_heads: 8,
            head_dim: 128,
            kv_dtype: M2PrefillKvDType::Bf16,
        };
        assert!(shape.validate().is_err());
    }
}
