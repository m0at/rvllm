module attributes {rvllm.kind = "m2_prefill_scan"} {
  func.func @rvllm_m2_prefill_scan(
      %token_ids: memref<1x20xi32>,
      %positions: memref<20xi32>,
      %slot_mapping: memref<20xi32>,
      %cu_seqlens_q: memref<2xi32>,
      %context_lens: memref<1xi32>,
      %kv_cache: memref<260046848xi8>,
      %last_hidden: memref<1x3072xbf16>) attributes {
        rvllm.signature = "tokens,positions,slots,cu_seqlens,context_lens,kv_cache,last_hidden",
        rvllm.prefill = "single_compiled_scan",
        rvllm.batch = 1 : i64,
        rvllm.prompt_len = 20 : i64,
        rvllm.total_tokens = 20 : i64,
        rvllm.ctx = 2048 : i64,
        rvllm.num_layers = 62 : i64,
        rvllm.kv_shape = "layers=62,B=1,ctx=2048,kv_heads=8,head_dim=128,dtype=i8",
        rvllm.kv_cache_bytes = 260046848 : i64,
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "scan prompt positions once, write every K/V slot, return last prompt hidden for LM head"
      } {
    // Lowering outline:
    //   flat_tokens = reshape token_ids[B, T] -> [B*T]
    //   residual = embedding(flat_tokens)
    //   for layer in 0..num_layers:
    //     run M2 layer with num_tokens=B*T and phase=Prefill
    //     attention consumes cu_seqlens_q/context_lens and writes K/V using slot_mapping
    //   copy hidden rows at each sequence's final prompt position to last_hidden[B, H]
    return
  }
}
