module attributes {rvllm.kind = "m2_decode_graph"} {
  func.func @rvllm_m2_decode(
      %token_ids: memref<8xi32>,
      %positions: memref<8xi32>,
      %kv_cache: memref<2080374784xi8>,
      %weight_arena: memref<134412674816xi8>)
      -> (memref<8x200064xbf16>, memref<8xi32>, memref<2080374784xi8>)
      attributes {
        rvllm.signature = "token_ids,positions,kv_cache,weight_arena -> logits,next_token,kv_cache",
        rvllm.phase = "decode",
        rvllm.batch = 8 : i64,
        rvllm.ctx = 2048 : i64,
        rvllm.layers = 62 : i64,
        rvllm.hidden = 3072 : i64,
        rvllm.vocab = 200064 : i64,
        rvllm.kv_cache_bytes = 2080374784 : i64,
        rvllm.weight_arena_bytes = 134412674816 : i64,
        rvllm.weight_entries = 191069 : i64,
        rvllm.weight_alignment = 128 : i64,
        rvllm.weight_metadata = "compile_time_offsets_from_M2WeightArenaPlan",
        rvllm.lowering = "rust_mlir_custom_call",
        rvllm.lowering_plan = "embed -> 62 layer scan -> flat-arena dense loads -> flat-arena NVFP4 expert custom calls -> final norm -> lm_head -> argmax"
      } {
    // Contract body placeholder. The next slice replaces this with real
    // region bodies/custom-calls that consume offsets from M2WeightArenaPlan.
    // No Python/JAX graph emission belongs on this path.
    %logits = memref.alloc() : memref<8x200064xbf16>
    %next_token = memref.alloc() : memref<8xi32>
    return %logits, %next_token, %kv_cache : memref<8x200064xbf16>, memref<8xi32>, memref<2080374784xi8>
  }
}
