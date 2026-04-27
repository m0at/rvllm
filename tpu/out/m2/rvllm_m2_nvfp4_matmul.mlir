module attributes {rvllm.kind = "m2_nvfp4_custom_call"} {
  func.func @rvllm_m2_nvfp4_matmul(
      %x: memref<8x3072xbf16>,
      %packed: memref<1536x1536xi8>,
      %scales: memref<1536x192xi8>,
      %global_scale: memref<f32>,
      %out: memref<8x1536xbf16>) attributes {
        rvllm.signature = "x_bf16,packed_u8,scale_fp8_e4m3,global_f32,out_bf16",
        rvllm.abi = "rvllm.m2.nvfp4.custom_call.v1",
        rvllm.abi_version = 1 : i64,
        rvllm.layout = "row_major",
        rvllm.block_size = 16 : i64,
        rvllm.tile = "BM=8,BN=512,BK=1024",
        rvllm.tiles = "M=1,N=3,K=3",
        rvllm.vmem_working_set_bytes = 1384448 : i64,
        rvllm.custom_call_target = "rvllm.m2.nvfp4_decode_bf16_matmul",
        rvllm.lowering = "rust_xla_custom_call",
        rvllm.lowering_plan = "XLA custom call receives bf16 activations, packed u8 NVFP4 weights, fp8_e4m3 scales, f32 global scale, and bf16 output buffer"
      } {
    "rvllm.m2.nvfp4_decode_bf16_matmul"(%x, %packed, %scales, %global_scale, %out) {
      rvllm.abi = "rvllm.m2.nvfp4.custom_call.v1",
      rvllm.abi_version = 1 : i64,
      rvllm.descriptor = "format=rvllm.m2.nvfp4.custom_call.v1;abi_version=1;target=rvllm.m2.nvfp4_decode_bf16_matmul;x_dtype=bf16;packed_dtype=u8;scale_dtype=fp8_e4m3;global_scale_dtype=f32;out_dtype=bf16;block_size=16;m=8;n=1536;k=3072;x_dims=8x3072;packed_dims=1536x1536;scale_dims=1536x192;out_dims=8x1536",
      rvllm.target = "rvllm.m2.nvfp4_decode_bf16_matmul",
      rvllm.x_dtype = "bf16",
      rvllm.packed_dtype = "u8",
      rvllm.scale_dtype = "fp8_e4m3",
      rvllm.global_scale_dtype = "f32",
      rvllm.out_dtype = "bf16",
      rvllm.block_size = 16 : i64,
      rvllm.x_dims = "8x3072",
      rvllm.packed_dims = "1536x1536",
      rvllm.scale_dims = "1536x192",
      rvllm.out_dims = "8x1536",
      rvllm.tile_bm = 8 : i64,
      rvllm.tile_bn = 512 : i64,
      rvllm.tile_bk = 1024 : i64,
      rvllm.tiles_m = 1 : i64,
      rvllm.tiles_n = 3 : i64,
      rvllm.tiles_k = 3 : i64,
      rvllm.vmem_working_set_bytes = 1384448 : i64,
      rvllm.packed_bytes = 2359296 : i64,
      rvllm.scale_bytes = 294912 : i64,
      rvllm.out_bytes = 24576 : i64
    } : (memref<8x3072xbf16>, memref<1536x1536xi8>, memref<1536x192xi8>, memref<f32>, memref<8x1536xbf16>) -> ()
    return
  }
}
