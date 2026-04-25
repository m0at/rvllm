module attributes {rvllm.kind = "m2_nvfp4_mosaic"} {
  func.func @rvllm_m2_nvfp4_matmul(
      %x: memref<8x3072xbf16>,
      %packed: memref<1536x1536xi8>,
      %scales: memref<1536x192xi8>,
      %global_scale: memref<f32>,
      %out: memref<8x1536xbf16>) attributes {
        rvllm.signature = "x_bf16,packed_u8,scale_fp8,global_f32,out_bf16",
        rvllm.layout = "row_major",
        rvllm.nvfp4_group = 16 : i64,
        rvllm.tile = "BM=8,BN=512,BK=1024",
        rvllm.tiles = "M=1,N=3,K=3",
        rvllm.vmem_working_set_bytes = 1384448 : i64,
        rvllm.custom_call_target = "rvllm.m2.nvfp4_decode_bf16_matmul",
        rvllm.lowering = "mosaic_custom_call",
        rvllm.lowering_plan = "tpu.load packed/scales -> unpack nibbles -> fp8/fp4 decode in VMEM/registers -> bf16 RHS tile -> tpu.matmul -> tpu.store"
      } {
    "rvllm.m2.nvfp4_decode_bf16_matmul"(%x, %packed, %scales, %global_scale, %out) {
      rvllm.tile_bm = 8 : i64,
      rvllm.tile_bn = 512 : i64,
      rvllm.tile_bk = 1024 : i64,
      rvllm.nvfp4_group = 16 : i64,
      rvllm.vmem_working_set_bytes = 1384448 : i64,
      rvllm.packed_bytes = 2359296 : i64,
      rvllm.scale_bytes = 294912 : i64,
      rvllm.out_bytes = 24576 : i64
    } : (memref<8x3072xbf16>, memref<1536x1536xi8>, memref<1536x192xi8>, memref<f32>, memref<8x1536xbf16>) -> ()
    return
  }
}
