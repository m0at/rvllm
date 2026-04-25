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
        rvllm.lowering = "mosaic_custom_call"
      } {
    // TODO: lower body to Mosaic TPU dialect:
    // 1. load packed uint8 and FP8 E4M3 scales from HBM
    // 2. decode FP4 E2M1 nibbles and FP8 scales in VMEM/registers
    // 3. feed bf16 RHS tiles to TPU matmul
    // 4. write bf16 output tile to %out
    return
  }
}
