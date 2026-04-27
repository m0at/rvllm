#w1_dot_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}
module attributes {"stable_mosaic.version" = "1"} {
  func.func @main(
      %hidden: memref<8x3072xbf16>,
      %positions: memref<8xi32>,
      %kv_in: memref<33554432xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %w1_block_t: memref<3072x128xi8>,
      %w1_row_scales: memref<128xf32>,
      %hidden_out: memref<8x3072xbf16>,
      %kv_out: memref<33554432xi8>) attributes {
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_observable_unrolled_bk32x4"
      } {
    %c0 = arith.constant 0 : index
    %ck_1 = arith.constant 32 : index
    %ck_2 = arith.constant 64 : index
    %ck_3 = arith.constant 96 : index
    %scale_v = vector.load %w1_row_scales[%c0] : memref<128xf32>, vector<128xf32>
    %scale_2d = vector.shape_cast %scale_v : vector<128xf32> to vector<1x128xf32>
    %scale_b = vector.broadcast %scale_2d : vector<1x128xf32> to vector<32x128xf32>
    %acc_init = arith.constant dense<0.000000e+00> : vector<8x128xf32>
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<8x3072xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<8x3072xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<33554432xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<33554432xi8>, vector<512xi8>
    %w_i8_0 = vector.load %w1_block_t[%c0, %c0] : memref<3072x128xi8>, vector<32x128xi8>
    %w_f32_0 = arith.sitofp %w_i8_0 : vector<32x128xi8> to vector<32x128xf32>
    %w_scaled_0 = arith.mulf %w_f32_0, %scale_b : vector<32x128xf32>
    %h_bf16_0 = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<8x32xbf16>
    %h_f32_0 = arith.extf %h_bf16_0 : vector<8x32xbf16> to vector<8x32xf32>
    %acc_0 = vector.contract #w1_dot_trait %h_f32_0, %w_scaled_0, %acc_init : vector<8x32xf32>, vector<32x128xf32> into vector<8x128xf32>
    %w_i8_1 = vector.load %w1_block_t[%ck_1, %c0] : memref<3072x128xi8>, vector<32x128xi8>
    %w_f32_1 = arith.sitofp %w_i8_1 : vector<32x128xi8> to vector<32x128xf32>
    %w_scaled_1 = arith.mulf %w_f32_1, %scale_b : vector<32x128xf32>
    %h_bf16_1 = vector.load %hidden[%c0, %ck_1] : memref<8x3072xbf16>, vector<8x32xbf16>
    %h_f32_1 = arith.extf %h_bf16_1 : vector<8x32xbf16> to vector<8x32xf32>
    %acc_1 = vector.contract #w1_dot_trait %h_f32_1, %w_scaled_1, %acc_0 : vector<8x32xf32>, vector<32x128xf32> into vector<8x128xf32>
    %w_i8_2 = vector.load %w1_block_t[%ck_2, %c0] : memref<3072x128xi8>, vector<32x128xi8>
    %w_f32_2 = arith.sitofp %w_i8_2 : vector<32x128xi8> to vector<32x128xf32>
    %w_scaled_2 = arith.mulf %w_f32_2, %scale_b : vector<32x128xf32>
    %h_bf16_2 = vector.load %hidden[%c0, %ck_2] : memref<8x3072xbf16>, vector<8x32xbf16>
    %h_f32_2 = arith.extf %h_bf16_2 : vector<8x32xbf16> to vector<8x32xf32>
    %acc_2 = vector.contract #w1_dot_trait %h_f32_2, %w_scaled_2, %acc_1 : vector<8x32xf32>, vector<32x128xf32> into vector<8x128xf32>
    %w_i8_3 = vector.load %w1_block_t[%ck_3, %c0] : memref<3072x128xi8>, vector<32x128xi8>
    %w_f32_3 = arith.sitofp %w_i8_3 : vector<32x128xi8> to vector<32x128xf32>
    %w_scaled_3 = arith.mulf %w_f32_3, %scale_b : vector<32x128xf32>
    %h_bf16_3 = vector.load %hidden[%c0, %ck_3] : memref<8x3072xbf16>, vector<8x32xbf16>
    %h_f32_3 = arith.extf %h_bf16_3 : vector<8x32xbf16> to vector<8x32xf32>
    %acc_3 = vector.contract #w1_dot_trait %h_f32_3, %w_scaled_3, %acc_2 : vector<8x32xf32>, vector<32x128xf32> into vector<8x128xf32>
    %out_bf16 = arith.truncf %acc_3 : vector<8x128xf32> to vector<8x128xbf16>
    vector.store %out_bf16, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<8x128xbf16>
    return
  }
}
