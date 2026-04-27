module attributes {"stable_mosaic.version" = "1"} {
  func.func @main(
      %hidden: memref<8x3072xbf16>,
      %positions: memref<8xi32>,
      %kv_in: memref<33554432xi8>,
      %layer_offsets: memref<34xi32>,
      %expert_directory: memref<256x25xi32>,
      %w1_block_t: memref<3072x128xi8>,
      %hidden_out: memref<8x3072xbf16>,
      %kv_out: memref<33554432xi8>) attributes {
        dimension_semantics = [],
        scalar_prefetch = 0 : i64,
        scratch_operands = 0 : i64,
        rvllm.int8_probe = "w1_i8_full_k_128_cols"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c_batch = arith.constant 8 : index
    %c_hidden = arith.constant 3072 : index
    %hidden_v = vector.load %hidden[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    vector.store %hidden_v, %hidden_out[%c0, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    %kv_v = vector.load %kv_in[%c0] : memref<33554432xi8>, vector<512xi8>
    vector.store %kv_v, %kv_out[%c0] : memref<33554432xi8>, vector<512xi8>
    %zero = arith.constant dense<0.000000e+00> : vector<1x128xf32>
    scf.for %m = %c0 to %c_batch step %c1 {
      %acc = scf.for %k = %c0 to %c_hidden step %c32 iter_args(%acc_iter = %zero) -> (vector<1x128xf32>) {
        %h_bf16 = vector.load %hidden[%m, %k] : memref<8x3072xbf16>, vector<32xbf16>
        %h_f32 = arith.extf %h_bf16 : vector<32xbf16> to vector<32xf32>
        %h_mat = vector.broadcast %h_f32 : vector<32xf32> to vector<1x32xf32>
        %w_i8 = vector.load %w1_block_t[%k, %c0] : memref<3072x128xi8>, vector<32x128xi8>
        %w_f32 = arith.sitofp %w_i8 : vector<32x128xi8> to vector<32x128xf32>
        %next = vector.contract {
          indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (k, n)>,
            affine_map<(m, n, k) -> (m, n)>
          ],
          iterator_types = ["parallel", "parallel", "reduction"],
          kind = #vector.kind<add>
        } %h_mat, %w_f32, %acc_iter : vector<1x32xf32>, vector<32x128xf32> into vector<1x128xf32>
        scf.yield %next : vector<1x128xf32>
      }
      %out_bf16 = arith.truncf %acc : vector<1x128xf32> to vector<1x128xbf16>
      vector.store %out_bf16, %hidden_out[%m, %c0] : memref<8x3072xbf16>, vector<1x128xbf16>
    }
    return
  }
}
