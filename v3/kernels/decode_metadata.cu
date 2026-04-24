extern "C" __global__ void decode_metadata_kernel(
    int* __restrict__ step,
    int* __restrict__ positions,
    int* __restrict__ slot_mapping,
    int* __restrict__ context_lens,
    int num_seqs,
    int max_blocks_per_seq,
    int block_size) {
    int tid = threadIdx.x;
    int s = step[0];
    if (tid < num_seqs) {
        positions[tid] = s;
        slot_mapping[tid] = s + tid * max_blocks_per_seq * block_size;
        context_lens[tid] = s + 1;
    }
    __syncthreads();
    if (tid == 0) {
        step[0] = s + 1;
    }
}
