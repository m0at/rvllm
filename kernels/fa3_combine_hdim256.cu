// Extra FA3 combine instantiations for head_dim=256.
// Upstream flash_fwd_combine.cu only instantiates 64 and 128.

#include "flash_fwd_combine_launch_template.h"

template void run_mha_fwd_combine_<cutlass::half_t, float, 256>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

template void run_mha_fwd_combine_<cutlass::bfloat16_t, float, 256>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
