#pragma once

// Compatibility facade for the production SM90 FP8 pipeline primitives.
// Existing K1/K2 callers keep this namespace while the implementation and
// layouts live in a production header that has no dependency on test code.
#if defined(KITTENS_SM90)

#include "sm90_fp8_block_pipeline_primitives.cuh"

namespace mok_sm90::fp8_block_gemm_core {

using fp8_block_pipeline::a_st;
using fp8_block_pipeline::acc_rt;
using fp8_block_pipeline::b_st;
using fp8_block_pipeline::d_st;
using fp8_block_pipeline::decode_tile;
using fp8_block_pipeline::PIPE_DEPTH;
using fp8_block_pipeline::run_tile;
using fp8_block_pipeline::tile;

} // namespace mok_sm90::fp8_block_gemm_core

#endif // defined(KITTENS_SM90)
