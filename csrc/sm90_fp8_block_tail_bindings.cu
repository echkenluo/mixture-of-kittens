#include "pyutils/torchutils.cuh"
#include "sm90_fp8_block_worker_test.cuh"
#include "sm90_fp8_block_tail_test.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(KITTENS_SM90)
    m.def("sm90_fp8_block_tail_test",
          &mok_sm90::fp8_block_tail_test::entry, "");
    m.def("sm90_fp8_block_tail_out_test",
          &mok_sm90::fp8_block_tail_test::entry_out, "");
    m.def("sm90_fp8_block_grouped_pipelined_out_test",
          &mok_sm90::fp8_block_test::grouped::entry_pipelined_out, "");
#endif
}
