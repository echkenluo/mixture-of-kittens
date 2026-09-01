#include "pyutils/torchutils.cuh"
#include "sm90_fp8_block_tail_test.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(KITTENS_SM90)
    m.def("sm90_fp8_block_tail_test",
          &mok_sm90::fp8_block_tail_test::entry, "");
#endif
}
