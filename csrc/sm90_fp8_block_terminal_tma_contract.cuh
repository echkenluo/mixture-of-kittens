#pragma once

// Host/device-independent address contract for terminal raw bulk TMA.
// Keep this header free of CUDA and ATen so the negative interval/alignment
// cases can run on hosts that do not have nvcc or a GPU.

#include <cstddef>
#include <cstdint>
#include <limits>

namespace mok_sm90::fp8_block_terminal_tma_contract {

constexpr std::uintptr_t RAW_BULK_ALIGNMENT = 16u;

struct byte_interval {
    std::uintptr_t begin;
    std::uintptr_t end;
};

constexpr bool is_raw_bulk_aligned(std::uintptr_t pointer) {
    return pointer != 0u && pointer % RAW_BULK_ALIGNMENT == 0u;
}

constexpr bool valid_byte_interval(std::uintptr_t pointer,
                                   std::uint64_t bytes) {
    return pointer != 0u && bytes != 0u
        && bytes <= std::numeric_limits<std::uintptr_t>::max() - pointer;
}

constexpr byte_interval make_byte_interval(std::uintptr_t pointer,
                                           std::uint64_t bytes) {
    return {pointer, pointer + static_cast<std::uintptr_t>(bytes)};
}

constexpr bool byte_intervals_overlap(byte_interval lhs,
                                      byte_interval rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

}  // namespace mok_sm90::fp8_block_terminal_tma_contract
