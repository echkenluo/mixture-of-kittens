#include <cstdint>
#include <limits>

#include "sm90_fp8_block_terminal_tma_contract.cuh"

namespace contract = mok_sm90::fp8_block_terminal_tma_contract;

int main() {
    static_assert(contract::is_raw_bulk_aligned(0x1000u));
    static_assert(!contract::is_raw_bulk_aligned(0x1001u));
    static_assert(!contract::is_raw_bulk_aligned(0u));

    constexpr std::uintptr_t maximum =
        std::numeric_limits<std::uintptr_t>::max();
    static_assert(contract::valid_byte_interval(0x1000u, 0x400u));
    static_assert(!contract::valid_byte_interval(maximum - 0xFu, 0x10u));
    static_assert(!contract::valid_byte_interval(0x1000u, 0u));

    constexpr auto source = contract::make_byte_interval(0x1000u, 0x400u);
    constexpr auto adjacent = contract::make_byte_interval(0x1400u, 0x100u);
    constexpr auto partial = contract::make_byte_interval(0x1300u, 0x200u);
    constexpr auto contained = contract::make_byte_interval(0x1100u, 0x100u);
    constexpr auto containing = contract::make_byte_interval(0x0800u, 0x1000u);
    static_assert(!contract::byte_intervals_overlap(source, adjacent));
    static_assert(contract::byte_intervals_overlap(source, partial));
    static_assert(contract::byte_intervals_overlap(source, contained));
    static_assert(contract::byte_intervals_overlap(source, containing));
    return 0;
}
