#include <cassert>
#include <cstdio>
#include <vector>
#include "../csrc/sm90_fp8_block_warprole_config.cuh"
using namespace mok_sm90::warprole;

// Every idx in [0, total_tasks) must map to a distinct (kind, m_tile, n_index)
// triple, and the whole product space must be covered exactly once.
template <int NC>
static void check_coverage(shape s, int expected_tiles) {
    using G = geometry<NC>;
    const int stride = 32;                            // max n_index count (NC == 1 W2 case)
    const int kind_span = expected_tiles * stride;
    std::vector<int> hist(2 * kind_span, 0);
    int64_t seen_w13 = 0, seen_w2 = 0;
    const int64_t total = total_tasks<NC>(s);
    for (int64_t idx = 0; idx < total; ++idx) {
        const task t = decode_task<NC>(idx, s);
        assert(t.kind != task_kind::none);
        assert(t.m_tile >= 0 && t.m_tile < expected_tiles);
        assert(t.minibatch == t.m_tile / MINIBATCH_TILES);
        const int limit = (t.kind == task_kind::w13) ? G::W13_TASKS_PER_TILE : G::W2_TASKS_PER_TILE;
        assert(t.n_index >= 0 && t.n_index < limit);
        if (t.kind == task_kind::w13) ++seen_w13; else ++seen_w2;
        const int slot = static_cast<int>(t.kind) * kind_span + t.m_tile * stride + t.n_index;
        assert(hist[slot] == 0);                      // no duplicate task
        hist[slot] = 1;
    }
    assert(seen_w13 == static_cast<int64_t>(expected_tiles) * G::W13_TASKS_PER_TILE);
    assert(seen_w2 == static_cast<int64_t>(expected_tiles) * G::W2_TASKS_PER_TILE);
    assert(seen_w13 + seen_w2 == total);
}

// The header promises constexpr (i.e. __host__ __device__ constexpr under nvcc),
// so the decode must be usable in a constant expression.
static_assert(minibatches(shape{12288}) == 12);
static_assert(total_tasks<2>(shape{12288}) == 6144);
static_assert(decode_task<2>(16 * 16, shape{12288}).kind == task_kind::w2);

int main() {
    shape s{12288};                         // 2048 token/rank, EP4, uniform
    assert(minibatches(s) == 12);
    assert(rows_in_minibatch(s, 11) == 1024);
    shape t{12288 + 64};                    // one tail tile
    assert(minibatches(t) == 13 && tiles_in_minibatch(t, 12) == 1);
    using G2 = geometry<2>;
    assert(G2::W13_TASKS_PER_TILE == 16 && G2::W2_TASKS_PER_TILE == 16);
    assert(total_tasks<2>(s) == 12288 / 64 * 32);
    task a = decode_task<2>(0, s);
    assert(a.kind == task_kind::w13 && a.minibatch == 0 && a.m_tile == 0 && a.n_index == 0);
    task b = decode_task<2>(1, s);          // m inner
    assert(b.kind == task_kind::w13 && b.m_tile == 1 && b.n_index == 0);
    task c = decode_task<2>(16, s);         // second i-tile
    assert(c.kind == task_kind::w13 && c.m_tile == 0 && c.n_index == 1);
    task d = decode_task<2>(16 * 16, s);    // first W2 of minibatch 0
    assert(d.kind == task_kind::w2 && d.minibatch == 0 && d.m_tile == 0 && d.n_index == 0);
    task e = decode_task<2>(16 * 32, s);    // first task of minibatch 1
    assert(e.kind == task_kind::w13 && e.minibatch == 1 && e.m_tile == 16);
    task f = decode_task<2>(total_tasks<2>(s), s);
    assert(f.kind == task_kind::none);
    task g = decode_task<2>(total_tasks<2>(t) - 1, t);   // tail minibatch, single tile
    assert(g.kind == task_kind::w2 && g.m_tile == 192 && g.n_index == 15);
    assert(dispatch_tickets(s, 0) == 128 && ticket_first_row(s, 1, 3) == 1024 + 24);
    assert(x_ready_target(s, 0) == 1024 && hidden_ready_target<2>() == 16 && y_ready_target<2>() == 16);

    // A negative index is out of range just like an index past the end.
    assert(decode_task<2>(-1, s).kind == task_kind::none);
    assert(decode_task<1>(-1, s).kind == task_kind::none);

    // Full enumeration is a bijection onto the (kind, m_tile, n_index) space.
    check_coverage<2>(s, 192);              // 192*16 W13 tasks + 192*16 W2 tasks
    static_assert(geometry<1>::W2_TASKS_PER_TILE == 32);
    assert(total_tasks<1>(s) == 12288 / 64 * 48);
    check_coverage<1>(s, 192);              // 192*16 W13 tasks + 192*32 W2 tasks
    check_coverage<2>(t, 193);              // tail minibatch of a single tile
    check_coverage<1>(t, 193);

    // A single-tile shape: one (partial) minibatch, one tile's worth of tasks.
    shape u{64};
    assert(minibatches(u) == 1);
    assert(tiles_in_minibatch(u, 0) == 1);
    assert(total_tasks<2>(u) == 32);
    assert(dispatch_tickets(u, 0) == 8);
    assert(ticket_first_row(u, 0, 7) == 56);
    assert(x_ready_target(u, 0) == 64);
    check_coverage<2>(u, 1);
    check_coverage<1>(u, 1);

    std::puts("warprole config probe OK");
    return 0;
}
