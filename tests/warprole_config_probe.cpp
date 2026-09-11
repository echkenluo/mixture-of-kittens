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

// W2(q) must start after the last W13(q) task and, for every minibatch but the
// last, after the last W13(q+1) task as well: the W2 phase trails by one phase so
// hidden_ready waits are off the critical path.  Within a phase the order stays
// n-major (m_tile inner).
template <int NC>
static void check_order(shape s) {
    const int Q = minibatches(s);
    std::vector<int64_t> last_w13(Q, -1), first_w2(Q, -1), prev_in_phase(Q * 2, -1);
    const int64_t total = total_tasks<NC>(s);
    task prev = decode_task<NC>(0, s);
    for (int64_t idx = 0; idx < total; ++idx) {
        const task t = decode_task<NC>(idx, s);
        if (t.kind == task_kind::w13) last_w13[t.minibatch] = idx;
        else if (first_w2[t.minibatch] < 0) first_w2[t.minibatch] = idx;
        if (idx > 0 && t.kind == prev.kind && t.minibatch == prev.minibatch) {
            const int tiles = tiles_in_minibatch(s, t.minibatch);
            const int64_t p = static_cast<int64_t>(prev.n_index) * tiles + prev.m_tile - first_tile_of_minibatch(t.minibatch);
            const int64_t c = static_cast<int64_t>(t.n_index) * tiles + t.m_tile - first_tile_of_minibatch(t.minibatch);
            assert(c == p + 1);                       // n-major, m inner, no gaps
        }
        prev = t;
    }
    for (int q = 0; q < Q; ++q) {
        assert(last_w13[q] >= 0 && first_w2[q] >= 0);
        assert(first_w2[q] > last_w13[q]);
        if (q + 1 < Q) assert(first_w2[q] > last_w13[q + 1]);
        if (q + 1 < Q) assert(first_w2[q + 1] > first_w2[q]);
    }
}

// The header promises constexpr (i.e. __host__ __device__ constexpr under nvcc),
// so the decode must be usable in a constant expression.
static_assert(minibatches(shape{12288}) == 12288 / MINIBATCH_ROWS);
static_assert(total_tasks<2>(shape{12288}) == 6144);
static_assert(decode_task<2>(MINIBATCH_TILES * 16, shape{12288}).kind == task_kind::w13);   // W13(1) follows W13(0)
static_assert(decode_task<2>(MINIBATCH_TILES * 32, shape{12288}).kind == task_kind::w2);    // then W2(0)

int main() {
    shape s{12288};                         // 2048 token/rank, EP4, uniform
    assert(minibatches(s) == 12288 / MINIBATCH_ROWS);
    assert(rows_in_minibatch(s, minibatches(s)-1) == MINIBATCH_ROWS);
    shape t{12288 + 64};                    // one tail tile
    assert(minibatches(t) == minibatches(s)+1 && tiles_in_minibatch(t, minibatches(s)) == 1);
    using G2 = geometry<2>;
    assert(G2::W13_TASKS_PER_TILE == 16 && G2::W2_TASKS_PER_TILE == 16);
    assert(total_tasks<2>(s) == 12288 / 64 * 32);
    task a = decode_task<2>(0, s);
    assert(a.kind == task_kind::w13 && a.minibatch == 0 && a.m_tile == 0 && a.n_index == 0);
    task b = decode_task<2>(1, s);          // m inner
    assert(b.kind == task_kind::w13 && b.m_tile == 1 && b.n_index == 0);
    task c = decode_task<2>(MINIBATCH_TILES, s);         // second i-tile
    assert(c.kind == task_kind::w13 && c.m_tile == 0 && c.n_index == 1);
    task d = decode_task<2>(MINIBATCH_TILES * 16, s);    // W13 of minibatch 1 comes before W2 of minibatch 0
    assert(d.kind == task_kind::w13 && d.minibatch == 1 && d.m_tile == MINIBATCH_TILES && d.n_index == 0);
    task e = decode_task<2>(MINIBATCH_TILES * 32, s);    // first W2 of minibatch 0
    assert(e.kind == task_kind::w2 && e.minibatch == 0 && e.m_tile == 0 && e.n_index == 0);
    task e2 = decode_task<2>(MINIBATCH_TILES * 48, s);   // then W13 of minibatch 2
    assert(e2.kind == task_kind::w13 && e2.minibatch == 2 && e2.m_tile == 2 * MINIBATCH_TILES && e2.n_index == 0);
    task f = decode_task<2>(total_tasks<2>(s), s);
    assert(f.kind == task_kind::none);
    task g = decode_task<2>(total_tasks<2>(t) - 1, t);   // tail minibatch, single tile
    assert(g.kind == task_kind::w2 && g.m_tile == 192 && g.n_index == 15);
    assert(dispatch_tickets(s, 0) == MINIBATCH_ROWS / 8 && ticket_first_row(s, 1, 3) == MINIBATCH_ROWS + 24);
    assert(x_ready_target(s, 0) == MINIBATCH_ROWS && hidden_ready_target<2>() == 16 && y_ready_target<2>() == 32);

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
    check_order<2>(s); check_order<1>(s); check_order<2>(t); check_order<1>(t);
    shape v{2 * MINIBATCH_ROWS};                          // exactly two minibatches: W13(0), W13(1), W2(0), W2(1)
    check_coverage<2>(v, 2 * MINIBATCH_TILES); check_order<2>(v);
    shape w{MINIBATCH_ROWS + 128};                    // two minibatches, partial second one
    check_coverage<2>(w, MINIBATCH_TILES + 2); check_order<2>(w); check_order<1>(w);

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
    check_order<2>(u);

    std::puts("warprole config probe OK");
    return 0;
}
