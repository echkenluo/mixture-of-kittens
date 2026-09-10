#pragma once
#include <cstdint>
#if defined(MOK_WARPROLE_HOST_ONLY)
#define MOK_WARPROLE_HD constexpr
#else
#define MOK_WARPROLE_HD __host__ __device__ constexpr
#endif
namespace mok_sm90::warprole {
constexpr int HIDDEN = 4096, INTER = 2048, TOPK = 6;
constexpr int M_TILE = 64, N_TILE = 128, K_TILE = 128;
constexpr int W13_K_BLOCKS = HIDDEN / K_TILE;   // 32
constexpr int W2_K_BLOCKS = INTER / K_TILE;     // 16
constexpr int MINIBATCH_ROWS = 1024;
constexpr int MINIBATCH_TILES = MINIBATCH_ROWS / M_TILE;   // 16
constexpr int DISPATCH_TICKET_ROWS = 8;
// Named barriers are CTA-wide, not private to a warpgroup.  Reserve 1-4 for
// consumers, 5-6 for W13 hand-off, 7 for epilogue warps, and 8 for comm.
constexpr int BAR_D_FULL = 5;
constexpr int BAR_D_EMPTY = 6;
constexpr int BAR_EPI = 7;
constexpr int BAR_COMM = 8;
// Pair-mode transitions: two consumers plus the producer's TMA warp.
constexpr int BAR_PAIR_MODE = 9;
static_assert(BAR_PAIR_MODE > BAR_COMM && BAR_PAIR_MODE < 16);
static_assert(BAR_COMM != BAR_EPI && BAR_COMM != BAR_D_FULL &&
              BAR_COMM != BAR_D_EMPTY && BAR_COMM > 4 && BAR_COMM < 16);
static_assert(MINIBATCH_ROWS % M_TILE == 0 && MINIBATCH_ROWS % DISPATCH_TICKET_ROWS == 0);

template <int NUM_CONSUMERS> struct geometry {
    static_assert(NUM_CONSUMERS == 1 || NUM_CONSUMERS == 2);
    static constexpr int W13_TASKS_PER_TILE = INTER / N_TILE;                    // 16
    static constexpr int W2_TASKS_PER_TILE = HIDDEN / (N_TILE * NUM_CONSUMERS);  // 32 or 16
    static constexpr int TASKS_PER_TILE = W13_TASKS_PER_TILE + W2_TASKS_PER_TILE;
};

struct shape { int num_rows; };
enum class task_kind : int { w13 = 0, w2 = 1, none = 2 };
struct task { task_kind kind; int minibatch; int m_tile; int n_index; };

MOK_WARPROLE_HD int minibatches(shape s) { return (s.num_rows + MINIBATCH_ROWS - 1) / MINIBATCH_ROWS; }
MOK_WARPROLE_HD int rows_in_minibatch(shape s, int q) {
    const int remaining = s.num_rows - q * MINIBATCH_ROWS;
    return remaining < MINIBATCH_ROWS ? remaining : MINIBATCH_ROWS;
}
MOK_WARPROLE_HD int tiles_in_minibatch(shape s, int q) { return rows_in_minibatch(s, q) / M_TILE; }
MOK_WARPROLE_HD int first_tile_of_minibatch(int q) { return q * MINIBATCH_TILES; }

template <int NC> MOK_WARPROLE_HD int64_t tasks_before_minibatch(shape s, int q) {
    // all minibatches before q are full except none (only the last can be partial)
    (void)s;
    return static_cast<int64_t>(q) * MINIBATCH_TILES * geometry<NC>::TASKS_PER_TILE;
}
template <int NC> MOK_WARPROLE_HD int64_t total_tasks(shape s) {
    return static_cast<int64_t>(s.num_rows / M_TILE) * geometry<NC>::TASKS_PER_TILE;
}
// Position of a task inside one phase (all the W13 or all the W2 tasks of one
// minibatch): n-major, so the 16 tiles of a row block share one weight slab in L2.
template <int NC> MOK_WARPROLE_HD task phase_task(task_kind kind, int q, int tiles, int64_t local) {
    return task{kind, q, first_tile_of_minibatch(q) + static_cast<int>(local % tiles),
                static_cast<int>(local / tiles)};
}
// Stream order: W13(0), then W13(q), W2(q-1) for q = 1..Q-1, then W2(Q-1).  The W2
// phase of a minibatch trails its W13 phase by one full phase, so hidden_ready[m]
// is complete well before any CTA reaches a W2 task of that row block; issuing
// W2(q) right after W13(q) made the first W2 wave wait for the last W13 wave that
// was still running on other CTAs (about one task time per minibatch).
template <int NC> MOK_WARPROLE_HD task decode_task(int64_t idx, shape s) {
    using G = geometry<NC>;
    if (idx < 0 || idx >= total_tasks<NC>(s)) return task{task_kind::none, -1, -1, -1};
    const int Q = minibatches(s);
    const int last_tiles = tiles_in_minibatch(s, Q - 1);
    if (Q == 1) {
        const int64_t w13_count = static_cast<int64_t>(last_tiles) * G::W13_TASKS_PER_TILE;
        return idx < w13_count ? phase_task<NC>(task_kind::w13, 0, last_tiles, idx)
                               : phase_task<NC>(task_kind::w2, 0, last_tiles, idx - w13_count);
    }
    constexpr int64_t F13 = static_cast<int64_t>(MINIBATCH_TILES) * G::W13_TASKS_PER_TILE;
    constexpr int64_t F2 = static_cast<int64_t>(MINIBATCH_TILES) * G::W2_TASKS_PER_TILE;
    if (idx < F13) return phase_task<NC>(task_kind::w13, 0, MINIBATCH_TILES, idx);
    idx -= F13;
    // pair k = [W13(k+1), W2(k)]; pairs 0..Q-3 are full, pair Q-2 holds the (possibly partial) last W13
    const int64_t k = idx / (F13 + F2);
    if (k < Q - 2) {
        const int64_t local = idx - k * (F13 + F2);
        return local < F13 ? phase_task<NC>(task_kind::w13, static_cast<int>(k) + 1, MINIBATCH_TILES, local)
                           : phase_task<NC>(task_kind::w2, static_cast<int>(k), MINIBATCH_TILES, local - F13);
    }
    int64_t local = idx - static_cast<int64_t>(Q - 2) * (F13 + F2);
    const int64_t last_w13 = static_cast<int64_t>(last_tiles) * G::W13_TASKS_PER_TILE;
    if (local < last_w13) return phase_task<NC>(task_kind::w13, Q - 1, last_tiles, local);
    local -= last_w13;
    if (local < F2) return phase_task<NC>(task_kind::w2, Q - 2, MINIBATCH_TILES, local);
    return phase_task<NC>(task_kind::w2, Q - 1, last_tiles, local - F2);
}
MOK_WARPROLE_HD int dispatch_tickets(shape s, int q) { return rows_in_minibatch(s, q) / DISPATCH_TICKET_ROWS; }
MOK_WARPROLE_HD int ticket_first_row(shape s, int q, int t) {
    (void)s;   // kept in the signature for symmetry with the other shape helpers
    return q * MINIBATCH_ROWS + t * DISPATCH_TICKET_ROWS;
}
MOK_WARPROLE_HD int x_ready_target(shape s, int q) { return rows_in_minibatch(s, q); }
template <int NC> MOK_WARPROLE_HD int hidden_ready_target() { return geometry<NC>::W13_TASKS_PER_TILE; }
// Each consumer publishes one increment per finished N128 tile of routed_y, so the
// target is the N128 tile count of a row block regardless of the consumer count.
template <int NC> MOK_WARPROLE_HD int y_ready_target() { return HIDDEN / N_TILE; }
}  // namespace mok_sm90::warprole
