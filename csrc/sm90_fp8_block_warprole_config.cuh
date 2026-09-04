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
template <int NC> MOK_WARPROLE_HD task decode_task(int64_t idx, shape s) {
    using G = geometry<NC>;
    if (idx < 0 || idx >= total_tasks<NC>(s)) return task{task_kind::none, -1, -1, -1};
    const int64_t full = static_cast<int64_t>(MINIBATCH_TILES) * G::TASKS_PER_TILE;
    const int q = static_cast<int>(idx / full);
    const int tiles = tiles_in_minibatch(s, q);
    int64_t local = idx - static_cast<int64_t>(q) * full;
    const int64_t w13_count = static_cast<int64_t>(tiles) * G::W13_TASKS_PER_TILE;
    task t{task_kind::none, q, -1, -1};
    if (local < w13_count) {
        t.kind = task_kind::w13;
        t.n_index = static_cast<int>(local / tiles);
        t.m_tile = first_tile_of_minibatch(q) + static_cast<int>(local % tiles);
    } else {
        local -= w13_count;
        t.kind = task_kind::w2;
        t.n_index = static_cast<int>(local / tiles);
        t.m_tile = first_tile_of_minibatch(q) + static_cast<int>(local % tiles);
    }
    return t;
}
MOK_WARPROLE_HD int dispatch_tickets(shape s, int q) { return rows_in_minibatch(s, q) / DISPATCH_TICKET_ROWS; }
MOK_WARPROLE_HD int ticket_first_row(shape s, int q, int t) {
    (void)s;   // kept in the signature for symmetry with the other shape helpers
    return q * MINIBATCH_ROWS + t * DISPATCH_TICKET_ROWS;
}
MOK_WARPROLE_HD int x_ready_target(shape s, int q) { return rows_in_minibatch(s, q); }
template <int NC> MOK_WARPROLE_HD int hidden_ready_target() { return geometry<NC>::W13_TASKS_PER_TILE; }
template <int NC> MOK_WARPROLE_HD int y_ready_target() { return geometry<NC>::W2_TASKS_PER_TILE; }
}  // namespace mok_sm90::warprole
