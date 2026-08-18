#pragma once

#include <cstdint>

// Phase 3B logical-decoder milestone only.  This header deliberately contains
// no forward entry point and no numerical task body.
namespace mok_sm90 {
namespace fp8_block_terminal {

constexpr int HIDDEN_SIZE = 4096;
constexpr int INTERMEDIATE_SIZE = 2048;
constexpr int TOP_K = 6;
constexpr int EP_SIZE = 4;

constexpr int CLUSTER_CTAS = 2;
constexpr int THREADS_PER_CTA = 128;
constexpr int M_TILE = 64;
constexpr int N_TILE = 128;
constexpr int W13_N_TILES = INTERMEDIATE_SIZE / N_TILE;
constexpr int W2_N_TILES = HIDDEN_SIZE / N_TILE;
constexpr int W13_CLUSTER_TASKS = W13_N_TILES;
constexpr int W2_CLUSTER_TASKS = W2_N_TILES;
constexpr int W13_N64_SUBTILES = CLUSTER_CTAS * W13_N_TILES;
constexpr int W2_N64_SUBTILES = CLUSTER_CTAS * W2_N_TILES;
constexpr int TASKS_PER_M64 =
    2 * W13_CLUSTER_TASKS + 1 + W2_CLUSTER_TASKS;
constexpr int N64_SUBTILES_PER_M64 =
    2 * W13_N64_SUBTILES + W2_N64_SUBTILES;
constexpr int COMM_ROWS_PER_CTA_TASK = 4;
constexpr int COMM_CTAS_PER_TICKET = CLUSTER_CTAS;
constexpr int COMM_ROWS_PER_TICKET =
    COMM_ROWS_PER_CTA_TASK * COMM_CTAS_PER_TICKET;

static_assert(HIDDEN_SIZE == 4096, "terminal specialization freezes H4096");
static_assert(INTERMEDIATE_SIZE == 2048,
              "terminal specialization freezes I2048");
static_assert(TOP_K == 6, "terminal specialization freezes top6");
static_assert(EP_SIZE == 4, "terminal specialization freezes EP4");
static_assert(CLUSTER_CTAS == 2, "terminal specialization freezes cluster2");
static_assert(THREADS_PER_CTA == 128,
              "terminal specialization freezes 128 threads per CTA");
static_assert(M_TILE == 64 && N_TILE == 128,
              "terminal specialization freezes M64/N128");
static_assert(W13_N_TILES == 16 && W2_N_TILES == 32,
              "terminal specialization tile counts changed");
static_assert(W13_CLUSTER_TASKS == 16 && W2_CLUSTER_TASKS == 32,
              "cluster-level N128 task counts changed");
static_assert(W13_N64_SUBTILES == 32 && W2_N64_SUBTILES == 64,
              "CTA-level N64 subtile counts changed");
static_assert(TASKS_PER_M64 == 65,
              "terminal specialization must expose 65 cluster tasks/M64");
static_assert(N64_SUBTILES_PER_M64 == 128,
              "expanded CTA N64 subtile cardinality changed");
static_assert(COMM_ROWS_PER_CTA_TASK == 4
                  && COMM_ROWS_PER_TICKET == 8,
              "native communication task geometry changed");
static_assert((M_TILE % COMM_ROWS_PER_TICKET) == 0,
              "one communication ticket must not straddle M64 tiles");

#if defined(__CUDACC__)
#define MOK_TERMINAL_HD __host__ __device__ constexpr
#else
#define MOK_TERMINAL_HD constexpr
#endif

enum class logical_stage : int {
    gate = 0,
    up = 1,
    activation = 2,
    w2 = 3,
    count = 4,
};

enum class ready_counter : int {
    x_routed = 0,
    gate_up_tile = 1,
    hidden_row_block = 2,
    y_routed = 3,
    y_routed_done = 4,
    count = 5,
};

enum class communication_stage : int {
    dispatch = 0,
    combine = 1,
};

struct logical_shape {
    bool valid;
    int num_tokens;
    int schedule_capacity;
    int minibatch_rows;
    int macrobatch_rows;
    int num_macrobatches;
    int minibatches_per_macrobatch;
    int num_global_minibatches;
    int last_macrobatch_minibatches;
    int64_t tasks_per_full_minibatch;
    int tail_m_tiles;
    int64_t tasks_in_tail_minibatch;
    int ordered_tail_minibatch;
    int64_t tail_ordinal;
    int64_t total_tasks;
};

struct ordinal_range {
    int64_t begin;
    int64_t end;
};

struct ordered_minibatch_range {
    bool valid;
    int ordered_minibatch;
    int macrobatch;
    int minibatch;
    int first_row;
    int first_m_tile;
    int active_m_tiles;
    ordinal_range all;
};

struct logical_coordinate {
    bool valid;
    int64_t cursor_ordinal;
    int ordered_minibatch;
    int macrobatch;
    int minibatch;
    int global_m;
    int n128;
    logical_stage stage;
};

// One dense communication ticket represents the lockstep pair of native
// four-row CTA tasks.  The two CTAs execute the same stage and adjacent task
// indices, preserving the original per-CTA D/C loop without assigning work to
// a physical role that may be admitted late.
struct communication_coordinate {
    bool valid;
    int64_t cursor_ordinal;
    communication_stage stage;
    int macrobatch;
    int round;
};

struct communication_cta_task {
    bool valid;
    int task_index;
    int first_row;
    int active_rows;
};

MOK_TERMINAL_HD int ceil_div_nonnegative(int value, int divisor) {
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

MOK_TERMINAL_HD bool valid_logical_contract(int num_tokens,
                                            int schedule_capacity,
                                            int minibatch_rows,
                                            int macrobatch_rows) {
    return num_tokens >= 0 && schedule_capacity >= num_tokens
        && (num_tokens % M_TILE) == 0
        && (schedule_capacity % M_TILE) == 0
        && minibatch_rows > 0 && (minibatch_rows % M_TILE) == 0
        && macrobatch_rows > 0
        && (macrobatch_rows % minibatch_rows) == 0;
}

MOK_TERMINAL_HD logical_shape make_logical_shape(
        int num_tokens, int schedule_capacity, int minibatch_rows,
        int macrobatch_rows) {
    logical_shape result{};
    result.valid = valid_logical_contract(
        num_tokens, schedule_capacity, minibatch_rows, macrobatch_rows);
    if (!result.valid)
        return result;

    result.num_tokens = num_tokens;
    result.schedule_capacity = schedule_capacity;
    result.minibatch_rows = minibatch_rows;
    result.macrobatch_rows = macrobatch_rows;
    result.minibatches_per_macrobatch = macrobatch_rows / minibatch_rows;
    result.num_macrobatches = ceil_div_nonnegative(
        num_tokens, macrobatch_rows);
    result.num_global_minibatches = ceil_div_nonnegative(
        num_tokens, minibatch_rows);
    result.last_macrobatch_minibatches =
        result.num_global_minibatches
        - (result.num_macrobatches == 0
               ? 0
               : (result.num_macrobatches - 1)
                     * result.minibatches_per_macrobatch);
    result.tasks_per_full_minibatch = static_cast<int64_t>(TASKS_PER_M64)
        * (minibatch_rows / M_TILE);
    result.total_tasks = static_cast<int64_t>(TASKS_PER_M64)
        * (num_tokens / M_TILE);

    if (num_tokens == 0) {
        result.ordered_tail_minibatch = -1;
        return result;
    }

    result.tail_m_tiles =
        (num_tokens - (result.num_global_minibatches - 1) * minibatch_rows)
        / M_TILE;
    result.tasks_in_tail_minibatch = static_cast<int64_t>(TASKS_PER_M64)
        * result.tail_m_tiles;
    result.ordered_tail_minibatch = result.last_macrobatch_minibatches - 1;
    result.tail_ordinal = static_cast<int64_t>(
        result.ordered_tail_minibatch) * result.tasks_per_full_minibatch;
    return result;
}

MOK_TERMINAL_HD ordered_minibatch_range decode_ordered_minibatch(
        const logical_shape &shape, int ordered_minibatch) {
    ordered_minibatch_range result{};
    if (!shape.valid || ordered_minibatch < 0
            || ordered_minibatch >= shape.num_global_minibatches)
        return result;

    result.valid = true;
    result.ordered_minibatch = ordered_minibatch;
    if (ordered_minibatch < shape.last_macrobatch_minibatches) {
        result.macrobatch = shape.num_macrobatches - 1;
        result.minibatch = ordered_minibatch;
    } else {
        const int z = ordered_minibatch
            - shape.last_macrobatch_minibatches;
        result.macrobatch = shape.num_macrobatches - 2
            - z / shape.minibatches_per_macrobatch;
        result.minibatch = z % shape.minibatches_per_macrobatch;
    }
    result.first_row = result.macrobatch * shape.macrobatch_rows
        + result.minibatch * shape.minibatch_rows;
    result.first_m_tile = result.first_row / M_TILE;
    int remaining_rows = shape.num_tokens - result.first_row;
    if (remaining_rows > shape.minibatch_rows)
        remaining_rows = shape.minibatch_rows;
    result.active_m_tiles = remaining_rows / M_TILE;

    if (ordered_minibatch < shape.ordered_tail_minibatch) {
        result.all.begin = static_cast<int64_t>(ordered_minibatch)
            * shape.tasks_per_full_minibatch;
    } else if (ordered_minibatch == shape.ordered_tail_minibatch) {
        result.all.begin = shape.tail_ordinal;
    } else {
        result.all.begin = shape.tail_ordinal
            + shape.tasks_in_tail_minibatch
            + static_cast<int64_t>(ordered_minibatch
                  - shape.ordered_tail_minibatch - 1)
                * shape.tasks_per_full_minibatch;
    }
    result.all.end = result.all.begin
        + static_cast<int64_t>(TASKS_PER_M64) * result.active_m_tiles;
    return result;
}

// Map a dense communication-queue ticket onto the same reverse-macrobatch,
// ordered-minibatch M64 traversal used by the logical compute decoder.  The
// queue is deliberately dense: a communication role may be admitted late, so
// work cannot be statically striped by role id.
MOK_TERMINAL_HD int decode_ordered_m_tile(
        const logical_shape &shape, int tile_ordinal) {
    if (!shape.valid || tile_ordinal < 0
            || tile_ordinal >= shape.num_tokens / M_TILE)
        return -1;

    int remaining = tile_ordinal;
    for (int ordered = 0; ordered < shape.num_global_minibatches; ++ordered) {
        const ordered_minibatch_range range =
            decode_ordered_minibatch(shape, ordered);
        if (!range.valid)
            return -1;
        if (remaining < range.active_m_tiles)
            return range.first_m_tile + remaining;
        remaining -= range.active_m_tiles;
    }
    return -1;
}

MOK_TERMINAL_HD int communication_rows(
        const logical_shape &shape, int macrobatch) {
    if (!shape.valid || macrobatch < 0
            || macrobatch >= shape.num_macrobatches)
        return 0;
    int rows = shape.num_tokens - macrobatch * shape.macrobatch_rows;
    if (rows > shape.macrobatch_rows)
        rows = shape.macrobatch_rows;
    return rows > 0 ? rows : 0;
}

MOK_TERMINAL_HD int communication_rounds(
        const logical_shape &shape, int macrobatch) {
    return ceil_div_nonnegative(
        communication_rows(shape, macrobatch), COMM_ROWS_PER_TICKET);
}

// Native communication order, flattened into a dense cluster-ticket stream:
//
//   D(Q-1)
//   for q = Q-1 .. 0:
//     for each native task round: C(q), then D(q-1) when it exists
//
// Q=1 therefore remains exactly D(0)->C(0); only Q>=2 contains the
// combine(q)/dispatch(q-1) task-level interleave.
MOK_TERMINAL_HD int64_t communication_total_tickets(
        const logical_shape &shape) {
    if (!shape.valid || shape.num_macrobatches == 0)
        return 0;
    // Every active row is dispatched once and combined once.  One dense
    // cluster ticket carries two four-row CTA tasks, so the complete D/C
    // sequence always contains num_tokens / 4 tickets.  Keeping this closed
    // form out of the resident kernel avoids a Q-dependent decoder loop.
    return static_cast<int64_t>(shape.num_tokens) / 4;
}

MOK_TERMINAL_HD communication_coordinate decode_communication_cursor(
        const logical_shape &shape, int64_t cursor_ordinal) {
    communication_coordinate result{};
    result.cursor_ordinal = cursor_ordinal;
    const int64_t total = communication_total_tickets(shape);
    if (!shape.valid || cursor_ordinal < 0 || cursor_ordinal >= total)
        return result;

    int remaining = static_cast<int>(cursor_ordinal);
    const int last = shape.num_macrobatches - 1;
    const int full_rounds = shape.macrobatch_rows / COMM_ROWS_PER_TICKET;
    const int initial_dispatch = communication_rounds(shape, last);
    if (remaining < initial_dispatch) {
        result.valid = true;
        result.stage = communication_stage::dispatch;
        result.macrobatch = last;
        result.round = static_cast<int>(remaining);
        return result;
    }
    remaining -= initial_dispatch;

    // Q=1 closes directly with C(0).
    if (shape.num_macrobatches == 1) {
        result.valid = true;
        result.stage = communication_stage::combine;
        result.macrobatch = 0;
        result.round = remaining;
        return result;
    }

    // The tail macrobatch can be shorter than A.  Pair its C(last) rounds
    // with the same prefix of D(last-1), then drain the remaining full-macro
    // dispatch rounds exactly as the native max(C,D) loop does.
    const int tail_paired = 2 * initial_dispatch;
    if (remaining < tail_paired) {
        result.valid = true;
        result.round = remaining / 2;
        if ((remaining & 1) == 0) {
            result.stage = communication_stage::combine;
            result.macrobatch = last;
        } else {
            result.stage = communication_stage::dispatch;
            result.macrobatch = last - 1;
        }
        return result;
    }
    remaining -= tail_paired;
    const int tail_dispatch_remainder = full_rounds - initial_dispatch;
    if (remaining < tail_dispatch_remainder) {
        result.valid = true;
        result.stage = communication_stage::dispatch;
        result.macrobatch = last - 1;
        result.round = initial_dispatch + remaining;
        return result;
    }
    remaining -= tail_dispatch_remainder;

    // All interior macrobatches are full, so each segment has the same
    // 2*full_rounds C(q),D(q-1) alternating shape and can be decoded in O(1).
    const int interior_segments = shape.num_macrobatches - 2;
    const int interior_tickets = interior_segments * 2 * full_rounds;
    if (remaining < interior_tickets) {
        const int segment = remaining / (2 * full_rounds);
        const int local = remaining - segment * 2 * full_rounds;
        const int q = last - 1 - segment;
        result.valid = true;
        result.round = local / 2;
        if ((local & 1) == 0) {
            result.stage = communication_stage::combine;
            result.macrobatch = q;
        } else {
            result.stage = communication_stage::dispatch;
            result.macrobatch = q - 1;
        }
        return result;
    }
    remaining -= interior_tickets;

    // Final C(0) has no following dispatch.
    result.valid = true;
    result.stage = communication_stage::combine;
    result.macrobatch = 0;
    result.round = remaining;
    return result;
}

MOK_TERMINAL_HD communication_cta_task decode_communication_cta_task(
        const logical_shape &shape,
        const communication_coordinate &coordinate,
        int cta_rank) {
    communication_cta_task result{};
    if (!shape.valid || !coordinate.valid || cta_rank < 0
            || cta_rank >= COMM_CTAS_PER_TICKET)
        return result;
    const int rows = communication_rows(shape, coordinate.macrobatch);
    const int task_index = coordinate.round * COMM_CTAS_PER_TICKET + cta_rank;
    const int row_in_macrobatch = task_index * COMM_ROWS_PER_CTA_TASK;
    if (row_in_macrobatch >= rows)
        return result;
    int active = rows - row_in_macrobatch;
    if (active > COMM_ROWS_PER_CTA_TASK)
        active = COMM_ROWS_PER_CTA_TASK;
    result.valid = true;
    result.task_index = task_index;
    result.first_row = coordinate.macrobatch * shape.macrobatch_rows
        + row_in_macrobatch;
    result.active_rows = active;
    return result;
}

MOK_TERMINAL_HD ordinal_range stage_range(
        const ordered_minibatch_range &range, logical_stage stage) {
    ordinal_range result{range.all.begin, range.all.begin};
    if (!range.valid)
        return result;
    const int64_t r = range.active_m_tiles;
    if (stage == logical_stage::gate) {
        result.end += W13_CLUSTER_TASKS * r;
    } else if (stage == logical_stage::up) {
        result.begin += W13_CLUSTER_TASKS * r;
        result.end = result.begin + W13_CLUSTER_TASKS * r;
    } else if (stage == logical_stage::activation) {
        result.begin += 2 * W13_CLUSTER_TASKS * r;
        result.end = result.begin + r;
    } else if (stage == logical_stage::w2) {
        result.begin += (2 * W13_CLUSTER_TASKS + 1) * r;
        result.end = result.begin + W2_CLUSTER_TASKS * r;
    }
    return result;
}

MOK_TERMINAL_HD logical_coordinate decode_logical_cursor(
        const logical_shape &shape, int64_t cursor_ordinal) {
    logical_coordinate result{};
    result.cursor_ordinal = cursor_ordinal;
    if (!shape.valid || cursor_ordinal < 0
            || cursor_ordinal >= shape.total_tasks)
        return result;

    int ordered_minibatch = 0;
    int64_t local_task = 0;
    if (cursor_ordinal < shape.tail_ordinal) {
        ordered_minibatch = static_cast<int>(
            cursor_ordinal / shape.tasks_per_full_minibatch);
        local_task = cursor_ordinal % shape.tasks_per_full_minibatch;
    } else if (cursor_ordinal
                   < shape.tail_ordinal + shape.tasks_in_tail_minibatch) {
        ordered_minibatch = shape.ordered_tail_minibatch;
        local_task = cursor_ordinal - shape.tail_ordinal;
    } else {
        const int64_t v = cursor_ordinal
            - (shape.tail_ordinal + shape.tasks_in_tail_minibatch);
        ordered_minibatch = shape.ordered_tail_minibatch + 1
            + static_cast<int>(v / shape.tasks_per_full_minibatch);
        local_task = v % shape.tasks_per_full_minibatch;
    }

    const ordered_minibatch_range range =
        decode_ordered_minibatch(shape, ordered_minibatch);
    if (!range.valid)
        return result;

    const int r = range.active_m_tiles;
    int stage_local = 0;
    int m_in_minibatch = 0;
    if (local_task < static_cast<int64_t>(W13_CLUSTER_TASKS) * r) {
        result.stage = logical_stage::gate;
        stage_local = static_cast<int>(local_task);
        m_in_minibatch = stage_local / W13_CLUSTER_TASKS;
        result.n128 = stage_local % W13_CLUSTER_TASKS;
    } else if (local_task
                   < static_cast<int64_t>(2 * W13_CLUSTER_TASKS) * r) {
        result.stage = logical_stage::up;
        stage_local = static_cast<int>(
            local_task - static_cast<int64_t>(W13_CLUSTER_TASKS) * r);
        m_in_minibatch = stage_local / W13_CLUSTER_TASKS;
        result.n128 = stage_local % W13_CLUSTER_TASKS;
    } else if (local_task
                   < static_cast<int64_t>(2 * W13_CLUSTER_TASKS + 1) * r) {
        result.stage = logical_stage::activation;
        stage_local = static_cast<int>(
            local_task - static_cast<int64_t>(2 * W13_CLUSTER_TASKS) * r);
        m_in_minibatch = stage_local;
        result.n128 = 0;
    } else {
        result.stage = logical_stage::w2;
        stage_local = static_cast<int>(
            local_task
            - static_cast<int64_t>(2 * W13_CLUSTER_TASKS + 1) * r);
        m_in_minibatch = stage_local / W2_CLUSTER_TASKS;
        result.n128 = stage_local % W2_CLUSTER_TASKS;
    }

    result.valid = m_in_minibatch >= 0 && m_in_minibatch < r;
    result.ordered_minibatch = ordered_minibatch;
    result.macrobatch = range.macrobatch;
    result.minibatch = range.minibatch;
    result.global_m = range.first_m_tile + m_in_minibatch;
    return result;
}

MOK_TERMINAL_HD int n64_for_cta(
        const logical_coordinate &coordinate, int cta_rank) {
    if (!coordinate.valid || coordinate.stage == logical_stage::activation
            || cta_rank < 0 || cta_rank >= CLUSTER_CTAS)
        return -1;
    return CLUSTER_CTAS * coordinate.n128 + cta_rank;
}

MOK_TERMINAL_HD int counter_entries_touched_per_cluster_task(
        ready_counter counter, logical_stage stage) {
    if (counter == ready_counter::gate_up_tile
            && (stage == logical_stage::gate || stage == logical_stage::up))
        return 1;
    if (counter == ready_counter::hidden_row_block
            && stage == logical_stage::activation)
        return 1;
    if (counter == ready_counter::y_routed && stage == logical_stage::w2)
        return 1;
    return 0;
}

MOK_TERMINAL_HD int counter_arrival_delta_per_entry(
        ready_counter counter, logical_stage stage) {
    if (counter == ready_counter::gate_up_tile
            && (stage == logical_stage::gate || stage == logical_stage::up))
        return 1;
    if (counter == ready_counter::hidden_row_block
            && stage == logical_stage::activation)
        return 1;
    if (counter == ready_counter::y_routed && stage == logical_stage::w2)
        return 1;
    return 0;
}

MOK_TERMINAL_HD int counter_arrivals_per_cluster_task(
        ready_counter counter, logical_stage stage) {
    return counter_entries_touched_per_cluster_task(counter, stage)
        * counter_arrival_delta_per_entry(counter, stage);
}

MOK_TERMINAL_HD int counter_entries_per_m64(ready_counter counter) {
    return counter == ready_counter::gate_up_tile ? W13_N_TILES : 1;
}

MOK_TERMINAL_HD int counter_expected_arrivals(ready_counter counter) {
    if (counter == ready_counter::x_routed
            || counter == ready_counter::y_routed_done)
        return M_TILE;
    if (counter == ready_counter::gate_up_tile)
        return 2;
    if (counter == ready_counter::hidden_row_block)
        return 1;
    if (counter == ready_counter::y_routed)
        return W2_N_TILES;
    return 0;
}

MOK_TERMINAL_HD int counter_total_arrivals_per_m64(
        ready_counter counter) {
    return counter_entries_per_m64(counter)
        * counter_expected_arrivals(counter);
}

MOK_TERMINAL_HD int64_t counter_cardinality(
        ready_counter counter, int rows) {
    if (rows < 0 || (rows % M_TILE) != 0
            || counter < ready_counter::x_routed
            || counter >= ready_counter::count)
        return -1;
    return static_cast<int64_t>(rows / M_TILE)
        * counter_entries_per_m64(counter);
}

MOK_TERMINAL_HD int64_t counter_index(
        ready_counter counter, int global_m, int n128 = 0) {
    if (global_m < 0 || counter < ready_counter::x_routed
            || counter >= ready_counter::count)
        return -1;
    if (counter == ready_counter::gate_up_tile) {
        if (n128 < 0 || n128 >= W13_N_TILES)
            return -1;
        return static_cast<int64_t>(global_m) * W13_N_TILES + n128;
    }
    return global_m;
}

#undef MOK_TERMINAL_HD

}  // namespace fp8_block_terminal
}  // namespace mok_sm90
