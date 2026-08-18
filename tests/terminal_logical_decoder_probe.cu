#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../csrc/sm90_fp8_block_megakernel.cuh"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace terminal = mok_sm90::fp8_block_terminal;

namespace {

struct probe_case {
    const char *name;
    int num_tokens;
    int schedule_capacity;
    int minibatch_rows;
    int macrobatch_rows;
    int expected_q;
};

const probe_case CASES[] = {
    {"empty-rank", 0, 2048, 256, 512, 0},
    {"single-m64", 64, 2048, 256, 512, 1},
    {"balanced-q1", 256, 2048, 128, 256, 1},
    {"balanced-q2", 256, 2048, 128, 128, 2},
    {"skew-heavy-q2", 1024, 2048, 256, 512, 2},
    {"skew-light-partial", 192, 2048, 256, 256, 1},
    {"skew-heavy-q3-partial", 1088, 2048, 256, 512, 3},
};

struct golden_coordinate {
    int64_t ordinal;
    int ordered_minibatch;
    int macrobatch;
    int minibatch;
    int global_m;
    terminal::logical_stage stage;
    int n128;
};

// nt=256, S=A=128: ordered j=0 is q=1/gm=2..3, then j=1 is
// q=0/gm=0..1.  These points pin every stage boundary and both M64s.
const golden_coordinate BALANCED_Q2_GOLDEN[] = {
    {0, 0, 1, 0, 2, terminal::logical_stage::gate, 0},
    {15, 0, 1, 0, 2, terminal::logical_stage::gate, 15},
    {16, 0, 1, 0, 3, terminal::logical_stage::gate, 0},
    {31, 0, 1, 0, 3, terminal::logical_stage::gate, 15},
    {32, 0, 1, 0, 2, terminal::logical_stage::up, 0},
    {63, 0, 1, 0, 3, terminal::logical_stage::up, 15},
    {64, 0, 1, 0, 2, terminal::logical_stage::activation, 0},
    {65, 0, 1, 0, 3, terminal::logical_stage::activation, 0},
    {66, 0, 1, 0, 2, terminal::logical_stage::w2, 0},
    {97, 0, 1, 0, 2, terminal::logical_stage::w2, 31},
    {98, 0, 1, 0, 3, terminal::logical_stage::w2, 0},
    {129, 0, 1, 0, 3, terminal::logical_stage::w2, 31},
    {130, 1, 0, 0, 0, terminal::logical_stage::gate, 0},
    {259, 1, 0, 0, 1, terminal::logical_stage::w2, 31},
};

// nt=1088, S=256, A=512: L=1, so the one-M64 partial tail at q=2
// is j=0.  Full ranges then walk (q,b)=(1,0),(1,1),(0,0),(0,1).
const golden_coordinate SKEW_Q3_PARTIAL_GOLDEN[] = {
    {0, 0, 2, 0, 16, terminal::logical_stage::gate, 0},
    {15, 0, 2, 0, 16, terminal::logical_stage::gate, 15},
    {16, 0, 2, 0, 16, terminal::logical_stage::up, 0},
    {31, 0, 2, 0, 16, terminal::logical_stage::up, 15},
    {32, 0, 2, 0, 16, terminal::logical_stage::activation, 0},
    {33, 0, 2, 0, 16, terminal::logical_stage::w2, 0},
    {64, 0, 2, 0, 16, terminal::logical_stage::w2, 31},
    {65, 1, 1, 0, 8, terminal::logical_stage::gate, 0},
    {324, 1, 1, 0, 11, terminal::logical_stage::w2, 31},
    {325, 2, 1, 1, 12, terminal::logical_stage::gate, 0},
    {584, 2, 1, 1, 15, terminal::logical_stage::w2, 31},
    {585, 3, 0, 0, 0, terminal::logical_stage::gate, 0},
    {844, 3, 0, 0, 3, terminal::logical_stage::w2, 31},
    {845, 4, 0, 1, 4, terminal::logical_stage::gate, 0},
    {1104, 4, 0, 1, 7, terminal::logical_stage::w2, 31},
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition)
        fail(message);
}

void check_golden_set(const probe_case &test,
                      const terminal::logical_shape &shape,
                      const golden_coordinate *golden, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        const auto &expected = golden[index];
        const auto actual =
            terminal::decode_logical_cursor(shape, expected.ordinal);
        const auto minibatch = terminal::decode_ordered_minibatch(
            shape, expected.ordered_minibatch);
        require(actual.valid
                    && minibatch.valid
                    && minibatch.ordered_minibatch
                        == expected.ordered_minibatch
                    && minibatch.macrobatch == expected.macrobatch
                    && minibatch.minibatch == expected.minibatch
                    && actual.global_m == expected.global_m
                    && actual.stage == expected.stage
                    && actual.n128 == expected.n128,
                std::string(test.name) + ": golden coordinate mismatch at "
                    + std::to_string(expected.ordinal));
    }
}

void check_case_goldens(const probe_case &test,
                        const terminal::logical_shape &shape) {
    const std::string name(test.name);
    if (name == "balanced-q2") {
        check_golden_set(test, shape, BALANCED_Q2_GOLDEN,
                         sizeof(BALANCED_Q2_GOLDEN)
                             / sizeof(BALANCED_Q2_GOLDEN[0]));
    } else if (name == "skew-heavy-q3-partial") {
        check_golden_set(test, shape, SKEW_Q3_PARTIAL_GOLDEN,
                         sizeof(SKEW_Q3_PARTIAL_GOLDEN)
                             / sizeof(SKEW_Q3_PARTIAL_GOLDEN[0]));
    }
}

void check_stage_decode_interlock(
        const probe_case &test, const terminal::logical_shape &shape,
        const terminal::ordered_minibatch_range &minibatch,
        terminal::logical_stage stage, terminal::ordinal_range ordinals) {
    for (int64_t ordinal = ordinals.begin; ordinal < ordinals.end;
         ++ordinal) {
        const auto coordinate =
            terminal::decode_logical_cursor(shape, ordinal);
        const int64_t local = ordinal - ordinals.begin;
        int expected_m = minibatch.first_m_tile;
        int expected_n128 = 0;
        if (stage == terminal::logical_stage::gate
                || stage == terminal::logical_stage::up) {
            expected_m += static_cast<int>(
                local / terminal::W13_CLUSTER_TASKS);
            expected_n128 = static_cast<int>(
                local % terminal::W13_CLUSTER_TASKS);
        } else if (stage == terminal::logical_stage::activation) {
            expected_m += static_cast<int>(local);
        } else {
            expected_m += static_cast<int>(
                local / terminal::W2_CLUSTER_TASKS);
            expected_n128 = static_cast<int>(
                local % terminal::W2_CLUSTER_TASKS);
        }
        require(coordinate.valid && coordinate.stage == stage
                    && coordinate.global_m == expected_m
                    && coordinate.n128 == expected_n128,
                std::string(test.name) + ": stage/decode interlock at "
                    + std::to_string(ordinal));
    }
}

#if defined(__CUDACC__)
__host__ __device__
#endif
int canonical_cluster_task(const terminal::logical_coordinate &coordinate,
                           int m_tiles) {
    const int gate_size = m_tiles * terminal::W13_CLUSTER_TASKS;
    const int up_size = gate_size;
    const int activation_size = m_tiles;
    if (coordinate.stage == terminal::logical_stage::gate)
        return coordinate.global_m * terminal::W13_CLUSTER_TASKS
            + coordinate.n128;
    if (coordinate.stage == terminal::logical_stage::up)
        return gate_size
            + coordinate.global_m * terminal::W13_CLUSTER_TASKS
            + coordinate.n128;
    if (coordinate.stage == terminal::logical_stage::activation)
        return gate_size + up_size + coordinate.global_m;
    if (coordinate.stage == terminal::logical_stage::w2)
        return gate_size + up_size + activation_size
            + coordinate.global_m * terminal::W2_CLUSTER_TASKS
            + coordinate.n128;
    return -1;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
int canonical_n64_subtile(const terminal::logical_coordinate &coordinate,
                          int m_tiles, int n64) {
    const int gate_size = m_tiles * terminal::W13_N64_SUBTILES;
    const int up_size = gate_size;
    if (coordinate.stage == terminal::logical_stage::gate)
        return coordinate.global_m * terminal::W13_N64_SUBTILES + n64;
    if (coordinate.stage == terminal::logical_stage::up)
        return gate_size
            + coordinate.global_m * terminal::W13_N64_SUBTILES + n64;
    if (coordinate.stage == terminal::logical_stage::w2)
        return gate_size + up_size
            + coordinate.global_m * terminal::W2_N64_SUBTILES + n64;
    return -1;
}

void check_counter_contract(const probe_case &test,
                            const terminal::logical_shape &shape) {
    const int active_m_tiles = test.num_tokens / terminal::M_TILE;
    const int capacity_m_tiles = test.schedule_capacity / terminal::M_TILE;
    for (int raw = 0; raw < static_cast<int>(terminal::ready_counter::count);
         ++raw) {
        const auto counter = static_cast<terminal::ready_counter>(raw);
        const int entries = terminal::counter_entries_per_m64(counter);
        const int expected = terminal::counter_expected_arrivals(counter);
        const int arrivals =
            terminal::counter_total_arrivals_per_m64(counter);
        const int64_t storage =
            terminal::counter_cardinality(counter, test.schedule_capacity);
        const int64_t active =
            terminal::counter_cardinality(counter, test.num_tokens);
        require(entries > 0 && expected > 0,
                std::string(test.name) + ": missing counter formula");
        require(storage == static_cast<int64_t>(capacity_m_tiles) * entries,
                std::string(test.name) + ": counter storage cardinality");
        require(active == static_cast<int64_t>(active_m_tiles) * entries,
                std::string(test.name) + ": counter active cardinality");
        require(arrivals == entries * expected,
                std::string(test.name) + ": counter arrival formula");
        require(static_cast<int64_t>(active_m_tiles) * arrivals
                    == active * expected,
                std::string(test.name) + ": counter aggregate arrivals");
        if (capacity_m_tiles > 0) {
            const int n = counter == terminal::ready_counter::gate_up_tile
                ? terminal::W13_N_TILES - 1
                : 0;
            require(terminal::counter_index(counter, 0, 0) == 0,
                    std::string(test.name) + ": counter first index");
            require(terminal::counter_index(
                        counter, capacity_m_tiles - 1, n) == storage - 1,
                    std::string(test.name) + ": counter last index");
        }
    }
    require(terminal::counter_expected_arrivals(
                terminal::ready_counter::x_routed) == 64,
            "x_routed expected count");
    require(terminal::counter_expected_arrivals(
                terminal::ready_counter::gate_up_tile) == 2,
            "gate_up expected count");
    require(terminal::counter_expected_arrivals(
                terminal::ready_counter::hidden_row_block) == 1,
            "hidden expected count");
    require(terminal::counter_expected_arrivals(
                terminal::ready_counter::y_routed) == 32,
            "y_routed expected count");
    require(terminal::counter_expected_arrivals(
                terminal::ready_counter::y_routed_done) == 64,
            "y_done expected count");
    require(shape.total_tasks
                == static_cast<int64_t>(terminal::TASKS_PER_M64)
                    * active_m_tiles,
            std::string(test.name) + ": total task formula");
    require(terminal::W13_N64_SUBTILES == 32
                && terminal::W2_N64_SUBTILES == 64,
            std::string(test.name) + ": expanded N64 subtile formula");
    require(terminal::counter_arrivals_per_cluster_task(
                terminal::ready_counter::gate_up_tile,
                terminal::logical_stage::gate) == 1
                && terminal::counter_arrivals_per_cluster_task(
                    terminal::ready_counter::gate_up_tile,
                    terminal::logical_stage::up) == 1
                && terminal::counter_arrivals_per_cluster_task(
                    terminal::ready_counter::hidden_row_block,
                    terminal::logical_stage::activation) == 1
                && terminal::counter_arrivals_per_cluster_task(
                    terminal::ready_counter::y_routed,
                    terminal::logical_stage::w2) == 1,
            std::string(test.name) + ": cluster publication delta");
    require(terminal::counter_entries_touched_per_cluster_task(
                terminal::ready_counter::gate_up_tile,
                terminal::logical_stage::gate) == 1
                && terminal::counter_arrival_delta_per_entry(
                    terminal::ready_counter::gate_up_tile,
                    terminal::logical_stage::gate) == 1
                && terminal::counter_entries_touched_per_cluster_task(
                       terminal::ready_counter::y_routed,
                       terminal::logical_stage::w2) == 1
                && terminal::counter_arrival_delta_per_entry(
                    terminal::ready_counter::y_routed,
                    terminal::logical_stage::w2) == 1,
            std::string(test.name) + ": counter entry/delta contract");
}

void check_communication_decode(
        const probe_case &test, const terminal::logical_shape &shape) {
    using stage = terminal::communication_stage;
    struct expected_coordinate {
        stage kind;
        int macrobatch;
        int round;
    };
    std::vector<expected_coordinate> expected;
    auto append = [&](stage kind, int q, int round) {
        expected.push_back({kind, q, round});
    };
    if (shape.num_macrobatches > 0) {
        const int last = shape.num_macrobatches - 1;
        for (int round = 0;
             round < terminal::communication_rounds(shape, last); ++round)
            append(stage::dispatch, last, round);
        for (int q = last; q >= 0; --q) {
            const int combine_rounds =
                terminal::communication_rounds(shape, q);
            const int dispatch_rounds = q > 0
                ? terminal::communication_rounds(shape, q - 1) : 0;
            for (int round = 0;
                 round < std::max(combine_rounds, dispatch_rounds);
                 ++round) {
                if (round < combine_rounds)
                    append(stage::combine, q, round);
                if (round < dispatch_rounds)
                    append(stage::dispatch, q - 1, round);
            }
        }
    }

    require(terminal::communication_total_tickets(shape)
                == static_cast<int64_t>(expected.size()),
            std::string(test.name) + ": communication ticket cardinality");
    require(!terminal::decode_communication_cursor(shape, -1).valid
                && !terminal::decode_communication_cursor(
                    shape, static_cast<int64_t>(expected.size())).valid,
            std::string(test.name) + ": communication cursor bounds");

    std::vector<int> dispatch_rows(static_cast<size_t>(test.num_tokens), 0);
    std::vector<int> combine_rows(static_cast<size_t>(test.num_tokens), 0);
    for (size_t ticket = 0; ticket < expected.size(); ++ticket) {
        const auto coordinate = terminal::decode_communication_cursor(
            shape, static_cast<int64_t>(ticket));
        const auto &golden = expected[ticket];
        require(coordinate.valid && coordinate.stage == golden.kind
                    && coordinate.macrobatch == golden.macrobatch
                    && coordinate.round == golden.round,
                std::string(test.name)
                    + ": communication order mismatch at ticket "
                    + std::to_string(ticket));
        for (int rank = 0; rank < terminal::CLUSTER_CTAS; ++rank) {
            const auto task = terminal::decode_communication_cta_task(
                shape, coordinate, rank);
            if (!task.valid)
                continue;
            require(task.active_rows > 0
                        && task.active_rows
                            <= terminal::COMM_ROWS_PER_CTA_TASK,
                    std::string(test.name)
                        + ": invalid communication CTA task extent");
            for (int local = 0; local < task.active_rows; ++local) {
                const int row = task.first_row + local;
                require(row >= 0 && row < test.num_tokens,
                        std::string(test.name)
                            + ": communication row out of range");
                auto &visits = coordinate.stage == stage::dispatch
                    ? dispatch_rows : combine_rows;
                ++visits[static_cast<size_t>(row)];
            }
        }
    }
    require(std::all_of(dispatch_rows.begin(), dispatch_rows.end(),
                        [](int count) { return count == 1; })
                && std::all_of(combine_rows.begin(), combine_rows.end(),
                               [](int count) { return count == 1; }),
            std::string(test.name)
                + ": communication row coverage is not exactly once");

    if (shape.num_macrobatches == 1 && !expected.empty()) {
        bool saw_combine = false;
        for (const auto &coordinate : expected) {
            if (coordinate.kind == stage::combine)
                saw_combine = true;
            else
                require(!saw_combine,
                        std::string(test.name)
                            + ": Q1 dispatch appears after combine");
        }
    }
    if (shape.num_macrobatches >= 2) {
        const int last = shape.num_macrobatches - 1;
        const int first_after_initial =
            terminal::communication_rounds(shape, last);
        require(first_after_initial + 1 < static_cast<int>(expected.size())
                    && expected[first_after_initial].kind == stage::combine
                    && expected[first_after_initial].macrobatch == last
                    && expected[first_after_initial + 1].kind
                        == stage::dispatch
                    && expected[first_after_initial + 1].macrobatch
                        == last - 1,
                std::string(test.name)
                    + ": Q2+ first C(q)/D(q-1) pair is not interleaved");
    }

    // Pin the two simplest timelines independently of the generic coverage.
    if (std::string(test.name) == "balanced-q1") {
        require(expected.size() == 64
                    && expected[0].kind == stage::dispatch
                    && expected[31].kind == stage::dispatch
                    && expected[32].kind == stage::combine
                    && expected[63].kind == stage::combine,
                "balanced-q1 native D0/C0 golden mismatch");
    } else if (std::string(test.name) == "balanced-q2") {
        require(expected.size() == 64
                    && expected[0].kind == stage::dispatch
                    && expected[0].macrobatch == 1
                    && expected[15].kind == stage::dispatch
                    && expected[16].kind == stage::combine
                    && expected[16].macrobatch == 1
                    && expected[17].kind == stage::dispatch
                    && expected[17].macrobatch == 0
                    && expected[46].kind == stage::combine
                    && expected[47].kind == stage::dispatch
                    && expected[48].kind == stage::combine
                    && expected[48].macrobatch == 0,
                "balanced-q2 native D1/C1-D0/C0 golden mismatch");
    }
}

void check_host_case(const probe_case &test) {
    const terminal::logical_shape shape = terminal::make_logical_shape(
        test.num_tokens, test.schedule_capacity, test.minibatch_rows,
        test.macrobatch_rows);
    require(shape.valid, std::string(test.name) + ": invalid shape");
    require(shape.num_macrobatches == test.expected_q,
            std::string(test.name) + ": Q mismatch");
    check_counter_contract(test, shape);
    check_case_goldens(test, shape);
    check_communication_decode(test, shape);

    int64_t next_ordinal = 0;
    std::vector<int> m_tile_ranges(test.num_tokens / terminal::M_TILE, 0);
    for (int j = 0; j < shape.num_global_minibatches; ++j) {
        const auto range = terminal::decode_ordered_minibatch(shape, j);
        require(range.valid, std::string(test.name) + ": invalid range");
        require(range.all.begin == next_ordinal,
                std::string(test.name) + ": range gap or overlap");
        const auto gate = terminal::stage_range(
            range, terminal::logical_stage::gate);
        const auto up = terminal::stage_range(
            range, terminal::logical_stage::up);
        const auto activation = terminal::stage_range(
            range, terminal::logical_stage::activation);
        const auto w2 = terminal::stage_range(
            range, terminal::logical_stage::w2);
        require(gate.begin == range.all.begin && gate.end == up.begin
                    && up.end == activation.begin
                    && activation.end == w2.begin
                    && w2.end == range.all.end,
                std::string(test.name) + ": stage range gap or overlap");
        require(gate.end - gate.begin
                    == static_cast<int64_t>(terminal::W13_CLUSTER_TASKS)
                        * range.active_m_tiles
                    && up.end - up.begin == gate.end - gate.begin
                    && activation.end - activation.begin
                        == range.active_m_tiles
                    && w2.end - w2.begin
                        == static_cast<int64_t>(terminal::W2_CLUSTER_TASKS)
                            * range.active_m_tiles,
                std::string(test.name) + ": stage range cardinality");
        check_stage_decode_interlock(
            test, shape, range, terminal::logical_stage::gate, gate);
        check_stage_decode_interlock(
            test, shape, range, terminal::logical_stage::up, up);
        check_stage_decode_interlock(
            test, shape, range, terminal::logical_stage::activation,
            activation);
        check_stage_decode_interlock(
            test, shape, range, terminal::logical_stage::w2, w2);
        for (int r = 0; r < range.active_m_tiles; ++r) {
            const int m = range.first_m_tile + r;
            require(m >= 0 && m < static_cast<int>(m_tile_ranges.size()),
                    std::string(test.name) + ": inactive M64 range");
            ++m_tile_ranges[m];
        }
        next_ordinal = range.all.end;
    }
    require(next_ordinal == shape.total_tasks,
            std::string(test.name) + ": final range endpoint");
    require(std::all_of(m_tile_ranges.begin(), m_tile_ranges.end(),
                        [](int visits) { return visits == 1; }),
            std::string(test.name) + ": M64 ranges are not exactly once");

    const int m_tiles = test.num_tokens / terminal::M_TILE;
    std::vector<int> cluster_visits(
        static_cast<size_t>(shape.total_tasks), 0);
    std::vector<int> work_visits(
        static_cast<size_t>(m_tiles * terminal::N64_SUBTILES_PER_M64), 0);
    std::vector<int> gate_up_arrivals(
        static_cast<size_t>(m_tiles * terminal::W13_N_TILES), 0);
    std::vector<int> hidden_arrivals(static_cast<size_t>(m_tiles), 0);
    std::vector<int> y_arrivals(static_cast<size_t>(m_tiles), 0);
    for (int64_t ordinal = 0; ordinal < shape.total_tasks; ++ordinal) {
        const auto coordinate =
            terminal::decode_logical_cursor(shape, ordinal);
        require(coordinate.valid,
                std::string(test.name) + ": invalid active coordinate");
        require(coordinate.global_m >= 0
                    && coordinate.global_m
                        < test.num_tokens / terminal::M_TILE,
                std::string(test.name) + ": inactive coordinate emitted");
        const int cluster_key = canonical_cluster_task(coordinate, m_tiles);
        require(cluster_key >= 0
                    && cluster_key < static_cast<int>(cluster_visits.size()),
                std::string(test.name) + ": invalid cluster task key");
        ++cluster_visits[cluster_key];

        const auto x_index = terminal::counter_index(
            terminal::ready_counter::x_routed, coordinate.global_m);
        require(x_index >= 0
                    && x_index < terminal::counter_cardinality(
                        terminal::ready_counter::x_routed,
                        test.schedule_capacity),
                std::string(test.name) + ": x counter out of bounds");
        if (coordinate.stage == terminal::logical_stage::gate
                || coordinate.stage == terminal::logical_stage::up) {
            for (int rank = 0; rank < terminal::CLUSTER_CTAS; ++rank) {
                const int n64 = terminal::n64_for_cta(coordinate, rank);
                const int work_key = canonical_n64_subtile(
                    coordinate, m_tiles, n64);
                require(n64 == terminal::CLUSTER_CTAS * coordinate.n128 + rank
                            && work_key >= 0
                            && work_key < static_cast<int>(work_visits.size()),
                        std::string(test.name) + ": invalid W13 CTA expansion");
                ++work_visits[work_key];
            }
            const auto gu_index = terminal::counter_index(
                terminal::ready_counter::gate_up_tile,
                coordinate.global_m, coordinate.n128);
            require(gu_index >= 0
                        && gu_index < static_cast<int64_t>(
                            gate_up_arrivals.size()),
                    std::string(test.name)
                        + ": gate/up counter out of bounds");
            gate_up_arrivals[static_cast<size_t>(gu_index)] +=
                terminal::counter_arrival_delta_per_entry(
                    terminal::ready_counter::gate_up_tile,
                    coordinate.stage);
        } else if (coordinate.stage == terminal::logical_stage::activation) {
            hidden_arrivals[coordinate.global_m] +=
                terminal::counter_arrivals_per_cluster_task(
                    terminal::ready_counter::hidden_row_block,
                    coordinate.stage);
        } else {
            for (int rank = 0; rank < terminal::CLUSTER_CTAS; ++rank) {
                const int n64 = terminal::n64_for_cta(coordinate, rank);
                const int work_key = canonical_n64_subtile(
                    coordinate, m_tiles, n64);
                require(n64 == terminal::CLUSTER_CTAS * coordinate.n128 + rank
                            && work_key >= 0
                            && work_key < static_cast<int>(work_visits.size()),
                        std::string(test.name) + ": invalid W2 CTA expansion");
                ++work_visits[work_key];
            }
            y_arrivals[coordinate.global_m] +=
                terminal::counter_arrivals_per_cluster_task(
                    terminal::ready_counter::y_routed, coordinate.stage);
        }
    }
    require(std::all_of(cluster_visits.begin(), cluster_visits.end(),
                        [](int count) { return count == 1; }),
            std::string(test.name) + ": cluster task is not exactly once");
    require(std::all_of(work_visits.begin(), work_visits.end(),
                        [](int count) { return count == 1; }),
            std::string(test.name) + ": expanded N64 subtile is not exactly once");
    require(std::all_of(gate_up_arrivals.begin(), gate_up_arrivals.end(),
                        [](int count) { return count == 2; })
                && std::all_of(hidden_arrivals.begin(), hidden_arrivals.end(),
                               [](int count) { return count == 1; })
                && std::all_of(y_arrivals.begin(), y_arrivals.end(),
                               [](int count) { return count == 32; }),
            std::string(test.name) + ": observed counter arrivals");
    require(std::accumulate(gate_up_arrivals.begin(),
                            gate_up_arrivals.end(), int64_t{0})
                    == terminal::counter_cardinality(
                           terminal::ready_counter::gate_up_tile,
                           test.num_tokens)
                        * terminal::counter_expected_arrivals(
                            terminal::ready_counter::gate_up_tile)
                && std::accumulate(hidden_arrivals.begin(),
                                   hidden_arrivals.end(), int64_t{0})
                    == terminal::counter_cardinality(
                           terminal::ready_counter::hidden_row_block,
                           test.num_tokens)
                        * terminal::counter_expected_arrivals(
                            terminal::ready_counter::hidden_row_block)
                && std::accumulate(y_arrivals.begin(), y_arrivals.end(),
                                   int64_t{0})
                    == terminal::counter_cardinality(
                           terminal::ready_counter::y_routed,
                           test.num_tokens)
                        * terminal::counter_expected_arrivals(
                            terminal::ready_counter::y_routed),
            std::string(test.name) + ": counter arrival aggregate");
    require(!terminal::decode_logical_cursor(shape, -1).valid
                && !terminal::decode_logical_cursor(
                    shape, shape.total_tasks).valid,
            std::string(test.name) + ": cursor bounds");

    std::cout << "TERMINAL_LOGICAL_DECODER_HOST|case=" << test.name
              << "|nt=" << test.num_tokens
              << "|Q=" << shape.num_macrobatches
              << "|G=" << shape.num_global_minibatches
              << "|tasks=" << shape.total_tasks << "|result=PASS\n";
}

#if defined(__CUDACC__)

void cuda_check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
}

__global__ void device_decode_probe(terminal::logical_shape shape,
                                    int *cluster_visits, int *work_visits,
                                    int *gate_up_arrivals,
                                    int *hidden_arrivals, int *y_arrivals,
                                    int *errors) {
    const int64_t ordinal = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const int m_tiles = shape.num_tokens / terminal::M_TILE;
    if (ordinal == 0) {
        if (shape.total_tasks
                != static_cast<int64_t>(terminal::TASKS_PER_M64) * m_tiles)
            atomicAdd(errors, 1);
        for (int raw = 0;
             raw < static_cast<int>(terminal::ready_counter::count); ++raw) {
            const auto counter = static_cast<terminal::ready_counter>(raw);
            const int64_t cardinality = terminal::counter_cardinality(
                counter, shape.schedule_capacity);
            const int capacity_m_tiles =
                shape.schedule_capacity / terminal::M_TILE;
            const int last_n =
                counter == terminal::ready_counter::gate_up_tile
                ? terminal::W13_N_TILES - 1
                : 0;
            if (cardinality <= 0
                    || terminal::counter_index(counter, 0, 0) != 0
                    || terminal::counter_index(
                        counter, capacity_m_tiles - 1, last_n)
                        != cardinality - 1)
                atomicAdd(errors, 1);
        }
    }
    if (ordinal >= shape.total_tasks)
        return;
    const auto coordinate = terminal::decode_logical_cursor(shape, ordinal);
    const int key = canonical_cluster_task(coordinate, m_tiles);
    if (!coordinate.valid || key < 0 || key >= shape.total_tasks) {
        atomicAdd(errors, 1);
        return;
    }
    atomicAdd(cluster_visits + key, 1);
    if (coordinate.stage == terminal::logical_stage::activation) {
        atomicAdd(hidden_arrivals + coordinate.global_m,
                  terminal::counter_arrivals_per_cluster_task(
                      terminal::ready_counter::hidden_row_block,
                      coordinate.stage));
        return;
    }

    for (int rank = 0; rank < terminal::CLUSTER_CTAS; ++rank) {
        const int n64 = terminal::n64_for_cta(coordinate, rank);
        const int work_key = canonical_n64_subtile(coordinate, m_tiles, n64);
        if (n64 != terminal::CLUSTER_CTAS * coordinate.n128 + rank
                || work_key < 0
                || work_key >= m_tiles * terminal::N64_SUBTILES_PER_M64) {
            atomicAdd(errors, 1);
            continue;
        }
        atomicAdd(work_visits + work_key, 1);
    }
    if (coordinate.stage == terminal::logical_stage::gate
            || coordinate.stage == terminal::logical_stage::up) {
        const int64_t counter = terminal::counter_index(
            terminal::ready_counter::gate_up_tile,
            coordinate.global_m, coordinate.n128);
        if (counter < 0
                || counter >= static_cast<int64_t>(m_tiles)
                    * terminal::W13_N_TILES)
            atomicAdd(errors, 1);
        else
            atomicAdd(gate_up_arrivals + counter,
                      terminal::counter_arrival_delta_per_entry(
                          terminal::ready_counter::gate_up_tile,
                          coordinate.stage));
    }
    if (coordinate.stage == terminal::logical_stage::w2)
        atomicAdd(y_arrivals + coordinate.global_m,
                  terminal::counter_arrivals_per_cluster_task(
                      terminal::ready_counter::y_routed,
                      coordinate.stage));
}

void check_device_case(const probe_case &test) {
    const auto shape = terminal::make_logical_shape(
        test.num_tokens, test.schedule_capacity, test.minibatch_rows,
        test.macrobatch_rows);
    const size_t visit_count = static_cast<size_t>(
        std::max<int64_t>(shape.total_tasks, 1));
    const size_t work_count = static_cast<size_t>(std::max(
        test.num_tokens / terminal::M_TILE
            * terminal::N64_SUBTILES_PER_M64,
        1));
    const size_t gate_up_count = static_cast<size_t>(std::max(
        test.num_tokens / terminal::M_TILE * terminal::W13_N_TILES, 1));
    const size_t m_count = static_cast<size_t>(std::max(
        test.num_tokens / terminal::M_TILE, 1));
    int *device_cluster_visits = nullptr;
    int *device_work_visits = nullptr;
    int *device_gate_up_arrivals = nullptr;
    int *device_hidden_arrivals = nullptr;
    int *device_y_arrivals = nullptr;
    int *device_errors = nullptr;
    cuda_check(cudaMalloc(&device_cluster_visits, visit_count * sizeof(int)),
               "cudaMalloc cluster visits");
    cuda_check(cudaMalloc(&device_work_visits, work_count * sizeof(int)),
               "cudaMalloc work visits");
    cuda_check(cudaMalloc(
                   &device_gate_up_arrivals, gate_up_count * sizeof(int)),
               "cudaMalloc gate/up arrivals");
    cuda_check(cudaMalloc(&device_hidden_arrivals, m_count * sizeof(int)),
               "cudaMalloc hidden arrivals");
    cuda_check(cudaMalloc(&device_y_arrivals, m_count * sizeof(int)),
               "cudaMalloc y arrivals");
    cuda_check(cudaMalloc(&device_errors, sizeof(int)), "cudaMalloc errors");
    cuda_check(cudaMemset(
                   device_cluster_visits, 0, visit_count * sizeof(int)),
               "cudaMemset cluster visits");
    cuda_check(cudaMemset(device_work_visits, 0, work_count * sizeof(int)),
               "cudaMemset work visits");
    cuda_check(cudaMemset(
                   device_gate_up_arrivals, 0, gate_up_count * sizeof(int)),
               "cudaMemset gate/up arrivals");
    cuda_check(cudaMemset(device_hidden_arrivals, 0, m_count * sizeof(int)),
               "cudaMemset hidden arrivals");
    cuda_check(cudaMemset(device_y_arrivals, 0, m_count * sizeof(int)),
               "cudaMemset y arrivals");
    cuda_check(cudaMemset(device_errors, 0, sizeof(int)),
               "cudaMemset errors");
    const int blocks = static_cast<int>(
        (std::max<int64_t>(shape.total_tasks, 1) + 255) / 256);
    device_decode_probe<<<blocks, 256>>>(
        shape, device_cluster_visits, device_work_visits,
        device_gate_up_arrivals, device_hidden_arrivals, device_y_arrivals,
        device_errors);
    cuda_check(cudaGetLastError(), "device decoder launch");
    std::vector<int> cluster_visits(visit_count);
    std::vector<int> work_visits(work_count);
    std::vector<int> gate_up_arrivals(gate_up_count);
    std::vector<int> hidden_arrivals(m_count);
    std::vector<int> y_arrivals(m_count);
    int errors = 0;
    cuda_check(cudaMemcpy(cluster_visits.data(), device_cluster_visits,
                          visit_count * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy cluster visits");
    cuda_check(cudaMemcpy(work_visits.data(), device_work_visits,
                          work_count * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy work visits");
    cuda_check(cudaMemcpy(gate_up_arrivals.data(), device_gate_up_arrivals,
                          gate_up_count * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy gate/up arrivals");
    cuda_check(cudaMemcpy(hidden_arrivals.data(), device_hidden_arrivals,
                          m_count * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy hidden arrivals");
    cuda_check(cudaMemcpy(y_arrivals.data(), device_y_arrivals,
                          m_count * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy y arrivals");
    cuda_check(cudaMemcpy(&errors, device_errors, sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy errors");
    cuda_check(cudaFree(device_cluster_visits), "cudaFree cluster visits");
    cuda_check(cudaFree(device_work_visits), "cudaFree work visits");
    cuda_check(cudaFree(device_gate_up_arrivals),
               "cudaFree gate/up arrivals");
    cuda_check(cudaFree(device_hidden_arrivals),
               "cudaFree hidden arrivals");
    cuda_check(cudaFree(device_y_arrivals), "cudaFree y arrivals");
    cuda_check(cudaFree(device_errors), "cudaFree errors");
    require(errors == 0, std::string(test.name) + ": device helper error");
    require(std::all_of(cluster_visits.begin(),
                        cluster_visits.begin()
                            + static_cast<size_t>(shape.total_tasks),
                        [](int count) { return count == 1; }),
            std::string(test.name)
                + ": device cluster task is not exactly once");
    const size_t active_work = static_cast<size_t>(
        test.num_tokens / terminal::M_TILE
        * terminal::N64_SUBTILES_PER_M64);
    const size_t active_gate_up = static_cast<size_t>(
        test.num_tokens / terminal::M_TILE * terminal::W13_N_TILES);
    const size_t active_m = static_cast<size_t>(
        test.num_tokens / terminal::M_TILE);
    require(std::all_of(work_visits.begin(),
                        work_visits.begin() + active_work,
                        [](int count) { return count == 1; })
                && std::all_of(gate_up_arrivals.begin(),
                               gate_up_arrivals.begin() + active_gate_up,
                               [](int count) { return count == 2; })
                && std::all_of(hidden_arrivals.begin(),
                               hidden_arrivals.begin() + active_m,
                               [](int count) { return count == 1; })
                && std::all_of(y_arrivals.begin(),
                               y_arrivals.begin() + active_m,
                               [](int count) { return count == 32; }),
            std::string(test.name)
                + ": device expanded work/counter arrivals");
    std::cout << "TERMINAL_LOGICAL_DECODER_DEVICE|case=" << test.name
              << "|tasks=" << shape.total_tasks << "|result=PASS\n";
}

#endif

}  // namespace

int main(int argc, char **argv) {
    try {
        bool run_device = false;
        for (int index = 1; index < argc; ++index) {
            if (std::string(argv[index]) == "--device")
                run_device = true;
            else
                fail(std::string("unknown argument: ") + argv[index]);
        }
        for (const auto &test : CASES)
            check_host_case(test);
#if defined(__CUDACC__)
        if (run_device) {
            for (const auto &test : CASES)
                check_device_case(test);
        }
#else
        if (run_device)
            fail("--device requires an nvcc build");
#endif
        std::cout << "TERMINAL_LOGICAL_DECODER_PROBE|result=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "TERMINAL_LOGICAL_DECODER_PROBE|result=FAIL|error="
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
