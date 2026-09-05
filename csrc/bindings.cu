#include "mok_megakernel.cuh"
#include "mxfp8.cuh"
#include "scheduler.cuh"
#include "sm90_fp8_block_routed.cuh"
#include "sm90_fp8_block_worker_test.cuh"
#include "sm90_fp8_block_warprole_gemm.cuh"
#include "sm90_fp8_block_warprole_epilogue.cuh"
#include "sm90_fp8_block_warprole_comm.cuh"
#include "sm90_fp8_block_warprole.cuh"
#include "utils.cuh"
#include "sm90_fp8_block_route_fused.cuh"
#include "sm90_fp8_block_dispatch_gemm.cuh"
#include "sm90_fp8_block_gemm_combine.cuh"
#include "sm90_fp8_block_terminal_entry.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_gather_top_experts", &utils::all_gather_top_experts::entrypoint, "",
          pybind11::arg("top_experts"), pybind11::arg("all_gather_top_experts_buffer"),
          pybind11::arg("all_gather_top_experts_buffer_multicast_ptr"), pybind11::arg("rank"), pybind11::arg("chunk_bytes"));
    m.def("barrier_all", &utils::barrier_all::entrypoint, "",
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"), pybind11::arg("target"));
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"),
          pybind11::arg("schedule_capacity"), pybind11::arg("rank"),
          pybind11::arg("expert_padding") = 256);
    m.def("mxfp8_quantize", &mxfp8_quantize::mxfp8_quantize_entrypoint, "",
          pybind11::arg("x_bf16"),
          pybind11::arg("return_normal"), pybind11::arg("return_transposed"));
    m.def("dispatch_mlp_swiglu_combine_fwd_mxfp8", &dispatch_mlp_swiglu_combine_fwd_mxfp8, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"), pybind11::arg("w_routed_down_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_bwd_mxfp8", &dispatch_mlp_swiglu_combine_bwd_mxfp8, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate_T"), pybind11::arg("w_routed_gate_T_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up_T"), pybind11::arg("w_routed_up_T_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down_T"), pybind11::arg("w_routed_down_T_sc"),
          pybind11::arg("x_fp8_t_routed"), pybind11::arg("x_sc_t_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_fp8_routed"), pybind11::arg("gate_sc_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_fp8_routed"), pybind11::arg("up_sc_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_fp8_t_routed"), pybind11::arg("hidden_sc_t_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_fwd_bf16", &dispatch_mlp_swiglu_combine_fwd_bf16, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_bwd_bf16", &dispatch_mlp_swiglu_combine_bwd_bf16, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("x_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("swiglu_limit"),
          pybind11::arg("num_comm_sms"), pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    #if defined(KITTENS_SM90)
    m.def("sm90_worker_test", &mok_sm90::wtest::entry, "");
    m.def("sm90_fp8_block_test", &mok_sm90::fp8_block_test::entry, "");
    m.def("sm90_fp8_block_grouped_test",
          &mok_sm90::fp8_block_test::grouped::entry, "");
    m.def("sm90_fp8_block_grouped_pipelined_test",
          &mok_sm90::fp8_block_test::grouped::entry_pipelined, "");
    m.def("sm90_fp8_block_grouped_out_test",
          &mok_sm90::fp8_block_test::grouped::entry_out, "");
    m.def("sm90_fp8_block_grouped_pipelined_out_test",
          &mok_sm90::fp8_block_test::grouped::entry_pipelined_out, "");
    m.def("fp8_block_grouped_pipelined_out",
          &mok_sm90::fp8_block_test::grouped::entry_pipelined_out, "",
          pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("masked_m"), pybind11::arg("output"));
    m.def("fp8_block_grouped_contiguous_out",
          &mok_sm90::fp8_block_test::contiguous::entry_pipelined_out, "",
          pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("m_indices"), pybind11::arg("output"));
    m.def("fp8_block_grouped_contiguous_dynamic_out",
          &mok_sm90::fp8_block_test::contiguous::entry_pipelined_dynamic_out,
          "", pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("output"));
    m.def("fp8_block_warprole_gemm_c1s6_out",
          &mok_sm90::warprole::gemm::standalone::entry_out<1, 6, 1>,
          "", pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("output"));
    m.def("fp8_block_warprole_gemm_c2s4_out",
          &mok_sm90::warprole::gemm::standalone::entry_out<2, 4, 1>,
          "", pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("output"));
    // two-CTA-per-SM retreat form disabled: ptxas refuses the 80-register launch budget
    // (C7602) for the N128 WGMMA even with setmaxnreg; see benchmarks/warprole/README.md.
    //     m.def("fp8_block_warprole_gemm_c1s3x2_out",
    //           &mok_sm90::warprole::gemm::standalone::entry_out<1, 3, 2>,
    //           "", pybind11::arg("input"), pybind11::arg("weight"),
    //           pybind11::arg("input_scale"), pybind11::arg("weight_scale"),
    //           pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
    //           pybind11::arg("output"));
    m.def("fp8_block_warprole_w13_c1s6_out",
          &mok_sm90::warprole::epilogue::standalone::entry_w13_out<1, 6, 1>,
          "", pybind11::arg("input"), pybind11::arg("input_scale"),
          pybind11::arg("w13_interleaved"), pybind11::arg("w13_interleaved_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("hidden"), pybind11::arg("hidden_scale"),
          pybind11::arg("swiglu_limit"));
    m.def("fp8_block_warprole_w13_c2s4_out",
          &mok_sm90::warprole::epilogue::standalone::entry_w13_out<2, 4, 1>,
          "", pybind11::arg("input"), pybind11::arg("input_scale"),
          pybind11::arg("w13_interleaved"), pybind11::arg("w13_interleaved_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("hidden"), pybind11::arg("hidden_scale"),
          pybind11::arg("swiglu_limit"));
    // two-CTA-per-SM retreat form disabled: ptxas refuses the 80-register launch budget
    // (C7602) for the N128 WGMMA even with setmaxnreg; see benchmarks/warprole/README.md.
    //     m.def("fp8_block_warprole_w13_c1s3x2_out",
    //           &mok_sm90::warprole::epilogue::standalone::entry_w13_out<1, 3, 2>,
    //           "", pybind11::arg("input"), pybind11::arg("input_scale"),
    //           pybind11::arg("w13_interleaved"), pybind11::arg("w13_interleaved_scale"),
    //           pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
    //           pybind11::arg("hidden"), pybind11::arg("hidden_scale"),
    //           pybind11::arg("swiglu_limit"));
    m.def("fp8_block_warprole_comm_bench_out",
          &mok_sm90::warprole::comm::bench::entry_comm_bench_out, "",
          pybind11::arg("mode"), pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("x_scale"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"),
          pybind11::arg("m_indices"), pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"), pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"),
          pybind11::arg("routed_y"), pybind11::arg("combine_ptrs"),
          pybind11::arg("push_done_ptrs"), pybind11::arg("ep_rank"),
          pybind11::arg("x_ready"), pybind11::arg("y_ready"),
          pybind11::arg("push_done_local"));
    m.def("fp8_block_warprole_comm_gemm_bench_out",
          &mok_sm90::warprole::comm::bench::entry_comm_gemm_bench_out, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("x_scale"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"),
          pybind11::arg("m_indices"), pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"), pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"),
          pybind11::arg("routed_y"), pybind11::arg("combine_ptrs"),
          pybind11::arg("push_done_ptrs"), pybind11::arg("ep_rank"),
          pybind11::arg("x_ready"), pybind11::arg("y_ready"),
          pybind11::arg("push_done_local"),
          pybind11::arg("gemm_input"), pybind11::arg("gemm_weight"),
          pybind11::arg("gemm_input_scale"), pybind11::arg("gemm_weight_scale"),
          pybind11::arg("gemm_m_indices"), pybind11::arg("gemm_num_tokens"),
          pybind11::arg("gemm_output"));
    m.def("fp8_block_warprole_probe_read", &mok_sm90::warprole::fused::probe_read,
          "Benchmark-only: the [num_sms, 8] int64 globaltimer stamps of the last fused call made with "
          "MOK_WARPROLE_PROBE=1 (entry, rank barrier, dispatch, first W13, last task, combine, push_done, reduce).");
    m.def("fp8_block_warprole_prepare_out", &mok_sm90::warprole::fused::entry_prepare_out, "",
          pybind11::arg("x_ready"), pybind11::arg("hidden_ready"), pybind11::arg("y_ready"),
          pybind11::arg("push_done_local"), pybind11::arg("push_done"), pybind11::arg("input_expected_scratch"));
    m.def("fp8_block_warprole_c1s6_out", &mok_sm90::warprole::fused::entry_out<1, 6>, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"), pybind11::arg("x_scale"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"), pybind11::arg("m_indices"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"), pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"),
          pybind11::arg("w13i"), pybind11::arg("w13i_scale"), pybind11::arg("w2"), pybind11::arg("w2_scale"),
          pybind11::arg("hidden"), pybind11::arg("hidden_scale"), pybind11::arg("routed_y"),
          pybind11::arg("combine_ptrs"), pybind11::arg("combine_local"), pybind11::arg("weights"), pybind11::arg("topk_ids"),
          pybind11::arg("output"), pybind11::arg("push_done_ptrs"), pybind11::arg("ep_rank"),
          pybind11::arg("x_ready"), pybind11::arg("hidden_ready"), pybind11::arg("y_ready"), pybind11::arg("push_done_local"),
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_target"), pybind11::arg("barrier_multicast_ptr"),
          pybind11::arg("input_expected_scratch"), pybind11::arg("trap_record_ptr"), pybind11::arg("swiglu_limit"),
          pybind11::arg("spin_limit"));
    m.def("fp8_block_warprole_c2s4_out", &mok_sm90::warprole::fused::entry_out<2, 4>, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"), pybind11::arg("x_scale"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"), pybind11::arg("m_indices"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"), pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"),
          pybind11::arg("w13i"), pybind11::arg("w13i_scale"), pybind11::arg("w2"), pybind11::arg("w2_scale"),
          pybind11::arg("hidden"), pybind11::arg("hidden_scale"), pybind11::arg("routed_y"),
          pybind11::arg("combine_ptrs"), pybind11::arg("combine_local"), pybind11::arg("weights"), pybind11::arg("topk_ids"),
          pybind11::arg("output"), pybind11::arg("push_done_ptrs"), pybind11::arg("ep_rank"),
          pybind11::arg("x_ready"), pybind11::arg("hidden_ready"), pybind11::arg("y_ready"), pybind11::arg("push_done_local"),
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_target"), pybind11::arg("barrier_multicast_ptr"),
          pybind11::arg("input_expected_scratch"), pybind11::arg("trap_record_ptr"), pybind11::arg("swiglu_limit"),
          pybind11::arg("spin_limit"));
    m.def("fp8_block_build_schedule_out",
          &mok_sm90::fp8_block_route_fused::build_schedule_out, "",
          pybind11::arg("top_experts"),
          pybind11::arg("all_gather_buffer"),
          pybind11::arg("all_gather_multicast_ptr"), pybind11::arg("rank"),
          pybind11::arg("chunk_bytes"), pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"),
          pybind11::arg("tokens_per_expert_and_peer"),
          pybind11::arg("expert_padding"));
    m.def("fp8_block_routed_dispatch_out",
          &mok_sm90::fp8_block_routed::dispatch_out, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("x_scale"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"));
    m.def("fp8_block_routed_combine_out",
          [](const at::Tensor &routed_y, const at::Tensor &combine_buffer,
             const std::vector<int64_t> &combine_buffer_ptrs,
             const at::Tensor &schedule_peer_rank,
             const at::Tensor &schedule_peer_token_idx,
             const at::Tensor &num_tokens, int64_t topk) {
              mok_sm90::fp8_block_routed::combine_out(
                  routed_y, combine_buffer, combine_buffer_ptrs,
                  schedule_peer_rank, schedule_peer_token_idx, num_tokens,
                  topk);
          },
          "", pybind11::arg("routed_y"), pybind11::arg("combine_buffer"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("topk"));
    m.def("fp8_block_routed_dispatch_copy_out",
          &mok_sm90::fp8_block_route_fused::dispatch_copy_out, "",
          pybind11::arg("x"), pybind11::arg("x_buffer"),
          pybind11::arg("x_buffer_ptrs"), pybind11::arg("x_scale"),
          pybind11::arg("x_scale_buffer"),
          pybind11::arg("x_scale_buffer_ptrs"),
          pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"), pybind11::arg("routed_x"),
          pybind11::arg("routed_x_scale"), pybind11::arg("m_indices"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"));
    m.def("fp8_block_routed_combine_reduce_out",
          &mok_sm90::fp8_block_route_fused::combine_reduce_out, "",
          pybind11::arg("routed_y"), pybind11::arg("combine_buffer"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("topk_weights"),
          pybind11::arg("output"), pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"), pybind11::arg("topk"),
          pybind11::arg("combine_precleared") = false,
          pybind11::arg("combine_completion") = pybind11::none(),
          pybind11::arg("barrier_expected_scratch") = pybind11::none());
    m.def("fp8_block_dispatch_gemm_fused_out",
          &mok_sm90::fp8_block_dispatch_gemm::entry_out, "",
          pybind11::arg("x_buffer"), pybind11::arg("x_ptrs"),
          pybind11::arg("x_scale_buffer"), pybind11::arg("x_scale_ptrs"),
          pybind11::arg("routed_x"), pybind11::arg("routed_x_scale"),
          pybind11::arg("m_indices"), pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("barrier_target"),
          pybind11::arg("input_expected_scratch"),
          pybind11::arg("tile_ready"), pybind11::arg("B"),
          pybind11::arg("B_scale"), pybind11::arg("D"),
          pybind11::arg("copy_clusters"), pybind11::arg("ep_rank"),
          pybind11::arg("ticket_counter"), pybind11::arg("worker_ticket"),
          pybind11::arg("trap_record_ptr"),
          pybind11::arg("forced_worker_clusters"),
          pybind11::arg("delay_ticket0_cycles"),
          pybind11::arg("spin_trap_iters"),
          pybind11::arg("ticket_visit"),
          pybind11::arg("record_visits"));
    m.def("fp8_block_gemm_combine_fused_out",
          &mok_sm90::fp8_block_gemm_combine::entry_out, "",
          pybind11::arg("down_input"), pybind11::arg("down_input_scale"),
          pybind11::arg("weight"), pybind11::arg("weight_scale"),
          pybind11::arg("m_indices"), pybind11::arg("num_tokens"),
          pybind11::arg("routed_y"), pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("combine_buffer"),
          pybind11::arg("combine_buffer_ptrs"), pybind11::arg("topk"),
          pybind11::arg("down_ready"), pybind11::arg("combine_completion"),
          pybind11::arg("barrier_target"),
          pybind11::arg("barrier_expected_scratch"),
          pybind11::arg("barrier_buffer_multicast_ptr"));
    m.def("routed_epilogue_fused_out",
          [](const at::Tensor &combine_buffer, const at::Tensor &topk_weights,
             const at::Tensor &output, const at::Tensor &barrier_buffer,
             const at::Tensor &barrier_expected_scratch,
             const at::Tensor &in_use, const at::Tensor &epilogue_done,
             int64_t trap_record_ptr, int64_t ep_rank, int64_t do_release) {
              // Guard onto the workspace's device before resolving the
              // host-mapped record or launching.
              c10::cuda::CUDAGuard guard(output.device());
              utils::routed_epilogue_out(
                  combine_buffer, topk_weights, output,
                  reinterpret_cast<const unsigned int *>(
                      barrier_buffer.data_ptr<int>()),
                  reinterpret_cast<const unsigned int *>(
                      barrier_expected_scratch.data_ptr<int>()),
                  do_release != 0
                      ? reinterpret_cast<unsigned int *>(
                            in_use.data_ptr<int>())
                      : nullptr,
                  do_release != 0
                      ? reinterpret_cast<unsigned int *>(
                            epilogue_done.data_ptr<int>())
                      : nullptr,
                  utils::mok_resolve_trap_record(trap_record_ptr),
                  static_cast<int>(ep_rank));
          },
          "", pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"),
          pybind11::arg("output"), pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_expected_scratch"),
          pybind11::arg("in_use"), pybind11::arg("epilogue_done"),
          pybind11::arg("trap_record_ptr"), pybind11::arg("ep_rank"),
          pybind11::arg("do_release"));
    m.def("mok_workspace_lease_acquire", &utils::workspace_lease_acquire, "",
          pybind11::arg("in_use"), pybind11::arg("trap_record_ptr"),
          pybind11::arg("ep_rank"));
    m.def("mok_workspace_lease_release", &utils::workspace_lease_release, "",
          pybind11::arg("in_use"));
    m.def("fp8_block_dispatch_gemm_prewarm",
          &mok_sm90::fp8_block_dispatch_gemm::entry_prewarm, "",
          pybind11::arg("device_index"));
    m.def("fp8_block_megakernel_prepare_out",
          &mok_sm90::fp8_block_terminal_entry::entry_prepare_out, "",
          pybind11::arg("topk_ids"), pybind11::arg("route_ready"),
          pybind11::arg("x_routed_ready"),
          pybind11::arg("gate_up_tile_ready"),
          pybind11::arg("hidden_row_block_ready"),
          pybind11::arg("y_routed_ready"),
          pybind11::arg("y_routed_done"),
          pybind11::arg("epilogue_claim"),
          pybind11::arg("next_logical_cluster"),
          pybind11::arg("next_reduce_probe"),
          pybind11::arg("role_cursor"), pybind11::arg("cluster_role"),
          pybind11::arg("dispatch_tile_cursor"),
          pybind11::arg("dispatch_tiles_done"),
          pybind11::arg("push_tile_cursor"),
          pybind11::arg("worker_ticket"), pybind11::arg("comm_owner"),
          pybind11::arg("comm_worker_ticket"),
          pybind11::arg("producer_done"),
          pybind11::arg("comm_closed"), pybind11::arg("push_done"),
          pybind11::arg("reduce_done"), pybind11::arg("terminate"),
          pybind11::arg("epilogue_done"),
          pybind11::arg("input_expected_scratch"));
    m.def("fp8_block_megakernel_out",
          &mok_sm90::fp8_block_terminal_entry::entry_out, "",
          pybind11::arg("x_buffer"), pybind11::arg("x_ptrs"),
          pybind11::arg("x_scale_buffer"),
          pybind11::arg("x_scale_ptrs"), pybind11::arg("routed_x"),
          pybind11::arg("routed_x_scale"), pybind11::arg("m_indices"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("w13"),
          pybind11::arg("w13_scale"), pybind11::arg("gate_up"),
          pybind11::arg("down_input"),
          pybind11::arg("down_input_scale"), pybind11::arg("w2"),
          pybind11::arg("w2_scale"), pybind11::arg("routed_y"),
          pybind11::arg("combine_buffer"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("route_ready"),
          pybind11::arg("route_ready_ptrs"),
          pybind11::arg("topk_weights"), pybind11::arg("topk_ids"),
          pybind11::arg("output"), pybind11::arg("x_routed_ready"),
          pybind11::arg("gate_up_tile_ready"),
          pybind11::arg("hidden_row_block_ready"),
          pybind11::arg("y_routed_ready"),
          pybind11::arg("y_routed_done"),
          pybind11::arg("epilogue_claim"),
          pybind11::arg("next_logical_cluster"),
          pybind11::arg("next_reduce_probe"),
          pybind11::arg("role_cursor"), pybind11::arg("cluster_role"),
          pybind11::arg("dispatch_tile_cursor"),
          pybind11::arg("dispatch_tiles_done"),
          pybind11::arg("push_tile_cursor"),
          pybind11::arg("worker_ticket"), pybind11::arg("comm_owner"),
          pybind11::arg("comm_worker_ticket"),
          pybind11::arg("producer_done"),
          pybind11::arg("comm_closed"), pybind11::arg("push_done"),
          pybind11::arg("reduce_done"), pybind11::arg("terminate"),
          pybind11::arg("epilogue_done"), pybind11::arg("in_use"),
          pybind11::arg("barrier_buffer"),
          pybind11::arg("barrier_target"),
          pybind11::arg("input_expected_scratch"),
          pybind11::arg("barrier_buffer_multicast_ptr"),
          pybind11::arg("trap_record_ptr"), pybind11::arg("ep_rank"),
          pybind11::arg("comm_clusters"),
          pybind11::arg("compute_clusters"),
          pybind11::arg("minibatch_rows"),
          pybind11::arg("macrobatch_rows"),
          pybind11::arg("swiglu_limit"), pybind11::arg("spin_limit"));
    m.def("fp8_block_megakernel_prewarm",
          &mok_sm90::fp8_block_terminal_entry::entry_prewarm, "",
          pybind11::arg("device_index"),
          pybind11::arg("comm_clusters") = 1);
#endif
    m.def("fwd_epilogue", &utils::fwd_epilogue, "",
          pybind11::arg("y_shared"), pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"));
    m.def("routed_epilogue_out",
          [](const at::Tensor &combine_buffer, const at::Tensor &topk_weights,
             const at::Tensor &output) {
              utils::routed_epilogue_out(combine_buffer, topk_weights, output);
          },
          "", pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"),
          pybind11::arg("output"));
    m.def("bwd_epilogue", &utils::bwd_epilogue, "",
          pybind11::arg("d_x_shared"), pybind11::arg("d_x_routed_buffer"));
}
