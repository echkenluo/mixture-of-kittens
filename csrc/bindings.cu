#include "mok_megakernel.cuh"
#include "mxfp8.cuh"
#include "scheduler.cuh"
#include "sm90_fp8_block_routed.cuh"
#include "sm90_fp8_block_worker_test.cuh"
#include "utils.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_gather_top_experts", &utils::all_gather_top_experts::entrypoint, "",
          pybind11::arg("top_experts"), pybind11::arg("all_gather_top_experts_buffer"),
          pybind11::arg("all_gather_top_experts_buffer_multicast_ptr"), pybind11::arg("rank"), pybind11::arg("chunk_bytes"));
    m.def("barrier_all", &utils::barrier_all::entrypoint, "",
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"), pybind11::arg("target"));
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("schedule_capacity"), pybind11::arg("rank"));
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
          &mok_sm90::fp8_block_routed::combine_out, "",
          pybind11::arg("routed_y"), pybind11::arg("combine_buffer"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("schedule_peer_rank"),
          pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("topk"));
#endif
    m.def("fwd_epilogue", &utils::fwd_epilogue, "",
          pybind11::arg("y_shared"), pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"));
    m.def("bwd_epilogue", &utils::bwd_epilogue, "",
          pybind11::arg("d_x_shared"), pybind11::arg("d_x_routed_buffer"));
}
