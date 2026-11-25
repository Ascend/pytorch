#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"

namespace fx_converter {

// Framework-shared Main resources (not op-type specific).
// Stream/notify counts match ModelDesc reusable resources (GE Mix needs sub-stream + 2 notifies).
struct SharedRuntime {
  fx_converter::Node* split_streams = nullptr;
  uint64_t s_stream = 0; // logic stream 0 (main)
  uint64_t s_sub_stream = 0; // logic stream 1 (Mix VectorCore sub-task)
  fx_converter::Node* create_notifies = nullptr;
  uint64_t s_notify0 = 0;
  uint64_t s_notify1 = 0;
  bool has_mix_resources = false;
  fx_converter::Node* select_l2 = nullptr;
  uint64_t s_allocator = 0;
  // GetPlatformInfo on the one logical graph. Output 0 is platform, output 1 is core_num.
  fx_converter::Node* inner_platform = nullptr;
  uint64_t s_platform = 0;
  fx_converter::Node* inner_core_num = nullptr;
  int32_t inner_core_num_port = 0;
  uint64_t s_core_num = 0;
  fx_converter::Node* c_options = nullptr;
  uint64_t s_options = 0;
  fx_converter::Node* c_dtype = nullptr;
  uint64_t s_dtype = 0;
};

// Process-long Const payloads used when wiring dual-stage Fwk for one OPP op type.
struct OppDualStageFuncs {
  gert::OpImplRegisterV2::InferShapeKernelFunc* infer = nullptr;
  gert::OpImplRegisterV2::OpExecPrepareFunc* prepare = nullptr;
  gert::OpImplRegisterV2::OpExecLaunchFunc* launch = nullptr;
  void** space_registry = nullptr;
  const char* op_type = nullptr;
};

// Per-op-type dual-stage Fwk nodes (created by each OpLowering that needs OPP prepare/launch).
struct DualStageOpFwk {
  fx_converter::Node* c_infer = nullptr;
  uint64_t s_infer = 0;
  fx_converter::Node* c_prepare = nullptr;
  fx_converter::Node* c_launch = nullptr;
  uint64_t s_prepare = 0;
  uint64_t s_launch = 0;
  fx_converter::Node* c_op_type = nullptr;
  fx_converter::Node* c_registry = nullptr;
};

// Start a single logical execution graph through the public ES builder.
// Runtime Init/Main/DeInit graphs are created later by GE CompileGraph.
void CreateGraphSkeleton(BuiltExecuteGraph* built, const std::string& name);

void BuildInitFeeds(ge::Graph* init, uint64_t& sym_gen);

// Uses process-shared options and the float dtype Const payload.
// CreateL1/CreateL2/GetPlatformInfo are wired on the one logical graph. SelectL2 and DualStage
// Fwk consume those output endpoints directly.
SharedRuntime BuildSharedRuntime(
    BuiltExecuteGraph* built,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes);

// Wire process-local Const(prepare/launch/infer) for one OPP type.
// OM2 restores function pointers through its runtime binding contract.
DualStageOpFwk BuildDualStageOpFwk(
    ge::Graph* main,
    const OppDualStageFuncs& funcs,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes);

} // namespace fx_converter
