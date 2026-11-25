#include "exe_graph/graph_skeleton_builder.h"

#include <string>

#include "api/fx_allocator_bridge.h"

#include "exe_graph/shared_const_store.h"
#include "register/op_impl_registry.h"

namespace fx_converter {

// The ES facade owns one online execution stream per loaded instance.  Keep the
// generated runtime resource graph aligned with that contract; the legacy
// ModelV2 path used two logical streams, which is not available to facade
// LoadGraph unless an external multi-stream binding is supplied.
constexpr int64_t kMixStreamNum = 1;
constexpr int64_t kMixNotifyNum = 2;
namespace {

// CreateL1/CreateL2/GetPlatformInfo live on the one logical graph. Callers use their
// output endpoints directly. The old InnerData/parent bridge belonged to the three-subgraph
// representation, which CreateGraphSkeleton no longer builds.
void WireInitAllocators(ge::Graph *init, const gert::TensorPlacement placement, const int64_t stream_num,
                        uint64_t s_l1, uint64_t s_l2s, uint64_t s_platform, uint64_t s_core_num, uint64_t &sym_gen,
                        fx_converter::Node **out_inner_l1, fx_converter::Node **out_inner_l2s, fx_converter::Node **out_inner_platform,
                        fx_converter::Node **out_inner_core_num, std::vector<fx_converter::Node *> *kernel_nodes) {
  const ge::TensorDesc empty_td;

  fx_converter::Node *c_placement = AddConstNode(init, "init_const_placement", &placement, sizeof(placement), false,
                                                 sym_gen, FxrtConstType::kGertTensorPlacement);
  const uint64_t s_placement = c_placement->GetExtendInfo()->GetOutputSymbol(0);
  fx_converter::Node *c_stream_num = AddConstNode(init, "init_const_stream_num", &stream_num, sizeof(stream_num), false,
                                                  sym_gen, FxrtConstType::kInt64T);
  const uint64_t s_stream_num = c_stream_num->GetExtendInfo()->GetOutputSymbol(0);

  auto l1_desc = std::make_shared<FxNodeSpec>("create_l1_allocator", "CreateL1Allocator");
  l1_desc->AddInputDesc(empty_td);
  l1_desc->AddOutputDesc(empty_td);
  fx_converter::Node *create_l1 = fx_converter::AddNode(init, l1_desc);
  WireSymbols(create_l1, {s_placement}, {s_l1});
  fx_converter::AddDataEdge(init, c_placement, 0, create_l1, 0);

  auto l2s_desc = std::make_shared<FxNodeSpec>("create_l2_allocators", "CreateL2Allocators");
  l2s_desc->AddInputDesc(empty_td);
  l2s_desc->AddInputDesc(empty_td);
  l2s_desc->AddOutputDesc(empty_td);
  l2s_desc->AddOutputDesc(empty_td);
  fx_converter::Node *create_l2s = fx_converter::AddNode(init, l2s_desc);
  const uint64_t s_l2_pools = sym_gen++;
  WireSymbols(create_l2s, {s_stream_num, s_placement}, {s_l2s, s_l2_pools});
  fx_converter::AddDataEdge(init, c_stream_num, 0, create_l2s, 0);
  fx_converter::AddDataEdge(init, c_placement, 0, create_l2s, 1);
  // Keep CreateL1 alive before CreateL2 (GE AddDependency pattern).
  fx_converter::AddDataEdge(init, create_l1, -1, create_l2s, -1);

  // One GetPlatformInfo for the whole model (GE GetOrCreateUniqueValueHolder("PlatformInfos")).
  auto plat_desc = std::make_shared<FxNodeSpec>("get_platform_info", "GetPlatformInfo");
  plat_desc->AddOutputDesc(empty_td);
  plat_desc->AddOutputDesc(empty_td);
  fx_converter::Node *get_plat = fx_converter::AddNode(init, plat_desc);
  WireSymbols(get_plat, {}, {s_platform, s_core_num});

  *out_inner_l1 = create_l1;
  *out_inner_l2s = create_l2s;
  *out_inner_platform = get_plat;
  *out_inner_core_num = get_plat;
  if (kernel_nodes != nullptr) {
    kernel_nodes->insert(kernel_nodes->end(), {c_placement, c_stream_num, create_l1, create_l2s, get_plat});
  }
}

// Match ModelDesc reusable stream/notify counts (GE Mix: main + sub stream, 2 notifies).
void BuildMainStreamNotifyNodes(ge::Graph *main, uint64_t &sym_gen, fx_converter::Node **split_streams,
                                uint64_t *s_stream, uint64_t *s_sub_stream, fx_converter::Node **create_notifies,
                                uint64_t *s_notify0, uint64_t *s_notify1, fx_converter::Node **c_stream_num,
                                fx_converter::Node **c_notify_num, fx_converter::Node **allocator_feed,
                                uint64_t *s_allocator_feed) {
  const ge::TensorDesc empty_td;
  fx_converter::Node *stream = AddFeedNode(main, "main_stream", "Data", kFeedStream, empty_td, sym_gen);
  const uint64_t s_stream_vec = stream->GetExtendInfo()->GetOutputSymbol(0);
  // ModelExecuteArg::external_allocator lands here. Keep this standard feed for
  // runtimes that forward the session allocator into online execute graphs.
  fx_converter::Node *alloc_feed = AddFeedNode(main, "main_allocator", "Data", kFeedAllocator, empty_td, sym_gen);
  if (allocator_feed != nullptr) {
    *allocator_feed = alloc_feed;
  }
  if (s_allocator_feed != nullptr) {
    *s_allocator_feed = alloc_feed->GetExtendInfo()->GetOutputSymbol(0);
  }
  (void)AddFeedNode(main, "main_events", "Data", kFeedEvents, empty_td, sym_gen);
  fx_converter::Node *notifies = AddFeedNode(main, "main_notifies", "Data", kFeedNotifies, empty_td, sym_gen);
  const uint64_t s_notify_vec = notifies->GetExtendInfo()->GetOutputSymbol(0);

  const int64_t stream_num = kMixStreamNum;
  const int64_t notify_num = kMixNotifyNum;
  *c_stream_num = AddConstNode(main, "const_stream_num", &stream_num, sizeof(stream_num), false, sym_gen,
                               FxrtConstType::kInt64T);
  const uint64_t s_stream_num = (*c_stream_num)->GetExtendInfo()->GetOutputSymbol(0);
  *c_notify_num = AddConstNode(main, "const_notify_num", &notify_num, sizeof(notify_num), false, sym_gen,
                               FxrtConstType::kInt64T);
  const uint64_t s_notify_num = (*c_notify_num)->GetExtendInfo()->GetOutputSymbol(0);

  auto split_desc = std::make_shared<FxNodeSpec>("split_streams", "SplitRtStreams");
  split_desc->AddInputDesc(empty_td);
  split_desc->AddInputDesc(empty_td);
  split_desc->AddOutputDesc(empty_td);
  split_desc->AddOutputDesc(empty_td);
  *split_streams = fx_converter::AddNode(main, split_desc);
  *s_stream = sym_gen++;
  *s_sub_stream = sym_gen++;
  WireSymbols(*split_streams, {s_stream_vec, s_stream_num}, {*s_stream, *s_sub_stream});
  fx_converter::AddDataEdge(main, stream, 0, *split_streams, 0);
  fx_converter::AddDataEdge(main, *c_stream_num, 0, *split_streams, 1);

  // GE graph_converter LoweringRtNotifies → CreateNotifies (reuse ModelV2 notify pool).
  auto notify_desc = std::make_shared<FxNodeSpec>("create_notifies", "CreateNotifies");
  notify_desc->AddInputDesc(empty_td);
  notify_desc->AddInputDesc(empty_td);
  notify_desc->AddOutputDesc(empty_td);
  notify_desc->AddOutputDesc(empty_td);
  *create_notifies = fx_converter::AddNode(main, notify_desc);
  *s_notify0 = sym_gen++;
  *s_notify1 = sym_gen++;
  WireSymbols(*create_notifies, {s_notify_vec, s_notify_num}, {*s_notify0, *s_notify1});
  fx_converter::AddDataEdge(main, notifies, 0, *create_notifies, 0);
  fx_converter::AddDataEdge(main, *c_notify_num, 0, *create_notifies, 1);
}

void BuildSelectL2AndSharedConsts(ge::Graph *main, fx_converter::Node *split_streams, uint64_t s_stream,
                                  uint64_t s_l1, uint64_t s_l2s, fx_converter::Node *inner_l1, fx_converter::Node *inner_l2s,
                                  gert::OpExecuteOptions *exec_options, ge::DataType *dtype_float, uint64_t &sym_gen,
                                  fx_converter::Node **select_l2, uint64_t *s_allocator, fx_converter::Node **c_stream_id,
                                  fx_converter::Node **c_options, uint64_t *s_options, fx_converter::Node **c_dtype,
                                  uint64_t *s_dtype, gert::TensorPlacement placement, fx_converter::Node *allocator_feed,
                                  uint64_t s_allocator_feed, fx_converter::Node **select_l1) {
  const ge::TensorDesc empty_td;
  const int64_t stream_id = 0;
  *c_stream_id = AddConstNode(main, "const_stream_id", &stream_id, sizeof(stream_id), false, sym_gen,
                              FxrtConstType::kInt64T);
  const uint64_t s_stream_id = (*c_stream_id)->GetExtendInfo()->GetOutputSymbol(0);

  // Re-select the L1 allocator on Main every run. SelectL1Allocator also calls
  // CachingMemAllocator::SetStream for GE-owned allocators. Without it the
  // allocator's stream_ stays null, and its out-of-memory path (AllocateWithTryRecycle ->
  // Synchronize -> Recycle) would drain the default stream instead of the one the model
  // actually runs on, and could recycle blocks still being written.
  fx_converter::Node *c_placement_main =
      AddConstNode(main, "const_placement_main", &placement, sizeof(placement), false, sym_gen,
                   FxrtConstType::kGertTensorPlacement);
  const uint64_t s_placement_main = c_placement_main->GetExtendInfo()->GetOutputSymbol(0);
  // The allocator reaches this graph through the standard main_allocator feed:
  // GE fills it from the AllocatorPtr registered on the stream
  // (GraphManager::GetExecuteGraphAllocators wraps it into a gert::Allocators for
  // OM2, and GetUserAllocatorOrFixedBaseAllocator reads it directly at run time).
  // No in-graph resolve node is needed, so this builder stays free of the
  // GE-internal gert::Allocators type.
  auto sel_l1_desc = std::make_shared<FxNodeSpec>("select_l1_allocator", "SelectL1Allocator");
  for (int i = 0; i < 4; ++i) {  // placement, external allocator, created allocator, stream
    sel_l1_desc->AddInputDesc(empty_td);
  }
  sel_l1_desc->AddOutputDesc(empty_td);
  *select_l1 = fx_converter::AddNode(main, sel_l1_desc);
  const uint64_t s_l1_selected = sym_gen++;
  WireSymbols(*select_l1, {s_placement_main, s_allocator_feed, s_l1, s_stream}, {s_l1_selected});
  fx_converter::AddDataEdge(main, c_placement_main, 0, *select_l1, 0);
  fx_converter::AddDataEdge(main, allocator_feed, 0, *select_l1, 1);
  fx_converter::AddDataEdge(main, inner_l1, 0, *select_l1, 2);
  fx_converter::AddDataEdge(main, split_streams, 0, *select_l1, 3);

  auto sel_desc = std::make_shared<FxNodeSpec>("select_l2_allocator", "SelectL2Allocator");
  for (int i = 0; i < 4; ++i) {
    sel_desc->AddInputDesc(empty_td);
  }
  sel_desc->AddOutputDesc(empty_td);
  *select_l2 = fx_converter::AddNode(main, sel_desc);
  *s_allocator = sym_gen++;
  // L2 now binds the stream-bound L1 rather than the raw one from Init.
  WireSymbols(*select_l2, {s_stream_id, s_stream, s_l1_selected, s_l2s}, {*s_allocator});
  fx_converter::AddDataEdge(main, *c_stream_id, 0, *select_l2, 0);
  fx_converter::AddDataEdge(main, split_streams, 0, *select_l2, 1);
  fx_converter::AddDataEdge(main, *select_l1, 0, *select_l2, 2);
  fx_converter::AddDataEdge(main, inner_l2s, 0, *select_l2, 3);

  *c_options = AddConstNode(main, "const_exec_options", exec_options, sizeof(*exec_options), false, sym_gen,
                            FxrtConstType::kGertOpExecuteOptionsPtr);
  *s_options = (*c_options)->GetExtendInfo()->GetOutputSymbol(0);
  *c_dtype = AddConstNode(main, "const_dtype_float", dtype_float, sizeof(*dtype_float), false, sym_gen,
                          FxrtConstType::kGeDataType);
  *s_dtype = (*c_dtype)->GetExtendInfo()->GetOutputSymbol(0);
}

void FillSharedRuntime(SharedRuntime *rt, fx_converter::Node *split_streams, uint64_t s_stream, uint64_t s_sub_stream,
                       fx_converter::Node *create_notifies, uint64_t s_notify0, uint64_t s_notify1, fx_converter::Node *select_l2,
                       uint64_t s_allocator, fx_converter::Node *inner_platform, uint64_t s_platform,
                       fx_converter::Node *inner_core_num, uint64_t s_core_num, fx_converter::Node *c_options, uint64_t s_options,
                       fx_converter::Node *c_dtype, uint64_t s_dtype) {
  rt->split_streams = split_streams;
  rt->s_stream = s_stream;
  rt->s_sub_stream = s_sub_stream;
  rt->create_notifies = create_notifies;
  rt->s_notify0 = s_notify0;
  rt->s_notify1 = s_notify1;
  // GE uses LaunchKernelV2 in a single-stream scene. SplitRtStreams only
  // initializes kMixStreamNum outputs, so output 1 is unavailable here.
  rt->has_mix_resources = kMixStreamNum > 1 && kMixNotifyNum >= 2;
  rt->select_l2 = select_l2;
  rt->s_allocator = s_allocator;
  rt->inner_platform = inner_platform;
  rt->s_platform = s_platform;
  rt->inner_core_num = inner_core_num;
  rt->inner_core_num_port = (inner_platform == inner_core_num) ? 1 : 0;
  rt->s_core_num = s_core_num;
  rt->c_options = c_options;
  rt->s_options = s_options;
  rt->c_dtype = c_dtype;
  rt->s_dtype = s_dtype;
}

SharedRuntime BuildSharedRuntimeImpl(BuiltExecuteGraph *built, gert::OpExecuteOptions *exec_options,
                                     ge::DataType *dtype_float, uint64_t &sym_gen,
                                     std::vector<fx_converter::Node *> *kernel_nodes) {
  SharedRuntime rt;
  ge::Graph *main = built->main.get();
  ge::Graph *init = built->init.get();

  fx_converter::Node *split_streams = nullptr;
  fx_converter::Node *create_notifies = nullptr;
  fx_converter::Node *c_stream_num = nullptr;
  fx_converter::Node *c_notify_num = nullptr;
  uint64_t s_stream = 0;
  uint64_t s_sub_stream = 0;
  uint64_t s_notify0 = 0;
  uint64_t s_notify1 = 0;
  fx_converter::Node *allocator_feed = nullptr;
  uint64_t s_allocator_feed = 0;
  BuildMainStreamNotifyNodes(main, sym_gen, &split_streams, &s_stream, &s_sub_stream, &create_notifies, &s_notify0,
                             &s_notify1, &c_stream_num, &c_notify_num, &allocator_feed, &s_allocator_feed);

  // Shared symbols for CreateL1/L2/GetPlatform <-> SelectL2 / DualStage Fwk.
  const gert::TensorPlacement placement = gert::kOnDeviceHbm;
  const int64_t stream_num = kMixStreamNum;
  const uint64_t s_l1 = sym_gen++;
  const uint64_t s_l2s = sym_gen++;
  const uint64_t s_platform = sym_gen++;
  const uint64_t s_core_num = sym_gen++;
  fx_converter::Node *inner_l1 = nullptr;
  fx_converter::Node *inner_l2s = nullptr;
  fx_converter::Node *inner_platform = nullptr;
  fx_converter::Node *inner_core_num = nullptr;
  WireInitAllocators(init, placement, stream_num, s_l1, s_l2s, s_platform, s_core_num, sym_gen, &inner_l1, &inner_l2s,
                     &inner_platform, &inner_core_num, kernel_nodes);

  fx_converter::Node *select_l2 = nullptr;
  fx_converter::Node *c_stream_id = nullptr;
  fx_converter::Node *c_options = nullptr;
  fx_converter::Node *c_dtype = nullptr;
  uint64_t s_allocator = 0;
  uint64_t s_options = 0;
  uint64_t s_dtype = 0;
  fx_converter::Node *select_l1 = nullptr;
  BuildSelectL2AndSharedConsts(main, split_streams, s_stream, s_l1, s_l2s, inner_l1, inner_l2s, exec_options,
                               dtype_float, sym_gen, &select_l2, &s_allocator, &c_stream_id, &c_options, &s_options,
                               &c_dtype, &s_dtype, placement, allocator_feed, s_allocator_feed, &select_l1);

  FillSharedRuntime(&rt, split_streams, s_stream, s_sub_stream, create_notifies, s_notify0, s_notify1, select_l2,
                    s_allocator, inner_platform, s_platform, inner_core_num, s_core_num, c_options, s_options, c_dtype,
                    s_dtype);

  if (kernel_nodes != nullptr) {
    kernel_nodes->insert(kernel_nodes->end(),
                         {split_streams, create_notifies, select_l1, select_l2, c_options, c_stream_num, c_notify_num,
                          c_stream_id, c_dtype});
  }
  return rt;
}

}  // namespace

void CreateGraphSkeleton(BuiltExecuteGraph *built, const std::string &name) {
  if (built == nullptr) {
    return;
  }
  built->fx_builder = std::make_unique<fx_converter::FxGraphBuilder>(name);
  if (built->fx_builder == nullptr || built->fx_builder->GetGraph() == nullptr) {
    built->error = "Create logical execution graph failed";
    return;
  }
  built->logical_graph = built->fx_builder->GetGraph();
  built->root = built->logical_graph;
  if (built->root == nullptr) {
    built->error = "Create logical execution graph backend failed";
    return;
  }
  built->main = built->root;
  built->init = built->root;
  built->deinit = built->root;
}

void BuildInitFeeds(ge::Graph *init, uint64_t &sym_gen) {
  if (fx_converter::FxGraphBuilder::Lookup(init) != nullptr) {
    return;
  }
  const ge::TensorDesc empty_td;
  (void)AddFeedNode(init, "const_rt_session", "ConstData", 0, empty_td, sym_gen);
  (void)AddFeedNode(init, "const_outer_weight", "ConstData", 1, empty_td, sym_gen);
  (void)AddFeedNode(init, "const_model_desc", "ConstData", 2, empty_td, sym_gen);

  // In the ES path Init and Main are represented by one logical graph. The
  // runtime Data feeds below are created once by BuildMainStreamNotifyNodes;
  // duplicating the same negative feed indices here creates ambiguous inputs.
  (void)AddFeedNode(init, "init_stream", "Data", kFeedStream, empty_td, sym_gen);
  (void)AddFeedNode(init, "init_allocator", "Data", kFeedAllocator, empty_td, sym_gen);
  (void)AddFeedNode(init, "init_events", "Data", kFeedEvents, empty_td, sym_gen);
  (void)AddFeedNode(init, "init_notifies", "Data", kFeedNotifies, empty_td, sym_gen);
}

SharedRuntime BuildSharedRuntime(BuiltExecuteGraph *built, uint64_t &sym_gen,
                                 std::vector<fx_converter::Node *> *kernel_nodes) {
  return BuildSharedRuntimeImpl(built, fx_converter::SharedExecOptions(), fx_converter::SharedDtypeFloat(), sym_gen,
                                kernel_nodes);
}

DualStageOpFwk BuildDualStageOpFwk(ge::Graph *main, const OppDualStageFuncs &funcs,
                                   uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes) {
  DualStageOpFwk fwk;
  const ge::TensorDesc empty_td;

  fwk.c_op_type = AddConstNode(main, std::string("const_op_type_") + funcs.op_type, funcs.op_type,
                               std::char_traits<char>::length(funcs.op_type) + 1, true, sym_gen);
  fwk.c_registry = AddConstNode(main, std::string("const_space_registry_") + funcs.op_type, funcs.space_registry,
                                sizeof(void *), false, sym_gen, FxrtConstType::kVoidPtr);
  fwk.c_infer = AddConstNode(main, std::string("const_infer_func_") + funcs.op_type, funcs.infer, sizeof(*funcs.infer),
                             false, sym_gen, FxrtConstType::kKernelFuncPtr);
  fwk.s_infer = fwk.c_infer->GetExtendInfo()->GetOutputSymbol(0);
  fwk.c_prepare = AddConstNode(main, std::string("const_prepare_func_") + funcs.op_type, funcs.prepare,
                               sizeof(*funcs.prepare), false, sym_gen, FxrtConstType::kKernelFuncPtr);
  fwk.s_prepare = fwk.c_prepare->GetExtendInfo()->GetOutputSymbol(0);
  fwk.c_launch = AddConstNode(main, std::string("const_launch_func_") + funcs.op_type, funcs.launch,
                              sizeof(*funcs.launch), false, sym_gen, FxrtConstType::kKernelFuncPtr);
  fwk.s_launch = fwk.c_launch->GetExtendInfo()->GetOutputSymbol(0);
  if (kernel_nodes != nullptr) {
    kernel_nodes->insert(kernel_nodes->end(),
                         {fwk.c_op_type, fwk.c_registry, fwk.c_infer, fwk.c_prepare, fwk.c_launch});
  }
  return fwk;
}

}  // namespace fx_converter
