/**
 * Convert orchestration: fxrt::ir::Graph -> ge::ExecuteGraph (Class-1 path).
 *
 * Pipeline (see CODE_STRUCTURE.md section 2):
 *   skeleton + SymbolEnv -> BuildGraphPlan -> WireConvertPrelude
 *   -> MakeLoweringContext -> LowerAllComputeNodes -> NetOutput + ModelV2 meta
 *
 * OpLowerings self-register via FX_REGISTER_OP_LOWERING at shared-lib load.
 * Unsupported compute ops fail conversion with an explicit error (no silent skip).
 */

#include "exe_graph/execute_graph_builder.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/logger.h"
#include "shape/bind_symbol_env_builder.h"
#include "exe_graph/execute_graph_metadata.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/graph_skeleton_builder.h"
#include "exe_graph/infer_shape_fusion.h"
#include "exe_graph/const_value_pass.h"
#include "exe_graph/offline_optimizer_run.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "shape/mrt_to_ge_expr.h"
#include "shape/shape_list_utils.h"
#include "register/op_catalog.h"
#include "lowering/op_lowering.h"
#include "lowering/op_lowering_registry.h"
#include "lowering/common/wiring.h"
#include "lowering/common/edge_layout_planner.h"
#include "lowering/common/output_layout_planner.h"
#include "lowering/common/view_utils.h"
#include "ops/op_def/ops_name.h"

namespace {

using fx_converter::DtypeOfNode;
using fx_converter::ShapeOfNode;
using fx_converter::IsModelFeedTensor;

std::string DescribeOp(const fxrt::ir::NodePtr &node) {
  if (node == nullptr) {
    return "<null>";
  }
  return fxrt::ops::ToStr(node->op);
}

BuiltExecuteGraph FailConvert(BuiltExecuteGraph built, const std::string &err) {
  built.error = err;
  RT_GLOG(ERROR) << "FxConverter error: " << err;
  return built;
}

// Flattened view of the MRT graph used by later Convert stages.
struct GraphPlan {
  std::unordered_map<const fxrt::ir::Node *, std::string> name_of;
  std::vector<fxrt::ir::NodePtr> placeholders;
  std::vector<fxrt::ir::NodePtr> compute_nodes;
  // Graph returns (GE NetOutput may consume N tensors). Single-output graphs use size==1.
  std::vector<fxrt::ir::NodePtr> return_values;
  std::vector<std::string> return_names;
};

// Mutable state for Stage 3 prelude through Stage 5 metadata.
struct ConvertWorkspace {
  uint64_t sym_gen = 0;
  std::string error;
  std::unordered_map<std::string, fx_converter::TensorRef> tensors;
  std::vector<ModelIoSpec> model_inputs;
  std::vector<fx_converter::Node *> graph_input_datas;
  std::vector<fxrt::ir::NodePtr> graph_input_irs;  // parallel to graph_input_datas
  std::vector<fx_converter::Node *> kernel_nodes;
  std::vector<OpCniSpec> cni_specs;
  std::unordered_set<int32_t> mutated_input_feeds;
  fx_converter::SharedRuntime rt;
  fx_converter::Node *bind_symbols = nullptr;
  std::unordered_map<std::string, fx_converter::DualStageOpFwk> fwk_by_op;
  // Multi-output ops (TopKV2): producer IR node → per-slot TensorRefs, recorded from
  // LoweringResult.outputs; tuple_getitem consumers alias into it natively.
  std::unordered_map<const fxrt::ir::Node *, std::vector<fx_converter::TensorRef>> multi_out_tensors;
  std::unordered_map<std::string, uint64_t> autofuse_resources;
  std::unordered_map<std::string, fx_converter::Node *> autofuse_resource_nodes;
};

void AssignPlaceholderNames(const fxrt::ir::Graph &graph, GraphPlan *plan) {
  // ModelV2 Execute only feeds tensor Data. Skip SymInt / empty-shape placeholders that
  // Dynamo may inject under dynamic=True (even if an older export_ir still lists them).
  size_t in_idx = 0;
  for (const auto &node : graph.inputs) {
    if (!IsModelFeedTensor(graph, node)) {
      continue;
    }
    plan->placeholders.push_back(node);
    plan->name_of[node.get()] = "input_" + std::to_string(in_idx++);
  }
  size_t param_idx = 0;
  for (const auto &node : graph.parameters) {
    if (!IsModelFeedTensor(graph, node)) {
      continue;
    }
    plan->placeholders.push_back(node);
    plan->name_of[node.get()] = "param_" + std::to_string(param_idx++);
  }
}

// True when the node yields a tuple of tensors (multi-output op like topk), as opposed
// to a Tuple<int> parameter pack (view size lists) or a make_tuple packing node.
bool IsTensorTupleOutput(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTuple() ||
      node->op == fxrt::ops::Op_make_tuple) {
    return false;
  }
  const auto tup = node->output->ToTuple();
  return tup != nullptr && tup->Size() > 0 && (*tup)[0] != nullptr && (*tup)[0]->IsTensor();
}

// A supported lowering owns its complete tensor tuple, including compiled-kernel
// mutations that have no catalog row. Project its recorded result without a
// separate device op; catalog membership is not an output-layout capability.
bool IsMultiOutGetitem(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->op != fxrt::ops::Op_tuple_getitem || node->inputs.empty() ||
      !IsTensorTupleOutput(node->inputs[0])) {
    return false;
  }
  return fx_converter::OpLoweringRegistry::Instance().Find(node->inputs[0]) != nullptr;
}

// Output slot a tuple_getitem projects. The index rides on Int inputs of the getitem
// node (exports that fold it into the node itself still keep one trailing Int input);
// the last Int input wins, 0 when there is none.
int64_t ResolveGetitemSlot(const fxrt::ir::NodePtr &node) {
  int64_t idx = 0;
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    if (node->inputs[i] != nullptr && node->inputs[i]->output != nullptr && node->inputs[i]->output->IsInt()) {
      idx = node->inputs[i]->output->ToInt();
    }
  }
  if (idx < 0 && !node->inputs.empty() && IsTensorTupleOutput(node->inputs[0])) {
    idx += static_cast<int64_t>(node->inputs[0]->output->ToTuple()->Size());
  }
  return idx;
}

// One flag per output slot of `node`: a slot is final when the graph returns it directly
// or through one of the node's tuple_getitems. Single-output nodes get {is_return}.
std::vector<bool> BuildOutputFinalMask(const fxrt::ir::NodePtr &node, const GraphPlan &plan, bool is_return) {
  if (!IsTensorTupleOutput(node)) {
    return {is_return};
  }
  const auto tup = node->output->ToTuple();
  std::vector<bool> mask(tup->Size(), false);
  for (const auto &ret : plan.return_values) {
    if (ret == node) {
      std::fill(mask.begin(), mask.end(), true);
      continue;
    }
    if (ret != nullptr && ret->op == fxrt::ops::Op_tuple_getitem && !ret->inputs.empty() &&
        ret->inputs[0] == node) {
      const int64_t idx = ResolveGetitemSlot(ret);
      if (idx >= 0 && static_cast<size_t>(idx) < mask.size()) {
        mask[static_cast<size_t>(idx)] = true;
      }
    }
  }
  return mask;
}

bool CollectComputeNodes(const fxrt::ir::Graph &graph, GraphPlan *plan, std::string *err) {
  size_t op_idx = 0;
  for (const auto &node : graph.nodes) {
    if (node == nullptr || node->op == fxrt::ops::Op_End) {
      continue;
    }
    if (node->op == fxrt::ops::Op_return) {
      if (node->inputs.empty()) {
        continue;
      }
      // Export may leave a make_tuple wrapper; unwrap to N tensor producers (TopKV2).
      if (node->inputs.size() == 1 && node->inputs[0] != nullptr &&
          node->inputs[0]->op == fxrt::ops::Op_make_tuple) {
        plan->return_values = node->inputs[0]->inputs;
      } else {
        plan->return_values = node->inputs;
      }
      continue;
    }
    if (node->output == nullptr || (!node->output->IsTensor() && !IsTensorTupleOutput(node))) {
      continue;
    }
    plan->name_of[node.get()] = "op" + std::to_string(op_idx++);
    plan->compute_nodes.push_back(node);
  }
  return true;
}

bool ValidateGraphPlan(const GraphPlan &plan, BuiltExecuteGraph *built) {
  if (plan.placeholders.empty()) {
    *built = FailConvert(std::move(*built), "Convert failed: graph has no inputs/parameters");
    return false;
  }
  if (plan.compute_nodes.empty()) {
    *built = FailConvert(std::move(*built), "Convert failed: graph has no compute ops");
    return false;
  }
  if (plan.return_values.empty()) {
    *built = FailConvert(std::move(*built), "Convert failed: graph has no return node");
    return false;
  }
  if (std::any_of(plan.return_values.begin(), plan.return_values.end(),
                   [&plan](const auto &rv) {
                     return rv == nullptr || plan.name_of.count(rv.get()) == 0;
                   })) {
    *built = FailConvert(std::move(*built), "Convert failed: return value is not a converted tensor producer");
    return false;
  }
  return true;
}

bool ValidateOpsSupported(const GraphPlan &plan, BuiltExecuteGraph *built) {
  auto &registry = fx_converter::OpLoweringRegistry::Instance();
  for (const auto &node : plan.compute_nodes) {
    if (IsMultiOutGetitem(node) || registry.Find(node) != nullptr) {
      // Multi-output tuple_getitem is aliased natively at lowering time, no dispatch.
      continue;
    }
    std::ostringstream oss;
    oss << "Convert failed: unsupported op '" << DescribeOp(node) << "'";
    const auto *entry = fx_converter::LookupOpCatalog(node);
    if (entry != nullptr) {
      oss << " (catalogued as '" << entry->logical_name << "' -> OPP '" << entry->opp_type
          << "', but no OpLowering matched)";
    }
    *built = FailConvert(std::move(*built), oss.str());
    return false;
  }
  return true;
}

// Stage 2: name + collect + validate; sets plan.return_names on success.
bool BuildGraphPlan(const fxrt::ir::Graph &graph, GraphPlan *plan, BuiltExecuteGraph *built) {
  AssignPlaceholderNames(graph, plan);
  std::string collect_err;
  if (!CollectComputeNodes(graph, plan, &collect_err)) {
    *built = FailConvert(std::move(*built), collect_err);
    return false;
  }
  if (!ValidateGraphPlan(*plan, built)) {
    return false;
  }
  plan->return_names.clear();
  plan->return_names.reserve(plan->return_values.size());
  for (const auto &rv : plan->return_values) {
    plan->return_names.push_back(plan->name_of[rv.get()]);
  }
  return ValidateOpsSupported(*plan, built);
}

fx_converter::TensorEndpoint AddRuntimeStorageOffsetNode(ge::Graph *graph, fx_converter::Node *feed,
                                                         const int64_t *offset_slot,
                                                         uint64_t &sym_gen,
                                                         std::vector<fx_converter::Node *> *kernel_nodes,
                                                         const std::string &tag) {
  if (graph == nullptr || feed == nullptr || feed->GetExtendInfo() == nullptr || offset_slot == nullptr) return {};
  const ge::TensorDesc empty_td;
  fx_converter::RtGraphBuilder builder(graph, empty_td, sym_gen, kernel_nodes);
  const auto slot = builder.Const("const_storage_offset_" + tag, &offset_slot, sizeof(offset_slot), false,
                                 fx_converter::FxrtConstType::kInt64T);
  // A live feed dependency keeps this read in the per-Execute graph instead of
  // letting GE move a Const-only subgraph into one-time initialization.
  const auto offset = builder.Node(
      "FxGetStorageOffset", "storage_offset_" + tag,
      {slot, {feed, 0, feed->GetExtendInfo()->GetOutputSymbol(0)}}, 1U);
  return offset.outs.empty() ? fx_converter::TensorEndpoint{} : offset.outs.front();
}

void WirePlaceholderFeeds(const fxrt::ir::Graph &graph, const GraphPlan &plan, BuiltExecuteGraph *built,
                          ConvertWorkspace *ws) {
  int32_t feed_idx = 0;
  for (const auto &ph : plan.placeholders) {
    const std::string ph_name = plan.name_of.at(ph.get());
    const auto logical_shape = ShapeOfNode(ph);
    const auto tensor = ph->output->ToTensor();
    const auto &saved_storage_shape = tensor->StorageShape();
    const auto storage_shape = saved_storage_shape.empty() ? logical_shape : saved_storage_shape;
    const ge::DataType dt = DtypeOfNode(ph);
    const ge::Format storage_format = static_cast<ge::Format>(tensor->Format());
    ge::TensorDesc td = fx_converter::MakeTensorDesc(storage_shape, dt, storage_format);
    td.SetOriginShape(ge::Shape(logical_shape));
    fx_converter::Node *data = fx_converter::AddFeedNode(built->main.get(), ph_name, "Data", feed_idx, td, ws->sym_gen);
    (void)fx_converter::AddInputNode(built->main.get(), data);

    fx_converter::TensorRef ref;
    ref.producer = data;
    ref.symbol = data->GetExtendInfo()->GetOutputSymbol(0);
    ref.shape = storage_shape;
    ref.storage_shape = storage_shape;
    ref.logical_shape = logical_shape;
    ref.stride = tensor->Strides();
    if (built->symbol_env != nullptr) {
      (void)SymbolicShapeExprsOfNode(ph, *built->symbol_env, &ref.logical_shape_exprs);
      const auto stride = ph->attrs.find("fxrt_feed_stride");
      if (stride != ph->attrs.end() && stride->second != nullptr && stride->second->IsTuple() &&
          !ReadIntListWithSymbols(stride->second->ToTuple(), *built->symbol_env, &ref.stride, &ref.stride_exprs)) {
        ws->error = "failed to bind input stride expressions for " + ph_name;
        return;
      }
    }
    const auto dense = ph->attrs.find("fxrt_proven_dense");
    ref.proven_dense = dense != ph->attrs.end() && dense->second != nullptr &&
                       dense->second->IsBool() && dense->second->ToBool();
    // ES dataset feeds drop TensorV2 metadata. Read the live storage offset from
    // session-owned host state below, independently of the tracing-time value.
    ref.storage_offset = 0;
    ref.dtype = dt;
    ref.format = storage_format;
    ref.feed_index = feed_idx;
    built->input_requires_base_format.push_back(false);
    ws->tensors[ph_name] = ref;
    ModelIoSpec feed_io;
    feed_io.name = ph_name;
    feed_io.shape = logical_shape;
    feed_io.storage_shape = storage_shape;
    feed_io.data_type = static_cast<int32_t>(dt);
    feed_io.format = storage_format;
    ws->model_inputs.push_back(std::move(feed_io));
    // BindSymbolEnv only needs graph.inputs feeds (not parameters), in Data order.
    const bool from_graph_input =
        std::find(graph.inputs.begin(), graph.inputs.end(), ph) != graph.inputs.end();
    if (from_graph_input) {
      ws->graph_input_datas.push_back(data);
      ws->graph_input_irs.push_back(ph);
    }
    ++feed_idx;
  }
  built->input_storage_offsets = std::make_shared<std::vector<int64_t>>(plan.placeholders.size(), 0);
  for (size_t i = 0; i < plan.placeholders.size(); ++i) {
    const auto &ph = plan.placeholders[i];
    const std::string &name = plan.name_of.at(ph.get());
    auto &ref = ws->tensors.at(name);
    ref.runtime_storage_offset = AddRuntimeStorageOffsetNode(
        built->main.get(), ref.producer, &(*built->input_storage_offsets)[i], ws->sym_gen, &ws->kernel_nodes, name);
    if (!ref.runtime_storage_offset.Valid()) {
      ws->error = "failed to read runtime storage-offset feed for " + name;
      return;
    }
  }
}

// Stage 3: init feeds, placeholder Data, SharedRuntime, BindSymbolEnv.
void WireConvertPrelude(const fxrt::ir::Graph &graph, const GraphPlan &plan, BuiltExecuteGraph *built,
                        ConvertWorkspace *ws) {
  fx_converter::BuildInitFeeds(built->init.get(), ws->sym_gen);
  WirePlaceholderFeeds(graph, plan, built, ws);
  ws->rt = fx_converter::BuildSharedRuntime(built, ws->sym_gen, &ws->kernel_nodes);
  ws->bind_symbols = fx_converter::AttachBindSymbolEnv(built->main.get(), built->symbol_env.get(),
                                                       ws->graph_input_datas, ws->graph_input_irs, ws->sym_gen,
                                                       &ws->kernel_nodes);
}

fx_converter::LoweringContext MakeLoweringContext(BuiltExecuteGraph *built, ConvertWorkspace *ws, GraphPlan *plan) {
  fx_converter::LoweringContext ctx;
  ctx.main = built->main.get();
  ctx.rt = &ws->rt;
  ctx.built = built;
  ctx.symbol_env = built->symbol_env.get();
  ctx.bind_symbols = ws->bind_symbols;
  ctx.graph_input_datas = &ws->graph_input_datas;
  ctx.sym_gen = &ws->sym_gen;
  ctx.kernel_nodes = &ws->kernel_nodes;
  ctx.tensors = &ws->tensors;
  ctx.name_of = &plan->name_of;
  ctx.compute_nodes = &plan->compute_nodes;
  ctx.return_values = &plan->return_values;
  ctx.multi_out_tensors = &ws->multi_out_tensors;
  ctx.fwk_by_op = &ws->fwk_by_op;
  ctx.autofuse_resources = &ws->autofuse_resources;
  ctx.autofuse_resource_nodes = &ws->autofuse_resource_nodes;
  return ctx;
}

// Stage 4: dispatch each compute node through OpLoweringRegistry.
// Control edges: previous Launch -> next Prepare so device ops run in graph order.
//
// Intermediate FreeMemory follows GE Guarder / last-consumer semantics: each AllocMemory
// is freed only after every Launch that reads that tensor (not after the next op, and not
// deferred to the final Launch). Same-Execute L2 reuse stays possible for linear chains.
struct PendingFree {
  fx_converter::Node *alloc_mem = nullptr;
  std::string tag;
  fx_converter::Node *producer_launch = nullptr;
  std::vector<fx_converter::Node *> consumer_launches;
};

// AllocMemory publishes a GertTensorData on output 0. SplitDataTensor publishes the
// shape Tensor on output 0 and the GertTensorData on output 1. FreeMemory reads its
// input as a GertTensorData; feeding it the shape Tensor jumps through a garbage Free.
int32_t FreeableOutputIndex(const fx_converter::Node *alloc_mem) {
  if (alloc_mem != nullptr && std::string(alloc_mem->GetType()) == "SplitDataTensor") {
    return 1;
  }
  return 0;
}

void WireFreeIntermediateMemory(ge::Graph *main, fx_converter::Node *alloc_mem,
                                const std::vector<fx_converter::Node *> &after_launches, const std::string &tag,
                                std::vector<fx_converter::Node *> *kernel_nodes) {
  if (main == nullptr || alloc_mem == nullptr || after_launches.empty()) {
    return;
  }
  const ge::TensorDesc empty_td;
  auto free_spec = std::make_shared<fx_converter::FxNodeSpec>("free_out_" + tag, "FreeMemory");
  free_spec->AddInputDesc(empty_td);
  const int32_t out_index = FreeableOutputIndex(alloc_mem);
  const uint64_t s_alloc = alloc_mem->GetExtendInfo()->GetOutputSymbol(static_cast<uint32_t>(out_index));
  fx_converter::Node *free_node = fx_converter::AddNode(main, free_spec);
  if (free_node == nullptr) return;
  fx_converter::WireSymbols(free_node, {s_alloc}, {});
  (void)fx_converter::AddDataEdge(main, alloc_mem, out_index, free_node, 0);
  // Control deps from every consumer Launch (GE CreateVoidGuarder AddDependencyToGuarder).
  for (fx_converter::Node *launch : after_launches) {
    if (launch != nullptr) {
      (void)fx_converter::AddControlEdge(main, launch, free_node);
    }
  }
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(free_node);
  }
}

// Register this node's Launch as a consumer of each Alloc it actually reads.
// GE Guarder: FreeMemory after every Launch that uses that GertTensorData*.
// IR-name matching misses view/identity: they copy addr_node to the alias TensorRef
// and have no Launch, so the root Alloc was freed after the producer Launch while
// a later Launch still held the same Alloc (NotEqual MTE).
void NoteOneTensorInput(const fxrt::ir::NodePtr &in_node, const GraphPlan &plan,
                        fx_converter::Node *consumer_launch,
                        const std::unordered_map<std::string, fx_converter::TensorRef> *tensors,
                        std::unordered_map<std::string, PendingFree> *pending_frees) {
  if (in_node == nullptr || in_node->output == nullptr || !in_node->output->IsTensor()) {
    return;
  }
  const auto name_it = plan.name_of.find(in_node.get());
  if (name_it == plan.name_of.end()) {
    return;
  }
  auto add_consumer = [consumer_launch](PendingFree *pending) {
    auto &consumers = pending->consumer_launches;
    if (std::find(consumers.begin(), consumers.end(), consumer_launch) == consumers.end()) {
      consumers.push_back(consumer_launch);
    }
  };
  const auto free_it = pending_frees->find(name_it->second);
  if (free_it != pending_frees->end()) {
    add_consumer(&free_it->second);
  }
  if (tensors == nullptr) {
    return;
  }
  const auto ref_it = tensors->find(name_it->second);
  if (ref_it == tensors->end() || !ref_it->second.unpacked.addr.Valid()) {
    return;
  }
  const auto &ref = ref_it->second;
  fx_converter::Node *addr = ref.allocation.Valid() ? ref.allocation.node : ref.unpacked.addr.node;
  for (auto &kv : *pending_frees) {
    if (kv.second.alloc_mem == addr) {
      add_consumer(&kv.second);
    }
  }
}

void NoteInputConsumers(const fxrt::ir::NodePtr &node, const GraphPlan &plan, fx_converter::Node *consumer_launch,
                        const std::unordered_map<std::string, fx_converter::TensorRef> *tensors,
                        std::unordered_map<std::string, PendingFree> *pending_frees) {
  if (node == nullptr || consumer_launch == nullptr || pending_frees == nullptr) {
    return;
  }
  auto note_inputs_of = [&](const fxrt::ir::NodePtr &src) {
    if (src == nullptr) {
      return;
    }
    for (const auto &in_node : src->inputs) {
      NoteOneTensorInput(in_node, plan, consumer_launch, tensors, pending_frees);
    }
  };
  note_inputs_of(node);
}

// Nodes read by compiled-kernel HOPs and by nothing else. A compiled kernel indexes its
// arguments with strides frozen into the generated code, so such a node has to reach it as a
// view onto the storage those strides address; materializing it would be a dense buffer the
// kernel then walks with the base's strides. Nothing else reads the node, so aliasing is safe.
std::unordered_set<const fxrt::ir::Node *> CollectCompiledKernelOnlyNodes(const GraphPlan &plan) {
  std::unordered_set<const fxrt::ir::Node *> kernel_read;
  std::unordered_set<const fxrt::ir::Node *> other_read;
  for (const auto &node : plan.compute_nodes) {
    auto &seen = node->op == fxrt::ops::Op_compiled_kernel_mutation ? kernel_read : other_read;
    for (const auto &in : node->inputs) {
      if (in != nullptr) {
        seen.insert(in.get());
      }
    }
  }
  for (const auto &ret : plan.return_values) {
    if (ret != nullptr) {
      other_read.insert(ret.get());
    }
  }
  std::unordered_set<const fxrt::ir::Node *> only;
  for (const auto *node : kernel_read) {
    if (other_read.count(node) == 0U) {
      only.insert(node);
    }
  }
  return only;
}

bool AliasMultiOutGetitem(const fxrt::ir::NodePtr &node, const std::string &tag, ConvertWorkspace *ws,
                          BuiltExecuteGraph *built) {
  const int64_t idx = ResolveGetitemSlot(node);
  const auto multi_it = ws->multi_out_tensors.find(node->inputs[0].get());
  if (multi_it == ws->multi_out_tensors.end() || idx < 0 || static_cast<size_t>(idx) >= multi_it->second.size()) {
    *built = FailConvert(std::move(*built),
                         "Convert failed: tuple_getitem '" + tag + "' has no lowered multi-output producer");
    return false;
  }
  ws->tensors[tag] = multi_it->second[static_cast<size_t>(idx)];
  return true;
}

// Adapters are ordinary intermediate producers: keep CNI, launch ordering and
// allocation lifetimes identical at consumer edges and the graph-output boundary.
void RecordLayoutAdapters(const fxrt::ir::NodePtr &node, const GraphPlan &plan,
                          fx_converter::EdgeLayoutPlan *layout_plan, fx_converter::LoweringContext &ctx,
                          ConvertWorkspace *ws, fx_converter::Node **prev_launch, fx_converter::Node **last_launch,
                          std::unordered_map<std::string, PendingFree> *pending_frees,
                          BuiltExecuteGraph *built) {
  for (auto &adapter : layout_plan->adapters) {
    if (*prev_launch != nullptr && adapter.prepare != nullptr) {
      (void)fx_converter::AddControlEdge(built->main.get(), *prev_launch, adapter.prepare);
    }
    // The adapter reads the original input allocation, not its substituted dense output.
    fx_converter::RestoreInputLayouts(ctx, *layout_plan);
    NoteInputConsumers(node, plan, adapter.launch, ctx.tensors, pending_frees);
    for (fx_converter::Node *alloc : adapter.mem_to_frees) {
      PendingFree pending;
      pending.alloc_mem = alloc;
      pending.tag = "layout_" + std::to_string(pending_frees->size());
      pending.producer_launch = adapter.launch;
      pending_frees->emplace(pending.tag, std::move(pending));
    }
    if (adapter.launch != nullptr) {
      *prev_launch = adapter.launch;
      *last_launch = adapter.launch;
    }
    if (!adapter.skip_cni && !adapter.cni.op_type.empty()) ws->cni_specs.push_back(std::move(adapter.cni));
  }
}

bool LowerAllComputeNodes(const GraphPlan &plan, fx_converter::LoweringContext &ctx, ConvertWorkspace *ws,
                          fx_converter::Node **last_launch, std::unordered_map<std::string, PendingFree> *pending_frees,
                          BuiltExecuteGraph *built) {
  auto &registry = fx_converter::OpLoweringRegistry::Instance();
  const auto compiled_kernel_only = CollectCompiledKernelOnlyNodes(plan);
  ctx.compiled_kernel_only = &compiled_kernel_only;
  fx_converter::Node *prev_launch = nullptr;
  *last_launch = nullptr;

  for (const auto &node : plan.compute_nodes) {
    const std::string tag = plan.name_of.at(node.get());
    if (IsMultiOutGetitem(node)) {
      // Native alias: the producer lowered earlier in topological order; publish its
      // recorded output slot.
      if (!AliasMultiOutGetitem(node, tag, ws, built)) {
        return false;
      }
      continue;
    }
    const bool is_return =
        std::find(plan.return_names.begin(), plan.return_names.end(), tag) != plan.return_names.end();
    ctx.output_final_mask = BuildOutputFinalMask(node, plan, is_return);
    auto *lowering = registry.Find(node, &ctx);
    if (lowering == nullptr) {
      *built = FailConvert(std::move(*built), "Convert failed: unsupported op '" + DescribeOp(node) + "'");
      return false;
    }
    fx_converter::LoweringResult result;
    std::string lower_err;
    fx_converter::EdgeLayoutPlan layout_plan;
    if (!fx_converter::PlanInputLayouts(node, *lowering, tag, ctx, &layout_plan, &lower_err)) {
      fx_converter::RestoreInputLayouts(ctx, layout_plan);
      *built = FailConvert(std::move(*built), "Convert layout planning '" + DescribeOp(node) + "': " + lower_err);
      return false;
    }
    RecordLayoutAdapters(node, plan, &layout_plan, ctx, ws, &prev_launch, last_launch, pending_frees, built);
    // Reinstall the substitutions after recording the adapter's source lifetimes.
    fx_converter::ApplyInputLayouts(ctx, layout_plan);
    if (!lowering->Lower(node, tag, ctx, &result, &lower_err)) {
      fx_converter::RestoreInputLayouts(ctx, layout_plan);
      *built =
          FailConvert(std::move(*built), "Convert failed while lowering '" + DescribeOp(node) + "': " + lower_err);
      return false;
    }
    if (result.outputs.empty()) {
      if (result.skip_cni) {
        continue;
      }
      *built = FailConvert(std::move(*built),
                           "Convert failed: lowering '" + DescribeOp(node) + "' produced no outputs");
      return false;
    }
    const auto *catalog_entry = fx_converter::LookupOpCatalog(node);
    if (catalog_entry != nullptr) {
      for (size_t output_index = 0; output_index < result.outputs.size(); ++output_index) {
        if (fx_converter::CatalogRefInput(*catalog_entry, output_index) >= 0 &&
            result.outputs[output_index].feed_index >= 0) {
          ws->mutated_input_feeds.insert(result.outputs[output_index].feed_index);
        }
      }
    }
    // Chain device ops: previous Launch -> next Prepare (explicit LoweringResult.prepare).
    if (prev_launch != nullptr && result.prepare != nullptr) {
      (void)fx_converter::AddControlEdge(built->main.get(), prev_launch, result.prepare);
    }
    NoteInputConsumers(node, plan, result.launch, ctx.tensors, pending_frees);
    fx_converter::RestoreInputLayouts(ctx, layout_plan);
    if (result.launch != nullptr) {
      prev_launch = result.launch;
      *last_launch = result.launch;
    }
    for (size_t free_i = 0; free_i < result.mem_to_frees.size(); ++free_i) {
      fx_converter::Node *alloc = result.mem_to_frees[free_i];
      if (alloc == nullptr) {
        continue;
      }
      PendingFree pending;
      pending.alloc_mem = alloc;
      // Key names are debug-facing only: consumers match by alloc-node identity.
      pending.tag = (free_i == 0) ? tag : tag + "_o" + std::to_string(free_i);
      pending.producer_launch = result.launch;
      pending_frees->emplace(pending.tag, std::move(pending));
    }
    ws->tensors[tag] = result.outputs[0];
    if (result.outputs.size() > 1U || IsTensorTupleOutput(node)) {
      // Record the full output vector so later tuple_getitem nodes alias it natively.
      ws->multi_out_tensors[node.get()] = result.outputs;
    }
    if (!result.skip_cni && !result.cni.op_type.empty()) {
      ws->cni_specs.push_back(std::move(result.cni));
    }
    // Unknown Python/compiled-kernel effects and writes to caller storage are
    // barriers for sharing layout conversions. The cache holds graph values,
    // but reusing one across a mutation would read a snapshot of old contents.
    bool may_write = lowering->MayWriteInputs();
    if (result.launch != nullptr) {
      for (const auto &input : node->inputs) {
        if (input == nullptr || input->output == nullptr || !input->output->IsTensor()) continue;
        const auto name = plan.name_of.find(input.get());
        if (name == plan.name_of.end()) continue;
        const auto ref = ws->tensors.find(name->second);
        if (ref == ws->tensors.end() || !ref->second.unpacked.addr.Valid()) continue;
        for (const auto &output : result.outputs) {
          may_write = may_write || (output.unpacked.addr.node == ref->second.unpacked.addr.node &&
                                    output.unpacked.addr.out == ref->second.unpacked.addr.out);
        }
      }
    }
    if (may_write) ctx.dense_layout_cache.clear();
  }

  ctx.compiled_kernel_only = nullptr;
  return true;
}

void WirePendingFrees(std::unordered_map<std::string, PendingFree> *pending_frees, BuiltExecuteGraph *built,
                      std::vector<fx_converter::Node *> *kernel_nodes, fx_converter::Node *last_launch) {
  for (auto &kv : *pending_frees) {
    PendingFree &pending = kv.second;
    std::vector<fx_converter::Node *> after = pending.consumer_launches;
    if (after.empty()) {
      // Dead intermediate (no reader): free after producer Launch, else final Launch.
      fx_converter::Node *fallback = pending.producer_launch != nullptr ? pending.producer_launch : last_launch;
      if (fallback != nullptr) {
        after.push_back(fallback);
      }
    }
    WireFreeIntermediateMemory(built->main.get(), pending.alloc_mem, after, pending.tag, kernel_nodes);
  }
}

// Materialize one model-output slot as OutputData + AllocModelOut + Ensure.
// Each ExecuteAlloc output index needs its own OutputData chain: GE binds
// output_values[i]->Set(outputs[i]) onto NetOutput's input chains, so two returns that
// share a TensorRef / OutputData node would leave the earlier shell unfilled.
//
// AllocModelOut fills the ExecuteAlloc-empty shell from L1 *before* Ensure. TensorToOut
// then copies from the source GTD (view / duplicate alias of AllocMemory) instead of the
// dst==nullptr share-out path, which calls MoveL2ToL1 / ConvertToRootBlock. Sharing the
// same L2 MultiStreamMemBlock to two Ensures (ETA model__1: embedding + as_strided view,
// unsqueeze(ge) returned twice) converts the span on the first Ensure and crashes on the
// second (inner block is already an L1 root, not a PageSpan).
bool MaterializeNetOutputSlot(const std::string &name, fx_converter::TensorRef &ref, ConvertWorkspace *ws,
                              fx_converter::Node *last_launch,
                              std::unordered_map<std::string, PendingFree> *pending_frees, BuiltExecuteGraph *built,
                              std::vector<fx_converter::Node *> *output_ensures, fx_converter::Node **producer, uint64_t *symbol) {
  if (ref.producer == nullptr && !ref.HasUnpackedPair()) {
    *built = FailConvert(std::move(*built), "Convert failed: return tensor '" + name +
                                               "' has neither Tensor* nor shape/address pair");
    return false;
  }
  if (!ref.HasUnpackedPair()) {
    *built = FailConvert(std::move(*built), "Convert failed: boxed return tensor '" + name +
                                               "' is not OutputData and has no shape/address pair");
    return false;
  }
  if (!ref.unpacked.shape.Valid()) {
    *built = FailConvert(std::move(*built), "Convert failed: return tensor '" + name +
                                               "' has no shape producer for AllocModelOut");
    return false;
  }
  const ge::TensorDesc empty_td;
  const ge::TensorDesc td = fx_converter::MakeNdTensorDesc(ref.shape, ref.dtype);
  const std::string tag = "net_output_" + name + "_" + std::to_string(ws->sym_gen);
  uint64_t output_symbol = 0;
  fx_converter::Node *output_data =
      fx_converter::AddOutputDataNode(built->main.get(), tag, td, ws->sym_gen, &output_symbol);
  ws->kernel_nodes.push_back(output_data);

  fx_converter::Node *alloc_model_out = nullptr;
  if (!ref.can_adopt_for_model_output) {
    ge::DataType out_dtype = ref.dtype;
    fx_converter::Node *c_out_dtype = fx_converter::AddConstNode(built->main.get(), "const_out_dtype_" + tag, &out_dtype,
                                                           sizeof(out_dtype), false, ws->sym_gen,
                                                           fx_converter::FxrtConstType::kGeDataType);
    ws->kernel_nodes.push_back(c_out_dtype);
    uint64_t s_size = 0;
    const fx_converter::CalcTensorSizeInputs size_inputs{
        {c_out_dtype, 0, c_out_dtype->GetExtendInfo()->GetOutputSymbol(0)},
        ref.unpacked.shape,
    };
    fx_converter::Node *calc_size =
        fx_converter::AddCalcTensorSizeNode(built->main.get(), tag, size_inputs, ws->sym_gen, &s_size);
    ws->kernel_nodes.push_back(calc_size);
    uint64_t s_model_out_addr = 0;
    alloc_model_out = fx_converter::AddAllocModelOutTensorNode(
        built->main.get(), tag, td, empty_td, ws->rt.s_allocator, ws->rt.select_l2, s_size, calc_size, output_symbol,
        output_data, ws->sym_gen, &s_model_out_addr);
    (void)s_model_out_addr;
    ws->kernel_nodes.push_back(alloc_model_out);
  }

  const fx_converter::TensorFieldEndpoints fields = fx_converter::AddTensorFieldConsts(
      built->main.get(), tag, ref.dtype, gert::kOnDeviceHbm, ge::FORMAT_ND, ws->sym_gen, &ws->kernel_nodes);
  const fx_converter::EnsureOutMemoryInputs ensure_inputs{
      ref.unpacked.shape,
      ref.unpacked.addr,
      fields,
      {ws->rt.split_streams, 0, ws->rt.s_stream},
      {output_data, 0, output_symbol},
      {ws->rt.select_l2, 0, ws->rt.s_allocator},
  };
  fx_converter::Node *ensure = fx_converter::AddEnsureTensorAtOutMemoryNode(built->main.get(), tag, ensure_inputs,
                                                                     last_launch, &ws->kernel_nodes);
  // AllocModelOut mutates OutputData in place; Ensure must see the filled dest.
  if (alloc_model_out != nullptr) {
    (void)fx_converter::AddControlEdge(built->main.get(), alloc_model_out, ensure);
  }
  output_ensures->push_back(ensure);

  // A returned view aliases its root AllocMemory. Keep that allocation alive until
  // EnsureTensorAtOutMemory has consumed it for this model output.
  for (auto &pending_kv : *pending_frees) {
    auto &pending = pending_kv.second;
    const auto *allocation = ref.allocation.Valid() ? ref.allocation.node : ref.unpacked.addr.node;
    if (pending.alloc_mem == allocation &&
        std::find(pending.consumer_launches.begin(), pending.consumer_launches.end(), ensure) ==
            pending.consumer_launches.end()) {
      pending.consumer_launches.push_back(ensure);
    }
  }
  *producer = output_data;
  *symbol = output_symbol;
  return true;
}

struct NetOutputSlot {
  fx_converter::Node *producer = nullptr;
  uint64_t symbol = 0;
  std::vector<int64_t> shape;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
};

// Group only proved layouts. A dense sibling can share the same output owner
// when a non-contiguous return enables storage export for that allocation.
struct StorageOutputGroup {
  fx_converter::OutputStorageCandidate candidate;
  fx_converter::TensorRef direct_output;
  int64_t elements = 0;
  bool enabled = false;
  int64_t slot = -1;
};

void PlanStorageOutputs(const GraphPlan &plan, const ConvertWorkspace &ws,
                        std::vector<fx_converter::OutputStorageCandidate> *candidates,
                        std::vector<int64_t> *group_ids, std::vector<StorageOutputGroup> *groups) {
  candidates->resize(plan.return_names.size());
  group_ids->assign(plan.return_names.size(), -1);
  for (size_t i = 0; i < plan.return_names.size(); ++i) {
    const auto &name = plan.return_names[i];
    const auto ref = ws.tensors.find(name);
    const auto value = std::find_if(plan.return_values.begin(), plan.return_values.end(),
        [&](const auto &item) { return plan.name_of.at(item.get()) == name; });
    if (ref == ws.tensors.end() || value == plan.return_values.end() ||
        !fx_converter::PlanOutputStorage(*value, ref->second, &candidates->at(i))) {
      continue;
    }
    const auto &candidate = candidates->at(i);
    auto group = std::find_if(groups->begin(), groups->end(), [&](const auto &item) {
      return fx_converter::SameOutputStorage(item.candidate, candidate);
    });
    if (group == groups->end()) {
      groups->push_back(StorageOutputGroup{});
      group = groups->end() - 1;
      group->candidate = candidate;
    }
    group->elements = std::max(group->elements, candidate.elements);
    group->enabled = group->enabled || candidate.noncontiguous;
    if (ref->second.producer != nullptr && std::string(ref->second.producer->GetType()) == "OutputData") {
      group->direct_output = ref->second;
    }
    group_ids->at(i) = std::distance(groups->begin(), group);
  }
}

// Stage 5.1: NetOutput consumes return tensor(s); control-dep on last Launch.
bool WireNetOutput(const GraphPlan &plan, ConvertWorkspace &ws, fx_converter::Node *last_launch,
                   std::unordered_map<std::string, PendingFree> *pending_frees, BuiltExecuteGraph *built,
                   fx_converter::LoweringContext &ctx) {
  if (plan.return_names.empty()) {
    *built = FailConvert(std::move(*built), "Convert failed: empty return list for NetOutput");
    return false;
  }
  std::vector<NetOutputSlot> slots;
  std::vector<fx_converter::Node *> output_ensures;
  std::unordered_set<const fx_converter::Node *> claimed_output_data;
  slots.reserve(plan.return_names.size());
  std::vector<fx_converter::OutputStorageCandidate> candidates;
  std::vector<int64_t> group_ids;
  std::vector<StorageOutputGroup> groups;
  PlanStorageOutputs(plan, ws, &candidates, &group_ids, &groups);
  // A returned complete dense root gives dynamic subviews a physical owner
  // without copying each view or binding one GE output shell more than once.
  std::vector<int64_t> returned_roots(plan.return_names.size(), -1);
  std::vector<int64_t> returned_root_slots(plan.return_names.size(), -1);
  std::vector<fx_converter::OutputViewSpec> returned_views(plan.return_names.size());
  const auto output_ref = [&](size_t index) {
    auto ref = ws.tensors.at(plan.return_names[index]);
    if (ref.logical_shape_exprs.empty()) {
      (void)SymbolicShapeExprsOfNode(plan.return_values[index], *built->symbol_env, &ref.logical_shape_exprs);
    }
    return ref;
  };
  for (size_t i = 0; i < plan.return_names.size(); ++i) {
    if (returned_roots[i] >= 0 || ws.tensors.count(plan.return_names[i]) == 0U) continue;
    const auto root = output_ref(i);
    std::vector<size_t> members;
    for (size_t j = 0; j < plan.return_names.size(); ++j) {
      if (returned_roots[j] >= 0 || ws.tensors.count(plan.return_names[j]) == 0U) continue;
      fx_converter::OutputViewSpec view;
      if (fx_converter::PlanReturnedRootView(plan.return_values[i], root, plan.return_values[j], output_ref(j), &view)) {
        view.output_index = j;
        returned_views[j] = std::move(view);
        members.push_back(j);
      }
    }
    if (members.size() > 1U) {
      for (const auto member : members) returned_roots[member] = static_cast<int64_t>(i);
    }
  }
  // Preserve aliases only when the output planner proves a shared physical slot.
  // Unproved output layouts use one dense output slot per logical return below.
  auto emit_slot = [&](fx_converter::TensorRef ref, const std::string &name) {
    const bool direct = ref.producer != nullptr && std::string(ref.producer->GetType()) == "OutputData" &&
                        claimed_output_data.count(ref.producer) == 0U;
    // A dense subview can still cover less storage than its allocation. Give
    // copy-out a borrowed header so GE cannot bind that owner to the smaller
    // output slot (including an empty output) through zero-copy optimization.
    const auto owner = ref.allocation.Valid() ? ref.allocation : ref.unpacked.addr;
    const auto &physical_shape = fx_converter::PhysicalShape(ref);
    const bool owns_header = owner.Valid() && owner.node == ref.unpacked.addr.node &&
                             owner.out == ref.unpacked.addr.out;
    const bool unknown_capacity = std::any_of(physical_shape.begin(), physical_shape.end(),
                                              [](int64_t dim) { return dim < 0; });
    if (!direct && !ref.can_adopt_for_model_output && owns_header && fx_converter::IsDenseTensorRef(ref) &&
        (physical_shape != fx_converter::LogicalShape(ref) || unknown_capacity)) {
      fx_converter::TensorRef borrowed;
      if (!fx_converter::RebaseContiguousView(ref, "output_view_" + name, ctx, &borrowed, &built->error)) {
        return false;
      }
      ref = std::move(borrowed);
    }
    NetOutputSlot slot;
    slot.shape = ref.shape;
    slot.dtype = ref.dtype;
    slot.format = ref.format;
    if (direct) {
      slot.producer = ref.producer;
      slot.symbol = ref.symbol;
    } else if (!MaterializeNetOutputSlot(name, ref, &ws, last_launch, pending_frees, built,
                                         &output_ensures, &slot.producer, &slot.symbol)) {
      return false;
    }
    claimed_output_data.insert(slot.producer);
    slots.push_back(slot);
    built->output_views.emplace_back();
    return true;
  };
  for (size_t output_index = 0; output_index < plan.return_names.size(); ++output_index) {
    const auto &name = plan.return_names[output_index];
    const auto it = ws.tensors.find(name);
    if (it == ws.tensors.end()) {
      *built = FailConvert(std::move(*built), "Convert failed: missing tensor for return '" + name + "'");
      return false;
    }
    auto output_tensor = it->second;
    if (output_tensor.feed_index >= 0 && !output_tensor.HasUnpackedPair()) {
      uint64_t shape_symbol = 0;
      uint64_t address_symbol = 0;
      auto *split = fx_converter::GetOrCreateSplitDataTensor(
          ctx, output_tensor, "mutated_feed_return_" + name, 0, &shape_symbol, &address_symbol);
      if (split == nullptr) {
        *built = FailConvert(std::move(*built), "Convert failed: cannot expose mutated input feed '" + name + "'");
        return false;
      }
      output_tensor.unpacked.shape = fx_converter::TensorEndpoint{split, 0, shape_symbol};
      output_tensor.unpacked.addr = fx_converter::TensorEndpoint{split, 1, address_symbol};
      ws.tensors[name] = output_tensor;
    }
    if (returned_roots[output_index] >= 0) {
      const auto root_index = static_cast<size_t>(returned_roots[output_index]);
      auto &slot = returned_root_slots[root_index];
      if (slot < 0) {
        slot = static_cast<int64_t>(slots.size());
        if (!emit_slot(output_ref(root_index), plan.return_names[root_index])) return false;
      }
      built->output_views[slot].push_back(returned_views[output_index]);
      continue;
    }
    if (group_ids[output_index] >= 0 && groups[group_ids[output_index]].enabled) {
      auto &group = groups[group_ids[output_index]];
      if (group.slot < 0) {
        auto physical = group.candidate.source;
        if (group.direct_output.producer != nullptr) {
          // An already returned root owns a correctly sized OutputData slot.
          // Reuse it directly instead of copying or adopting that owner twice.
          physical = group.direct_output;
        } else {
          physical.producer = nullptr;
          physical.shape = {group.elements};
          physical.logical_shape.clear();
          physical.storage_shape.clear();
          physical.logical_shape_ep = {};
          physical.storage_offset = 0;
          physical.stride = {1};
          physical.proven_dense = true;
          if (!fx_converter::AddConcreteStorageShapeConst(built->main.get(), built, "output_storage_" + name,
                  physical.shape, ws.sym_gen, &ws.kernel_nodes, &physical.unpacked.shape.node,
                  &physical.unpacked.shape.sym, &built->error)) {
            return false;
          }
          physical.unpacked.shape.out = 0;
          // Establish a bounded borrowed span. GE ZeroCopy must not bind the
          // complete producer allocation to this possibly smaller copy-out.
          // Keep the original owner for the last-reader/free dependency.
          fx_converter::TensorRef bounded;
          if (!fx_converter::RebaseContiguousView(physical, "output_storage_" + name, ctx,
                                                  &bounded, &built->error)) {
            return false;
          }
          physical = std::move(bounded);
        }
        group.slot = static_cast<int64_t>(slots.size());
        if (!emit_slot(physical, "storage_" + name)) {
          return false;
        }
      }
      const auto &candidate = candidates[output_index];
      built->output_views[group.slot].push_back(
          fx_converter::OutputViewSpec{output_index, true, candidate.shape, candidate.stride, 0});
      continue;
    }
    // Unproved layouts and RT2 keep the existing dense ABI consumer. Reuse the
    // same planner/cache; a producer must not write dense just for a return.
    auto consumer = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    consumer->op = fxrt::ops::Op_contiguous;
    const auto return_it = std::find_if(plan.return_values.begin(), plan.return_values.end(),
        [&](const auto &value) { return plan.name_of.at(value.get()) == name; });
    if (return_it == plan.return_values.end()) {
      *built = FailConvert(std::move(*built), "Convert output layout: unresolved return '" + name + "'");
      return false;
    }
    consumer->inputs = {*return_it};
    auto *lowering = fx_converter::OpLoweringRegistry::Instance().Find(consumer, &ctx);
    fx_converter::EdgeLayoutPlan output_plan;
    std::string error;
    if (lowering == nullptr || !fx_converter::PlanInputLayouts(consumer, *lowering, "output_" + name,
                                                              ctx, &output_plan, &error)) {
      fx_converter::RestoreInputLayouts(ctx, output_plan);
      *built = FailConvert(std::move(*built), "Convert output layout: " + error);
      return false;
    }
    fx_converter::Node *previous = last_launch;
    RecordLayoutAdapters(consumer, plan, &output_plan, ctx, &ws, &previous, &last_launch, pending_frees, built);
    fx_converter::ApplyInputLayouts(ctx, output_plan);
    auto ref = ws.tensors.at(name);
    fx_converter::RestoreInputLayouts(ctx, output_plan);
    if (!emit_slot(ref, name)) {
      return false;
    }
    built->output_views.back().push_back(fx_converter::OutputViewSpec{output_index, false, {}, {}, 0});
  }
  if (built->fx_builder != nullptr) {
    std::vector<std::pair<fx_converter::Node *, int32_t>> outputs;
    for (const auto &slot : slots) {
      outputs.emplace_back(slot.producer, 0);
      built->output_shapes.push_back(slot.shape);
      built->output_dtypes.push_back(static_cast<int32_t>(slot.dtype));
      built->output_formats.push_back(static_cast<int32_t>(slot.format));
    }
    return fx_converter::SetOutputs(built->main.get(), outputs) == ge::GRAPH_SUCCESS;
  }
  built->error = "ES graph builder is required";
  return false;
}

}  // namespace

BuiltExecuteGraph BuildExecuteGraphFromMrtIr(const fxrt::ir::Graph &graph, const std::string &name) {
  BuiltExecuteGraph built;
  RT_GLOG(INFO) << "FxConverter Convert start: name=" << name;
  // Stage 1: skeleton + shared SymbolEnv (seeded from graph input sym shapes).
  RT_GLOG(INFO) << "Stage 1: CreateGraphSkeleton + InitSymbolEnvFromGraphInputs";
  fx_converter::CreateGraphSkeleton(&built, name);
  fx_converter::InitSymbolEnvFromGraphInputs(&built, graph);
  if (built.symbol_env == nullptr) {
    return FailConvert(std::move(built), "Convert failed: SymbolEnv init failed");
  }
  // Empty seeds are OK for fully static graphs (as_strided / empty_strided with concrete
  // sizes). OpLowerings that need symbolic InferShape must fail clearly or use Const
  // StorageShape fallback (ViewAlias / EmptyStridedAlloc).
  if (built.symbol_env->seeds.empty()) {
    RT_GLOG(INFO) << "SymbolEnv: 0 free-var seeds (static shapes)";
  }

  // Stage 2: plan + validate (no Main wiring yet).
  RT_GLOG(INFO) << "Stage 2: BuildGraphPlan";
  GraphPlan plan;
  if (!BuildGraphPlan(graph, &plan, &built)) {
    return built;
  }
  RT_GLOG(INFO) << "Stage 2 done: placeholders=" << plan.placeholders.size()
                              << " compute_nodes=" << plan.compute_nodes.size()
                              << " returns=" << plan.return_names.size();

  // Stage 3: init feeds, placeholder Data, SharedRuntime, BindSymbolEnv.
  RT_GLOG(INFO) << "Stage 3: WireConvertPrelude";
  ConvertWorkspace ws;
  WireConvertPrelude(graph, plan, &built, &ws);
  if (!ws.error.empty()) return FailConvert(std::move(built), ws.error);
  RT_GLOG(INFO) << "Stage 3 done: kernel_nodes=" << ws.kernel_nodes.size()
                              << " tensors=" << ws.tensors.size()
                              << " bind_symbols=" << (ws.bind_symbols != nullptr ? "set" : "null");

  // Stage 4–5: lower ops, NetOutput, then public metadata attrs (gert.source_op / gert.model.*).
  fx_converter::LoweringContext ctx = MakeLoweringContext(&built, &ws, &plan);
  fx_converter::UnpackGraphInputTensors(ctx);
  fx_converter::Node *last_launch = nullptr;
  std::unordered_map<std::string, PendingFree> pending_frees;
  RT_GLOG(INFO) << "Stage 4: LowerAllComputeNodes (n=" << plan.compute_nodes.size() << ")";
  if (!LowerAllComputeNodes(plan, ctx, &ws, &last_launch, &pending_frees, &built)) {
    return built;
  }
  RT_GLOG(INFO) << "Stage 4 done: cni_specs=" << ws.cni_specs.size()
                              << " last_launch=" << (last_launch != nullptr ? "set" : "null");
  if (!WireNetOutput(plan, ws, last_launch, &pending_frees, &built, ctx)) {
    return built;
  }
  WirePendingFrees(&pending_frees, &built, &ws.kernel_nodes, last_launch);

  // Stage 4.5: collapse the per-op InferShape nodes onto one fused node. Runs before Stage 5 so
  // the fused node is present for gert.source_op attachment and the removed nodes drop out of
  // kernel_nodes / cni_specs.
  RT_GLOG(INFO) << "Stage 4.5: FuseInferShapeNodes";
  (void)fx_converter::FuseInferShapeNodes(&built, ws.bind_symbols, &ws.kernel_nodes, &ws.cni_specs, ws.sym_gen);

  RT_GLOG(INFO) << "Stage 4.6: RewriteTypedConstNodes";
  if (fx_converter::RewriteTypedConstNodes(built.root.get(), built.fx_builder.get(), &ws.kernel_nodes, ws.sym_gen) !=
      ge::GRAPH_SUCCESS) {
    return FailConvert(std::move(built), "Convert failed: RewriteTypedConstNodes failed");
  }

  RT_GLOG(INFO) << "Stage 5: AttachModelV2Metadata";
  std::vector<ModelIoSpec> model_outputs;
  model_outputs.reserve(built.output_shapes.size());
  for (size_t i = 0; i < built.output_shapes.size(); ++i) {
    ModelIoSpec output;
    output.name = "output_" + std::to_string(i);
    output.shape = built.output_shapes[i];
    output.storage_shape = built.output_shapes[i];
    output.data_type = built.output_dtypes[i];
    output.format = static_cast<ge::Format>(built.output_formats[i]);
    model_outputs.push_back(std::move(output));
  }
  if (!AttachModelV2Metadata(built.root.get(), ws.kernel_nodes, ws.cni_specs, ws.model_inputs, model_outputs)) {
    return FailConvert(std::move(built), "Convert failed: AttachModelV2Metadata failed");
  }
  built.input_dtypes.clear();
  built.input_shapes.clear();
  built.input_formats.clear();
  built.input_storage_shapes.clear();
  built.input_dtypes.reserve(ws.model_inputs.size());
  built.input_shapes.reserve(ws.model_inputs.size());
  built.input_formats.reserve(ws.model_inputs.size());
  built.input_storage_shapes.reserve(ws.model_inputs.size());
  for (const auto &io : ws.model_inputs) {
    built.input_dtypes.push_back(io.data_type);
    built.input_shapes.push_back(io.shape);
    built.input_formats.push_back(static_cast<int32_t>(io.format));
    built.input_storage_shapes.push_back(io.storage_shape.empty() ? io.shape : io.storage_shape);
  }
  // Topological sorting / OfflineOptimizer run inside GE CompileGraph
  // (PrepareExecuteGraph). FXRT no longer calls ExecuteGraph::TopologicalSorting.
  if (built.fx_builder == nullptr) {
    return FailConvert(std::move(built), "Convert failed: ES graph builder is missing");
  }
  RT_GLOG(INFO) << "FxConverter Convert OK: inputs=" << built.input_dtypes.size()
                << " outputs=" << built.output_dtypes.size();
  return built;
}
