/**
 * Generic OPP dual-stage lowering (shared by MatMul / BatchMatMul / Concat / Add / ...).
 *
 * Aligns with GE LoweringAclnnNode: one ExecuteGraph template; OPP type selects
 * prepare/launch via OpImplSpaceRegistryV2::GetOpImpl(op_type).
 *
 * Extension: prefer OpCatalog / Fx prepare+launch patch. For a different RT kernel
 * sequence, add a new OpLowering subclass instead of branching this file heavily.
 *
 * Per-op node chain (LowerNaryDualStage):
 *   WireInferShape -> WireFinalOutput | WireIntermediateOutput
 *   -> EnsureTensorRefBoxed (GE ExecuteOp ABI) -> WirePrepare
 *   -> WireAllocWorkspace -> WireLaunch
 *
 * N-ary IO:
 *   InferShape:  [in0..inN-1, rule, infer_func] -> shape
 *   Prepare:     [in0..inN-1, out, options, prepare_func, stream] -> params, ws
 *   Launch:      [in0..inN-1, out, params, ws_addr, ws_size, stream, launch_func]
 */

#include "lowering/dual_stage/dual_stage_op_lowering.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/logger.h"
#include "exe_graph/dual_stage_raw_tensor_kernels.h"
#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "register/op_catalog.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "opp/opp_binding.h"
#include "opp/opp_registry.h"
#include "ops/op_def/ops_name.h"
#include "shape/mrt_to_ge_expr.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {

struct DualStageNodes {
  fx_converter::Node *infer = nullptr;
  fx_converter::Node *out_tensor_node = nullptr;   // boxed Tensor* shared by Prepare and Launch
  fx_converter::Node *output_data_node = nullptr;  // final OutputData only (model IO / Ensure target)
  fx_converter::Node *alloc_out = nullptr;         // intermediate AllocMemory or final AllocModelOutTensor
  TensorFieldEndpoints out_fields;
  fx_converter::Node *prepare = nullptr;
  fx_converter::Node *alloc_ws = nullptr;
  fx_converter::Node *launch = nullptr;
  uint64_t s_shape = 0;
  uint64_t s_out_tensor = 0;   // boxed Tensor* symbol
  uint64_t s_output_data = 0;  // OutputData symbol (final)
  uint64_t s_out_addr = 0;     // GertTensorData* from AllocModelOut / AllocMemory
  uint64_t s_params = 0;
  uint64_t s_ws_size = 0;
  uint64_t s_ws_addr = 0;
};

struct DualStageIoDesc {
  size_t n = 0;
  ge::TensorDesc td_o;
  ge::TensorDesc empty_td;
  std::vector<ge::TensorDesc> td_ins;
};

DualStageIoDesc MakeIoDescs(const std::vector<TensorRef> &ins, const std::vector<int64_t> &out_shape,
                            ge::DataType out_dtype) {
  DualStageIoDesc io;
  io.n = ins.size();
  io.td_o = MakeNdTensorDesc(out_shape, out_dtype);
  io.td_ins.reserve(io.n);
  for (const auto &in : ins) {
    // GE StorageShape is physical. Logical ATen size + transpose_x/adj is a double-count.
    io.td_ins.push_back(MakeTensorDesc(PhysicalShape(in), in.dtype, in.format));
  }
  return io;
}

// InferShape RT node; control-dep on BindSymbolEnv so free vars are bound first.
void WireInferShape(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io,
                    const std::vector<TensorRef> &ins, const DualStageOpFwk &fwk, fx_converter::Node *c_rule,
                    fx_converter::Node *bind_symbols, uint64_t &sym_gen, DualStageNodes *stage) {
  stage->infer = AddInferShapeNode(main, tag, ins, io.td_o, io.empty_td, c_rule, fwk.c_infer, sym_gen, &stage->s_shape);
  if (bind_symbols != nullptr) {
    fx_converter::AddDataEdge(main, bind_symbols, -1, stage->infer, -1);
  }
}

// Graph-final op (GE NetOutput pattern):
//   OutputData + AllocModelOutTensor; Prepare/Launch share one boxed Tensor* via
//   EnsureTensorRefBoxed; later Ensure writes back onto OutputData.
// Prepare must not use empty-shape OutputData shells from ExecuteAlloc.
void WireFinalOutput(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io, const SharedRuntime &rt,
                     uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes, DualStageNodes *stage,
                     bool es_model) {
  stage->output_data_node = AddOutputDataNode(main, tag, io.td_o, sym_gen, &stage->s_output_data);
  uint64_t s_size = 0;
  fx_converter::Node *size_node = nullptr;
  if (es_model) {
    const auto out_shape = io.td_o.GetShape().GetDims();
    const size_t capacity = EsModelOutputCapacityBytes(out_shape, io.td_o.GetDataType());
    size_node = AddConstNode(main, "const_out_capacity_" + tag, &capacity, sizeof(capacity), false, sym_gen,
                             FxrtConstType::kSizeT);
    s_size = size_node->GetExtendInfo()->GetOutputSymbol(0);
  } else {
    // GE allocates this buffer (ExecuteAlloc passes a null output address), so the size
    // must follow InferShape. Use this output's own dtype const: rt.c_dtype is float and
    // would under-allocate an int64/double output.
    ge::DataType out_dtype = io.td_o.GetDataType();
    fx_converter::Node *c_out_dtype =
        AddConstNode(main, "const_out_dtype_" + tag, &out_dtype, sizeof(out_dtype), false, sym_gen,
                     FxrtConstType::kGeDataType);
    const uint64_t s_out_dtype = c_out_dtype->GetExtendInfo()->GetOutputSymbol(0);
    const CalcTensorSizeInputs size_inputs{{c_out_dtype, 0, s_out_dtype}, {stage->infer, 0, stage->s_shape}};
    size_node = AddCalcTensorSizeNode(main, tag, size_inputs, sym_gen, &s_size);
  }
  stage->alloc_out =
    AddAllocModelOutTensorNode(main, tag, io.td_o, io.empty_td, rt.s_allocator, rt.select_l2, s_size, size_node,
                               stage->s_output_data, stage->output_data_node, sym_gen, &stage->s_out_addr);
  stage->out_fields =
      AddTensorFieldConsts(main, tag, io.td_o.GetDataType(), gert::kOnDeviceHbm, io.td_o.GetFormat(), sym_gen, kernel_nodes);
  stage->out_tensor_node = nullptr;
  stage->s_out_tensor = 0;
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(stage->output_data_node);
    kernel_nodes->push_back(size_node);
    kernel_nodes->push_back(stage->alloc_out);
  }
}

// Intermediate op: CalcTensorSize -> AllocMemory. TensorRef keeps the unpacked
// (shape, addr) pair for Path C; DualStage ABI boxes it later.
void WireIntermediateOutput(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io,
                            const SharedRuntime &rt, uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                            DualStageNodes *stage) {
  uint64_t s_size = 0;
  const auto dtype = io.td_o.GetDataType();
  auto *dtype_node = AddConstNode(main, "const_out_dtype_" + tag, &dtype, sizeof(dtype), false, sym_gen,
                                  FxrtConstType::kGeDataType);
  kernel_nodes->push_back(dtype_node);
  const CalcTensorSizeInputs size_inputs{{dtype_node, 0, dtype_node->GetExtendInfo()->GetOutputSymbol(0)},
                                         {stage->infer, 0, stage->s_shape}};
  fx_converter::Node *calc_size = AddCalcTensorSizeNode(main, tag, size_inputs, sym_gen, &s_size);
  kernel_nodes->push_back(calc_size);

  uint64_t s_gtd = 0;
  stage->alloc_out =
    AddAllocMemoryNode(main, tag, io.empty_td, rt.s_allocator, rt.select_l2, s_size, calc_size, sym_gen, &s_gtd);
  stage->s_out_addr = s_gtd;
  kernel_nodes->push_back(stage->alloc_out);

  stage->out_tensor_node = nullptr;
  stage->s_out_tensor = 0;
}

// GE ExecuteOpPrepare/Launch require Tensor*. Box unpacked pairs once so Prepare
// and Launch share the same boxed producer. Already-boxed refs are reused.
bool BoxDualStageTensors(ge::Graph *main, BuiltExecuteGraph *built, const std::string &tag,
                         const DualStageIoDesc &io,
                         std::vector<TensorRef> *ins, const TensorRef *target, DualStageNodes *stage, uint64_t &sym_gen,
                         std::vector<fx_converter::Node *> *kernel_nodes, const OpCatalogEntry &entry) {
  if (ins == nullptr || stage == nullptr) {
    return false;
  }
  for (size_t i = 0; i < ins->size(); ++i) {
    // The matrix adapter creates aclTensor from TensorV2's live logical layout.
    // Boxing an internal dynamic view as V1 would lose the transposed strides.
    const bool matrix_layout = entry.behavior != nullptr && entry.behavior->infer_transpose_from_layout;
    const bool boxed = (entry.strided_inputs || matrix_layout)
                           ? EnsureStridedTensorRefBoxed(&(*ins)[i], main, built,
                                                         tag + "_in" + std::to_string(i), sym_gen, kernel_nodes)
                           : EnsureTensorRefBoxed(&(*ins)[i], main, tag + "_in" + std::to_string(i), sym_gen,
                                                  kernel_nodes);
    if (!boxed) {
      return false;
    }
  }
  TensorRef out_ref;
  if (target != nullptr) {
    const size_t ref_index = entry.output_ref_count == 1 && entry.output_refs != nullptr &&
                                     entry.output_refs[0].output == 0 && entry.output_refs[0].input >= 0
                                 ? static_cast<size_t>(CatalogRefInput(entry, 0))
                                 : ins->size();
    const bool output_only = entry.arity != 0 && ref_index == entry.arity;
    if (!output_only && ref_index < ins->size()) {
      // An aliased output must use the same boxed Tensor as its input. In-place
      // view ops need the strided descriptor on both Assign input and output.
      out_ref = (*ins)[ref_index];
    } else {
      out_ref = *target;
      const bool boxed = entry.strided_inputs
                             ? EnsureStridedTensorRefBoxed(&out_ref, main, built, tag + "_out", sym_gen, kernel_nodes)
                             : EnsureTensorRefBoxed(&out_ref, main, tag + "_out", sym_gen, kernel_nodes);
      if (!boxed) {
        return false;
      }
    }
  } else {
    out_ref.shape = io.td_o.GetShape().GetDims();
    out_ref.dtype = io.td_o.GetDataType();
    out_ref.unpacked.shape = TensorEndpoint{stage->infer, 0, stage->s_shape};
    out_ref.unpacked.addr = TensorEndpoint{stage->alloc_out, 0, stage->s_out_addr};
  }
  if (target == nullptr) {
    out_ref.allocation = out_ref.unpacked.addr;
    if (!EnsureTensorRefBoxed(&out_ref, main, tag + "_out", sym_gen, kernel_nodes)) {
      return false;
    }
  }
  stage->out_tensor_node = out_ref.producer;
  stage->s_out_tensor = out_ref.symbol;
  return stage->out_tensor_node != nullptr;
}

// ExecuteOpPrepare: OPP prepare_func -> opaque params + workspace size.
void WirePrepare(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io,
                 const std::vector<TensorRef> &ins, const SharedRuntime &rt, const DualStageOpFwk &fwk,
                 const OpCatalogEntry &entry, bool transpose_x1, bool transpose_x2, uint64_t &sym_gen,
                 DualStageNodes *stage) {
  const char *opp_type = entry.opp_type;
  // The synthetic raw-tensor attrs replace the complete CNI attr list. Only
  // transpose-only rows can use the two-bool layout; Addmm also needs its
  // beta/alpha floats before the transpose flags, so retain its original CNI.
  const bool matrix_semantics = entry.behavior != nullptr && entry.behavior->infer_transpose_from_layout;
  const bool transpose_only_attrs = matrix_semantics && entry.attr_count == 2U && entry.attrs != nullptr &&
                                    entry.attrs[0].type == GeAttrType::kBool &&
                                    entry.attrs[1].type == GeAttrType::kBool;
  auto prep_desc = std::make_shared<fx_converter::FxNodeSpec>("prepare_" + tag, "FxExecuteOpPrepare");
  if (opp_type != nullptr && std::string(opp_type) == "ReduceSum") {
    (void)prep_desc->SetAttr("keep_dims", true);
  } else if (transpose_only_attrs) {
    (void)prep_desc->SetAttr("transpose_x1", transpose_x1);
    (void)prep_desc->SetAttr("transpose_x2", transpose_x2);
  }
  prep_desc->AddInputDesc(io.empty_td);  // raw tensor layout
  for (size_t i = 0; i < io.n; ++i) {
    prep_desc->AddInputDesc(io.td_ins[i]);
  }
  prep_desc->AddInputDesc(io.td_o);      // out
  prep_desc->AddInputDesc(io.empty_td);  // options
  prep_desc->AddInputDesc(io.empty_td);  // prepare function
  prep_desc->AddInputDesc(io.empty_td);  // stream
  prep_desc->AddOutputDesc(io.empty_td);
  prep_desc->AddOutputDesc(io.empty_td);
  stage->prepare = fx_converter::AddNode(main, prep_desc);
  stage->s_params = sym_gen++;
  stage->s_ws_size = sym_gen++;

  std::vector<uint64_t> in_syms;
  in_syms.reserve(io.n + 4);
  std::transform(ins.begin(), ins.end(), std::back_inserter(in_syms),
                 [](const auto &in) { return in.symbol; });
  in_syms.push_back(stage->s_out_tensor);
  in_syms.push_back(rt.s_options);
  in_syms.push_back(fwk.s_prepare);
  in_syms.push_back(rt.s_stream);
  DualStageTensorLayout layout;
  layout.tensor_count = static_cast<uint32_t>(io.n + 1U);
  if (opp_type != nullptr && std::string(opp_type) == "ReduceSum") {
    layout.attr_count = 1U;
    layout.bool_attrs[0] = 1U;
  } else if (transpose_only_attrs) {
    layout.attr_count = 2U;
    layout.bool_attrs[0] = transpose_x1 ? 1U : 0U;
    layout.bool_attrs[1] = transpose_x2 ? 1U : 0U;
  }
  auto *layout_node = AddConstNode(main, "prepare_raw_tensor_layout_" + tag, &layout, sizeof(layout), false, sym_gen,
                                   FxrtConstType::kBytePtr);
  const auto s_layout = layout_node->GetExtendInfo()->GetOutputSymbol(0);
  in_syms.insert(in_syms.begin(), s_layout);
  WireSymbols(stage->prepare, in_syms, {stage->s_params, stage->s_ws_size});
  fx_converter::AddDataEdge(main, layout_node, 0, stage->prepare, 0);
  for (size_t i = 0; i < io.n; ++i) {
    const auto edge_status = fx_converter::AddDataEdge(main, ins[i].producer, 0, stage->prepare,
                                                       static_cast<int32_t>(i + 1U));
    if (edge_status != ge::GRAPH_SUCCESS) {
      RT_GLOG(ERROR) << "DualStage " << tag << " failed to connect input " << i << " producer="
                     << (ins[i].producer == nullptr ? "null" : ins[i].producer->GetName())
                     << " status=" << edge_status;
    }
  }
  const auto output_edge_status = fx_converter::AddDataEdge(main, stage->out_tensor_node, 0, stage->prepare,
                                                            static_cast<int32_t>(io.n + 1U));
  if (output_edge_status != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "DualStage " << tag << " failed to connect output producer="
                   << (stage->out_tensor_node == nullptr ? "null" : stage->out_tensor_node->GetName())
                   << " status=" << output_edge_status;
  }
  fx_converter::AddDataEdge(main, rt.c_options, 0, stage->prepare, static_cast<int32_t>(io.n + 2U));
  fx_converter::AddDataEdge(main, fwk.c_prepare, 0, stage->prepare, static_cast<int32_t>(io.n + 3U));
  fx_converter::AddDataEdge(main, rt.split_streams, 0, stage->prepare, static_cast<int32_t>(io.n + 4U));
  fx_converter::AddDataEdge(main, stage->infer, -1, stage->prepare, -1);
  // Final AllocModelOutTensor must run before Prepare (fills Tensor addr when null).
  if (stage->alloc_out != nullptr) {
    fx_converter::AddDataEdge(main, stage->alloc_out, -1, stage->prepare, -1);
  }
}

// Device workspace for prepare/launch (size from Prepare output 1).
void WireAllocWorkspace(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io,
                        const SharedRuntime &rt, uint64_t &sym_gen, DualStageNodes *stage) {
  auto ws_desc = std::make_shared<fx_converter::FxNodeSpec>("alloc_ws_" + tag, "AllocBatchHbm");
  ws_desc->AddInputDesc(io.empty_td);
  ws_desc->AddInputDesc(io.empty_td);
  ws_desc->AddOutputDesc(io.empty_td);
  stage->alloc_ws = fx_converter::AddNode(main, ws_desc);
  stage->s_ws_addr = sym_gen++;
  WireSymbols(stage->alloc_ws, {rt.s_allocator, stage->s_ws_size}, {stage->s_ws_addr});
  fx_converter::AddDataEdge(main, rt.select_l2, 0, stage->alloc_ws, 0);
  fx_converter::AddDataEdge(main, stage->prepare, 1, stage->alloc_ws, 1);
}

// ExecuteOpLaunch: OPP launch_func on device stream.
void WireLaunch(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io,
                const std::vector<TensorRef> &ins, const SharedRuntime &rt, const DualStageOpFwk &fwk,
                uint64_t &sym_gen, DualStageNodes *stage) {
  auto launch_desc = std::make_shared<fx_converter::FxNodeSpec>("launch_" + tag, "FxExecuteOpLaunch");
  launch_desc->AddInputDesc(io.empty_td);  // raw tensor layout
  for (size_t i = 0; i < io.n; ++i) {
    launch_desc->AddInputDesc(io.td_ins[i]);
  }
  launch_desc->AddInputDesc(io.td_o);      // out
  launch_desc->AddInputDesc(io.empty_td);  // params
  launch_desc->AddInputDesc(io.empty_td);  // ws_addr
  launch_desc->AddInputDesc(io.empty_td);  // ws_size
  launch_desc->AddInputDesc(io.empty_td);  // stream
  launch_desc->AddInputDesc(io.empty_td);  // launch function
  stage->launch = fx_converter::AddNode(main, launch_desc);

  std::vector<uint64_t> in_syms;
  in_syms.reserve(io.n + 6);
  std::transform(ins.begin(), ins.end(), std::back_inserter(in_syms),
                 [](const auto &in) { return in.symbol; });
  in_syms.push_back(stage->s_out_tensor);
  in_syms.push_back(stage->s_params);
  in_syms.push_back(stage->s_ws_addr);
  in_syms.push_back(stage->s_ws_size);
  in_syms.push_back(rt.s_stream);
  in_syms.push_back(fwk.s_launch);
  DualStageTensorLayout layout;
  layout.tensor_count = static_cast<uint32_t>(io.n + 1U);
  auto *layout_node = AddConstNode(main, "launch_raw_tensor_layout_" + tag, &layout, sizeof(layout), false, sym_gen,
                                   FxrtConstType::kBytePtr);
  const auto s_layout = layout_node->GetExtendInfo()->GetOutputSymbol(0);
  in_syms.insert(in_syms.begin(), s_layout);
  WireSymbols(stage->launch, in_syms, {});
  fx_converter::AddDataEdge(main, layout_node, 0, stage->launch, 0);
  for (size_t i = 0; i < io.n; ++i) {
    fx_converter::AddDataEdge(main, ins[i].producer, 0, stage->launch, static_cast<int32_t>(i + 1U));
  }
  fx_converter::AddDataEdge(main, stage->out_tensor_node, 0, stage->launch, static_cast<int32_t>(io.n + 1U));
  fx_converter::AddDataEdge(main, stage->prepare, 0, stage->launch, static_cast<int32_t>(io.n + 2U));
  fx_converter::AddDataEdge(main, stage->alloc_ws, 0, stage->launch, static_cast<int32_t>(io.n + 3U));
  fx_converter::AddDataEdge(main, stage->prepare, 1, stage->launch, static_cast<int32_t>(io.n + 4U));
  fx_converter::AddDataEdge(main, rt.split_streams, 0, stage->launch, static_cast<int32_t>(io.n + 5U));
  fx_converter::AddDataEdge(main, fwk.c_launch, 0, stage->launch, static_cast<int32_t>(io.n + 6U));
}

// Free AllocBatchHbm after Launch (GE FreeWorkspaceMem / CreateVoidGuarder).
void WireFreeWorkspace(ge::Graph *main, const std::string &tag, const DualStageIoDesc &io, DualStageNodes *stage,
                       std::vector<fx_converter::Node *> *kernel_nodes) {
  (void)AddFreeBatchHbmNode(main, tag, io.empty_td, stage->s_ws_addr, stage->alloc_ws, stage->launch, kernel_nodes);
}

void CollectStageKernels(const DualStageNodes &stage, std::vector<fx_converter::Node *> *kernel_nodes) {
  kernel_nodes->push_back(stage.infer);
  kernel_nodes->push_back(stage.prepare);
  kernel_nodes->push_back(stage.alloc_ws);
  kernel_nodes->push_back(stage.launch);
}

// Assemble the standard dual-stage subgraph for one logical op (N tensor inputs).
// Mutates *ins in place so boxed Tensor* producers are reused by later DualStage ops.
DualStageNodes LowerNaryDualStage(ge::Graph *main, BuiltExecuteGraph *built, const std::string &tag,
                                  std::vector<TensorRef> *ins,
                                  const std::vector<int64_t> &out_shape, ge::DataType out_dtype, bool is_final,
                                  const TensorRef *target, const SharedRuntime &rt, const DualStageOpFwk &fwk,
                                  fx_converter::Node *c_rule, fx_converter::Node *bind_symbols, const OpCatalogEntry &entry,
                                  bool transpose_x1, bool transpose_x2, bool es_model, uint64_t &sym_gen,
                                  std::vector<fx_converter::Node *> *kernel_nodes) {
  DualStageNodes stage;
  if (ins == nullptr) {
    return stage;
  }
  const DualStageIoDesc io = MakeIoDescs(*ins, out_shape, out_dtype);
  WireInferShape(main, tag, io, *ins, fwk, c_rule, bind_symbols, sym_gen, &stage);
  // Explicit out= storage is owned by the caller; never allocate or free it.
  if (target == nullptr) {
    if (is_final) {
      WireFinalOutput(main, tag, io, rt, sym_gen, kernel_nodes, &stage, es_model);
    } else {
      WireIntermediateOutput(main, tag, io, rt, sym_gen, kernel_nodes, &stage);
    }
  }
  if (!BoxDualStageTensors(main, built, tag, io, ins, target, &stage, sym_gen, kernel_nodes, entry)) {
    return stage;
  }
  WirePrepare(main, tag, io, *ins, rt, fwk, entry, transpose_x1, transpose_x2, sym_gen, &stage);
  WireAllocWorkspace(main, tag, io, rt, sym_gen, &stage);
  WireLaunch(main, tag, io, *ins, rt, fwk, sym_gen, &stage);
  WireFreeWorkspace(main, tag, io, &stage, kernel_nodes);
  if (is_final && target == nullptr) {
    // Ensure updates OutputData shape/dtype and shares/copies GTD (GE NetOutput contract).
    const EnsureOutMemoryInputs ensure_inputs{
        {stage.infer, 0, stage.s_shape},
        {stage.alloc_out, 0, stage.s_out_addr},
        stage.out_fields,
        {rt.split_streams, 0, rt.s_stream},
        {stage.output_data_node, 0, stage.s_output_data},
        {rt.select_l2, 0, rt.s_allocator},
    };
    (void)AddEnsureTensorAtOutMemoryNode(main, tag, ensure_inputs, stage.launch, kernel_nodes);
  }
  CollectStageKernels(stage, kernel_nodes);
  return stage;
}

std::shared_ptr<FxShapeRule> BuildShapeRuleForEntry(const OpCatalogEntry &entry, const fxrt::ir::NodePtr &node,
                                                    FxSymbolEnv *env) {
  return BuildOutputSymbolicShapeRule(node, env, entry.logical_name);
}

std::string FormatOppNotReadyError(const OpCatalogEntry &entry) {
  return std::string("unsupported op '") + entry.logical_name + "' (OPP type '" + entry.opp_type +
         "' has no dual-stage prepare/launch)";
}

struct CollectedInputs {
  std::vector<fxrt::ir::NodePtr> nodes;
  CatalogAttrExtras attrs;
  std::vector<TensorRef> refs;
  TensorRef target;
  bool has_target = false;
  int64_t axis = std::numeric_limits<int64_t>::min();
  bool transpose_x1 = false;
  bool transpose_x2 = false;
};

void ApplyMatMulLayoutFromRefs(const OpCatalogEntry &entry, CollectedInputs *out) {
  if (out == nullptr || out->refs.size() < 2U) {
    return;
  }
  if (entry.behavior == nullptr || !entry.behavior->infer_transpose_from_layout) {
    return;
  }
  out->transpose_x1 = IsLastTwoDimsTransposed(out->refs[0]);
  out->transpose_x2 = IsLastTwoDimsTransposed(out->refs[1]);
  if (out->transpose_x1 || out->transpose_x2) {
    RT_GLOG(INFO) << "DualStage " << entry.logical_name << " transpose_x1=" << out->transpose_x1
                  << " transpose_x2=" << out->transpose_x2;
  }
}

void RegisterAxesRefresh(BuiltExecuteGraph *built, void *host, const std::vector<int64_t> &dims,
                         const std::vector<fxrt::ir::SymbolicExprPtr> &exprs) {
  if (built == nullptr || host == nullptr ||
      std::none_of(exprs.begin(), exprs.end(),
                   [](const fxrt::ir::SymbolicExprPtr &expr) { return expr != nullptr; })) {
    return;
  }
  // A Dynamo-lifted axis is a SymInt. The const payload is filled with the sentinel until
  // Execute binds that symbol; reduce axes may be 0 or negative, so skip the broadcast check.
  BuiltExecuteGraph::AiCoreShapeTensorRefresh refresh;
  refresh.host = host;
  refresh.size_template = dims;
  refresh.payload_type = ge::DT_INT64;
  refresh.axis_exprs = exprs;
  refresh.allow_nonpositive = true;
  built->aicore_shape_refreshes.push_back(std::move(refresh));
}

bool MaterializeHostInt64Axes(ge::Graph *main, BuiltExecuteGraph *built, const std::vector<int64_t> &dims,
                              const std::string &tag, uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                              void **host_out, TensorRef *out, std::string *err) {
  if (main == nullptr || built == nullptr || out == nullptr) {
    return false;
  }
  const size_t nbytes = dims.size() * sizeof(int64_t);
  auto payload = std::make_shared<std::vector<uint8_t>>(nbytes);
  if (!dims.empty()) {
    std::memcpy(payload->data(), dims.data(), nbytes);
  }
  built->binary_bindata_blobs.push_back(payload);
  if (host_out != nullptr) {
    *host_out = payload->empty() ? nullptr : static_cast<void *>(payload->data());
  }

  const int64_t rank = static_cast<int64_t>(dims.size());
  gert::StorageShape shape({rank}, {rank});
  gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
  gert::Tensor host_tensor(shape, fmt, gert::kOnHost, ge::DT_INT64, payload->data());
  auto tensor_blob = std::make_shared<std::vector<uint8_t>>(sizeof(gert::Tensor));
  std::memcpy(tensor_blob->data(), &host_tensor, sizeof(gert::Tensor));
  built->binary_bindata_blobs.push_back(tensor_blob);

  fx_converter::Node *c_axes =
    AddConstNode(main, "const_dual_axes_" + tag, tensor_blob->data(), tensor_blob->size(), false, sym_gen,
                 FxrtConstType::kGertTensorPtr);
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(c_axes);
  }
  out->producer = c_axes;
  out->symbol = c_axes->GetExtendInfo()->GetOutputSymbol(0);
  out->shape = {rank};
  out->dtype = ge::DT_INT64;
  (void)err;
  return true;
}

bool AppendResolvedTensorInput(const fxrt::ir::NodePtr &in_node, const LoweringContext &ctx, const std::string &tag,
                               CollectedInputs *out, std::string *err) {
  const std::string in_name = (*ctx.name_of)[in_node.get()];
  const auto it = ctx.tensors->find(in_name);
  if (it == ctx.tensors->end()) {
    if (err != nullptr) {
      *err = std::string("dual-stage lowering: unresolved input '") + in_name + "' for " + tag;
    }
    return false;
  }
  out->nodes.push_back(in_node);
  out->refs.push_back(it->second);
  return true;
}

bool AppendShapeListAxes(const fxrt::ir::NodePtr &in_node, const LoweringContext &ctx, const std::string &tag,
                         CollectedInputs *out, std::string *err) {
  const auto tup = in_node->output->ToTuple();
  if (tup == nullptr) {
    if (err != nullptr) {
      *err = "dual-stage lowering: null shape-list tuple for " + tag;
    }
    return false;
  }
  if (ctx.symbol_env == nullptr) {
    if (err != nullptr) {
      *err = "dual-stage lowering: missing symbol env for " + tag;
    }
    return false;
  }
  std::vector<int64_t> dims;
  std::vector<fxrt::ir::SymbolicExprPtr> exprs;
  if (!ReadIntListWithSymbols(tup, *ctx.symbol_env, &dims, &exprs)) {
    if (err != nullptr) {
      *err = "dual-stage lowering: unparsable symbolic axis for " + tag;
    }
    return false;
  }
  TensorRef axes_ref;
  void *host = nullptr;
  if (!MaterializeHostInt64Axes(ctx.main, ctx.built, dims, tag, *ctx.sym_gen, ctx.kernel_nodes, &host, &axes_ref,
                                err)) {
    if (err != nullptr && err->empty()) {
      *err = "dual-stage lowering: host axes materialize failed for " + tag;
    }
    return false;
  }
  RegisterAxesRefresh(ctx.built, host, dims, exprs);
  out->nodes.push_back(in_node);
  out->refs.push_back(axes_ref);
  return true;
}

enum class DualInputKind : uint8_t { kSkip, kAxisInt, kBoolInput, kShapeList, kTensor };

DualInputKind ClassifyDualInput(const fxrt::ir::NodePtr &in_node, const OpCatalogEntry &entry) {
  if (in_node == nullptr || in_node->output == nullptr) {
    return DualInputKind::kSkip;
  }
  if (in_node->output->IsInt()) {
    return DualInputKind::kAxisInt;
  }
  if (in_node->output->IsBool()) {
    return DualInputKind::kBoolInput;
  }
  if (in_node->output->IsTuple() && entry.shape_input_index >= 0) {
    return DualInputKind::kShapeList;
  }
  if (in_node->output->IsTensor()) {
    return DualInputKind::kTensor;
  }
  return DualInputKind::kSkip;
}

bool ConsumeDualInput(const fxrt::ir::NodePtr &in_node, DualInputKind kind, const OpCatalogEntry &entry,
                      const LoweringContext &ctx, const std::string &tag, CollectedInputs *out,
                      std::string *err) {
  switch (kind) {
    case DualInputKind::kSkip:
      return true;
    case DualInputKind::kAxisInt:
      out->axis = in_node->output->ToInt();
      return true;
    case DualInputKind::kBoolInput:
      // Dropping a bool must be declared on the row (ReduceSum: the rewrite applies
      // keepdim as a following view, so SkipBool drops the keepdim Bool). An
      // undeclared bool input is a contract violation; fail loud instead of
      // silently swallowing it.
      if (HasInputRole(entry.behavior, IrInputKind::kBool, InputRole::kSkip)) {
        return true;
      }
      if (err != nullptr) {
        *err = std::string("dual-stage lowering: undeclared bool input for '") + entry.logical_name +
               "'; declare SkipBool on the catalog row";
      }
      return false;
    case DualInputKind::kShapeList:
      return AppendShapeListAxes(in_node, ctx, tag, out, err);
    case DualInputKind::kTensor:
      return AppendResolvedTensorInput(in_node, ctx, tag, out, err);
  }
  return false;
}

// Reuse the catalog's output/input alias contract. A ref within the input arity
// aliases a compute input; a ref at arity names an optional output-only tail.
bool BindOutputRef(const OpCatalogEntry &entry, CollectedInputs *out, std::string *err) {
  if (entry.output_ref_count == 0) return true;
  if (entry.output_ref_count != 1 || entry.output_refs == nullptr || entry.output_refs[0].output != 0 ||
      entry.output_refs[0].input < 0) {
    if (err != nullptr)
      *err = "dual-stage lowering: expected one valid output ref for " + std::string(entry.logical_name);
    return false;
  }
  const size_t index = static_cast<size_t>(CatalogRefInput(entry, 0));
  const bool output_only = entry.arity != 0 && index == entry.arity;
  if (output_only && index == out->refs.size()) return true;
  if (index >= out->refs.size()) {
    if (err != nullptr)
      *err = "dual-stage lowering: ref input is outside the collected inputs for " + std::string(entry.logical_name);
    return false;
  }
  out->target = out->refs[index];
  out->has_target = true;
  if (output_only) {
    out->refs.erase(out->refs.begin() + index);
    out->nodes.erase(out->nodes.begin() + index);
  }
  return true;
}

bool CollectTensorInputs(const fxrt::ir::NodePtr &node, const OpCatalogEntry &entry, const LoweringContext &ctx,
                         const std::string &tag, CollectedInputs *out, std::string *err) {
  std::array<int64_t, kIrInputKindCount> ordinals{};
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const auto &in_node = node->inputs[i];
    if (in_node != nullptr && in_node->output != nullptr) {
      const auto &value = in_node->output;
      // Strings share the kDouble bucket: IrInputKind has no string kind. The text
      // borrows IR storage for the convert scope, same as the aicore collector.
      if (value->IsInt() || value->IsBool() || value->IsDouble() || value->IsString()) {
        const auto kind = value->IsBool() ? IrInputKind::kBool :
                          (value->IsInt() ? IrInputKind::kInt : IrInputKind::kDouble);
        const auto ordinal = ordinals[static_cast<size_t>(kind)]++;
        const auto *role = FindInputRoleAt(entry.behavior, kind, InputRole::kNamedAttr, ordinal);
        if (role != nullptr) {
          if (value->IsBool()) out->attrs.SetBool(role->attr_name, value->ToBool());
          else if (value->IsInt()) out->attrs.SetInt(role->attr_name, value->ToInt());
          else if (value->IsString()) out->attrs.SetStr(role->attr_name, value->ToString().c_str());
          else out->attrs.SetDouble(role->attr_name, value->ToDouble());
          continue;
        }
      }
    }
    const DualInputKind kind = ClassifyDualInput(in_node, entry);
    if (!ConsumeDualInput(in_node, kind, entry, ctx, tag, out, err)) {
      return false;
    }
  }
  if (!BindOutputRef(entry, out, err)) {
    return false;
  }
  if (out->refs.empty()) {
    if (err != nullptr) {
      *err = std::string("dual-stage lowering: no tensor inputs for ") + tag;
    }
    return false;
  }
  if (entry.arity != 0 && out->refs.size() != entry.arity) {
    if (err != nullptr) {
      *err = std::string("dual-stage lowering: arity mismatch for ") + entry.logical_name;
    }
    return false;
  }
  return true;
}

void FillLoweringResult(const OpCatalogEntry &entry, const fxrt::ir::NodePtr &node, const DualStageNodes &stage,
                        const CollectedInputs &inputs, const std::vector<int64_t> &out_shape, ge::DataType out_dtype,
                        bool is_final, LoweringResult *out) {
  out->outputs.resize(1);
  TensorRef &out0 = out->outputs[0];
  // Final AllocModelOutTensor is model-owned; do not FreeMemory it.
  if (!is_final && !inputs.has_target) {
    out->mem_to_frees.push_back(stage.alloc_out);
  }
  if (inputs.has_target) {
    out0 = inputs.target;
  } else if (is_final) {
    out0.producer = stage.output_data_node;
    out0.symbol = stage.s_output_data;
  } else {
    out0.producer = stage.out_tensor_node;
    out0.symbol = stage.s_out_tensor;
  }
  out0.shape = out_shape;
  out0.dtype = out_dtype;
  // Path C Launch can skip SplitDataTensor: shape = InferShape, addr = Alloc.
  if (!inputs.has_target) {
    out0.unpacked.shape = TensorEndpoint{stage.infer, 0, stage.s_shape};
    out0.unpacked.addr = TensorEndpoint{stage.alloc_out, 0, stage.s_out_addr};
    out0.allocation = out0.unpacked.addr;
  }
  out->launch = stage.launch;
  out->prepare = stage.prepare;
  FillLoweringCniCommon(entry, inputs.refs, out_shape, out_dtype, out);
  out->cni.compute_nodes = {stage.infer, stage.alloc_out, stage.prepare, stage.alloc_ws, stage.launch};
  out->cni.axis = inputs.axis;
  out->cni.attrs = inputs.attrs;
  for (uint8_t i = 0; i < entry.attr_count; ++i) {
    const char *name = entry.attrs[i].name;
    const auto it = node->attrs.find(name);
    if (it == node->attrs.end() || it->second == nullptr) continue;
    if (it->second->IsInt()) out->cni.attrs.SetInt(name, it->second->ToInt());
    if (it->second->IsDouble()) out->cni.attrs.SetDouble(name, it->second->ToDouble());
  }
  if (inputs.transpose_x1 || inputs.transpose_x2) {
    out->cni.attrs.SetBool("transpose_x1", inputs.transpose_x1);
    out->cni.attrs.SetBool("transpose_x2", inputs.transpose_x2);
    out->cni.attrs.SetBool("adj_x1", inputs.transpose_x1);
    out->cni.attrs.SetBool("adj_x2", inputs.transpose_x2);
  }
}

// Share boxed DualStage inputs with later consumers of the same TensorRef.
void WriteBackBoxedInputs(const CollectedInputs &inputs, LoweringContext &ctx) {
  if (ctx.tensors == nullptr || ctx.name_of == nullptr) {
    return;
  }
  for (size_t i = 0; i < inputs.nodes.size() && i < inputs.refs.size(); ++i) {
    const auto &in_node = inputs.nodes[i];
    if (in_node == nullptr) {
      continue;
    }
    const auto it_name = ctx.name_of->find(in_node.get());
    if (it_name == ctx.name_of->end()) {
      continue;
    }
    auto it = ctx.tensors->find(it_name->second);
    if (it != ctx.tensors->end()) {
      auto &cached = it->second;
      const auto &boxed = inputs.refs[i];
      if (cached.producer != nullptr && std::string(cached.producer->GetType()) == "OutputData") {
        // Keep the model-output binding. Replacing it with BuildTensor leaves
        // the original AllocModelOutTensor attached to an unbound shell, whose
        // buffer survives Execute and cannot grow with a dynamic batch.
        cached.tiling_ep = TensorEndpoint{boxed.producer, 0, boxed.symbol};
      } else {
        cached = boxed;
      }
    }
  }
}

}  // namespace

bool IsOppDualStageReady(const char *opp_type) {
  if (opp_type == nullptr) {
    return false;
  }
  gert::OpImplSpaceRegistryV2Ptr reg;
  if (!EnsureOppRegistryReady(&reg) || reg == nullptr) {
    return false;
  }
  const auto *funcs = reg->GetOpImpl(opp_type);
  return funcs != nullptr && funcs->op_execute_prepare_func != nullptr && funcs->op_execute_launch_func != nullptr;
}

const char *DualStageOpLowering::Name() const { return "dual_stage_opp"; }

InputLayoutKind DualStageOpLowering::InputLayout(const fxrt::ir::NodePtr &node, size_t index) const {
  const auto *entry = Entry(node);
  if (entry != nullptr && entry->strided_inputs) return InputLayoutKind::kStridedTensor;
  if (index < 2U && entry != nullptr && entry->behavior != nullptr &&
      entry->behavior->infer_transpose_from_layout) return InputLayoutKind::kMatrix;
  // Stride-aware adapters opt in through the catalog and receive TensorV2.
  // Other adapters retain the dense input contract.
  return InputLayoutKind::kDense;
}

const OpCatalogEntry *DualStageOpLowering::Entry(const fxrt::ir::NodePtr &node) const {
  return LookupOpCatalog(node);
}

bool DualStageOpLowering::Match(const fxrt::ir::NodePtr &node, const LoweringContext *) const {
  const OpCatalogEntry *entry = Entry(node);
  return entry != nullptr && entry->lowering == LoweringKind::kDualStageAclnn && IsOppDualStageReady(entry->opp_type);
}

bool DualStageOpLowering::Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                                LoweringResult *out, std::string *err) {
  if (out == nullptr || !ValidateLoweringContext(ctx, true, "dual-stage lowering", err)) {
    return false;
  }
  // 1) Catalog + readiness (Match already checked; re-check for clear errors).
  const OpCatalogEntry *entry = Entry(node);
  if (entry == nullptr || entry->lowering != LoweringKind::kDualStageAclnn) {
    if (err != nullptr) {
      *err = "dual-stage lowering: node not in DualStageAclnn OpCatalog";
    }
    return false;
  }
  if (!IsOppDualStageReady(entry->opp_type)) {
    if (err != nullptr) {
      *err = FormatOppNotReadyError(*entry);
    }
    return false;
  }

  // 2) Resolve (or reuse) OPP prepare/launch Const Fwk for this opp_type.
  const DualStageOpFwk *fwk = nullptr;
  if (!EnsureFwk(ctx, entry->opp_type, &fwk, err) || fwk == nullptr) {
    return false;
  }

  // 3) Map MRT tensor inputs to already-lowered TensorRefs (axis Int optional).
  CollectedInputs inputs;
  if (!CollectTensorInputs(node, *entry, ctx, tag, &inputs, err)) {
    return false;
  }
  ApplyMatMulLayoutFromRefs(*entry, &inputs);
  if (CatalogOutputArity(*entry) > 1U) {
    return LowerMultiOutputDualStage(node, tag, ctx, *entry, *fwk, &inputs.refs, inputs.attrs, err, out);
  }
  // 4) Shape rule Const for Expression InferShape (shared SymbolEnv only; no per-op fallback).
  auto rule = BuildShapeRuleForEntry(*entry, node, ctx.symbol_env);
  if (rule == nullptr) {
    if (err != nullptr) {
      *err = std::string("dual-stage lowering: no symbolic output shape / SymbolEnv seeds for '") +
             entry->logical_name + "'";
    }
    return false;
  }
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(std::move(rule));
  fx_converter::Node *c_rule =
    AddConstNode(ctx.main, "const_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false, *ctx.sym_gen,
                 FxrtConstType::kVoidPtr);
  ctx.kernel_nodes->push_back(c_rule);

  // 5) Wire Infer -> Out -> Prepare -> Ws -> Launch; fill CNI for ModelV2.
  const auto out_shape = ShapeOfNode(node);
  const ge::DataType out_dtype = DtypeOfNode(node);
  DualStageNodes stage =
      LowerNaryDualStage(ctx.main, ctx.built, tag, &inputs.refs, out_shape, out_dtype, IsFinalOutput(ctx),
                         inputs.has_target ? &inputs.target : nullptr, *ctx.rt, *fwk, c_rule, ctx.bind_symbols, *entry,
                         inputs.transpose_x1, inputs.transpose_x2,
                         ctx.built != nullptr && ctx.built->fx_builder != nullptr, *ctx.sym_gen, ctx.kernel_nodes);
  if (stage.prepare == nullptr || stage.launch == nullptr) {
    if (err != nullptr) {
      *err = "dual-stage lowering: failed to box Tensor arguments for " + tag;
    }
    return false;
  }
  WriteBackBoxedInputs(inputs, ctx);
  FillLoweringResult(*entry, node, stage, inputs, out_shape, out_dtype, IsFinalOutput(ctx), out);
  return true;
}

bool DualStageOpLowering::EnsureFwk(LoweringContext &ctx, const char *opp_type, const DualStageOpFwk **fwk_out,
                                    std::string *err) {
  auto it = ctx.fwk_by_op->find(opp_type);
  if (it != ctx.fwk_by_op->end()) {
    *fwk_out = &it->second;
    return true;
  }
  OppDualStageFuncs funcs;
  if (!ResolveOppDualStageBinding(opp_type, &funcs)) {
    if (err != nullptr) {
      *err = std::string("OPP binding failed for type=") + opp_type;
    }
    return false;
  }
  DualStageOpFwk fwk = BuildDualStageOpFwk(ctx.main, funcs, *ctx.sym_gen, ctx.kernel_nodes);
  auto inserted = ctx.fwk_by_op->emplace(opp_type, std::move(fwk));
  *fwk_out = &inserted.first->second;
  return true;
}

}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(::fx_converter::kDefaultDualStagePriority,
                        std::make_unique<::fx_converter::DualStageOpLowering>());
