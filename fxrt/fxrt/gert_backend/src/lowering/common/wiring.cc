#include "lowering/common/wiring.h"
#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/runtime/tensor.h"

#include <algorithm>
#include <iterator>
#include <limits>

#include "lowering/common/view_utils.h"
#include "register/op_catalog.h"


namespace fx_converter {
namespace {

constexpr int64_t kRuntimeDenseStrideSentinel = std::numeric_limits<int64_t>::min();

}  // namespace

bool AddRuntimeLayoutTensor(ge::Graph *main, BuiltExecuteGraph *built, const std::string &tag,
                            const std::vector<int64_t> &values,
                            const std::vector<fxrt::ir::SymbolicExprPtr> &expressions, uint64_t &sym_gen,
                            std::vector<fx_converter::Node *> *kernel_nodes, TensorEndpoint *endpoint) {
  if (main == nullptr || built == nullptr || endpoint == nullptr || values.empty() ||
      values.size() != expressions.size()) {
    return false;
  }
  auto initial_values = values;
  for (size_t i = 0; i < initial_values.size(); ++i) {
    if (expressions[i] != nullptr && initial_values[i] < 0) initial_values[i] = 0;
  }
  auto payload = std::make_shared<std::vector<int64_t>>(std::move(initial_values));
  built->runtime_layout_payloads.push_back(payload);
  if (std::any_of(expressions.begin(), expressions.end(), [](const auto &expr) { return expr != nullptr; })) {
    BuiltExecuteGraph::AiCoreShapeTensorRefresh refresh;
    refresh.host = payload->data();
    refresh.size_template = values;
    refresh.payload_type = ge::DT_INT64;
    refresh.axis_exprs = expressions;
    refresh.allow_nonpositive = true;
    built->aicore_shape_refreshes.push_back(std::move(refresh));
  }

  gert::StorageShape shape;
  shape.MutableStorageShape().SetDimNum(1U);
  shape.MutableStorageShape().SetDim(0U, static_cast<int64_t>(values.size()));
  shape.MutableOriginShape().SetDimNum(1U);
  shape.MutableOriginShape().SetDim(0U, static_cast<int64_t>(values.size()));
  const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
  gert::Tensor tensor(shape, format, gert::kOnHost, ge::DT_INT64, payload->data());
  auto *constant = AddConstNode(main, "const_runtime_layout_" + tag, &tensor, sizeof(tensor), false, sym_gen,
                                FxrtConstType::kGertTensorPtr);
  if (constant == nullptr || constant->GetExtendInfo() == nullptr) return false;
  if (kernel_nodes != nullptr) kernel_nodes->push_back(constant);
  *endpoint = {constant, 0, constant->GetExtendInfo()->GetOutputSymbol(0)};
  return true;
}

fx_converter::Node *AddInferShapeNode(ge::Graph *main, const std::string &tag,
                                const std::vector<TensorRef> &inputs, const std::vector<ge::TensorDesc> &td_outs,
                                const ge::TensorDesc &empty_td, fx_converter::Node *c_rule, fx_converter::Node *c_infer,
                                uint64_t &sym_gen, std::vector<uint64_t> *s_shapes) {
  auto infer_desc = std::make_shared<FxNodeSpec>("infer_" + tag, "InferShape");
  // Keep one input slot per logical tensor for GE scheduling, but feed the
  // unpacked StorageShape value when available instead of boxing a Tensor.
  for (const auto &in : inputs) {
    infer_desc->AddInputDesc(MakeTensorDesc(PhysicalShape(in), in.dtype, in.format));
  }
  infer_desc->AddInputDesc(empty_td);  // rule
  infer_desc->AddInputDesc(empty_td);  // infer_func
  for (const auto &td_o : td_outs) {
    infer_desc->AddOutputDesc(td_o);
  }
  fx_converter::Node *infer = AddNode(main, infer_desc);

  std::vector<uint64_t> out_syms;
  out_syms.reserve(td_outs.size());
  for (size_t o = 0; o < td_outs.size(); ++o) {
    out_syms.push_back(sym_gen++);
  }
  if (s_shapes != nullptr) {
    *s_shapes = out_syms;
  }

  std::vector<uint64_t> in_syms;
  in_syms.reserve(inputs.size() + 2);
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_syms),
                 [](const TensorRef &in) { return in.HasUnpackedPair() ? in.unpacked.shape.sym : in.symbol; });
  in_syms.push_back(c_rule->GetExtendInfo()->GetOutputSymbol(0));
  in_syms.push_back(c_infer->GetExtendInfo()->GetOutputSymbol(0));
  WireSymbols(infer, in_syms, out_syms);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const TensorEndpoint shape_ep = inputs[i].HasUnpackedPair()
                                        ? inputs[i].unpacked.shape
                                        : TensorEndpoint{inputs[i].producer, 0, inputs[i].symbol};
    if (shape_ep.Valid()) {
      AddDataEdge(main, shape_ep.node, shape_ep.out, infer, static_cast<int32_t>(i));
    }
  }
  AddDataEdge(main, c_rule, 0, infer, static_cast<int32_t>(inputs.size()));
  AddDataEdge(main, c_infer, 0, infer, static_cast<int32_t>(inputs.size() + 1));
  return infer;
}

fx_converter::Node *AddInferShapeNode(ge::Graph *main, const std::string &tag,
                                const std::vector<TensorRef> &inputs, const ge::TensorDesc &td_o,
                                const ge::TensorDesc &empty_td, fx_converter::Node *c_rule, fx_converter::Node *c_infer,
                                uint64_t &sym_gen, uint64_t *s_shape) {
  std::vector<uint64_t> shapes;
  fx_converter::Node *infer =
      AddInferShapeNode(main, tag, inputs, std::vector<ge::TensorDesc>{td_o}, empty_td, c_rule, c_infer, sym_gen,
                        &shapes);
  if (s_shape != nullptr && !shapes.empty()) {
    *s_shape = shapes[0];
  }
  return infer;
}

fx_converter::Node *AddOutputDataNode(ge::Graph *main, const std::string &tag, const ge::TensorDesc &td_o,
                                uint64_t &sym_gen, uint64_t *s_out_tensor) {
  auto out_data_desc = std::make_shared<FxNodeSpec>("user_output_" + tag, "OutputData");
  out_data_desc->AddOutputDesc(td_o);
  fx_converter::Node *out_tensor_node = AddNode(main, out_data_desc);
  const uint64_t out_tensor_sym = sym_gen++;
  if (s_out_tensor != nullptr) {
    *s_out_tensor = out_tensor_sym;
  }
  WireSymbols(out_tensor_node, {}, {out_tensor_sym});
  return out_tensor_node;
}

static void AddEmptyInputDescs(FxNodeSpec *desc, size_t count) {
  const ge::TensorDesc empty_td;
  for (size_t i = 0; i < count; ++i) {
    desc->AddInputDesc(empty_td);
  }
}

static void WireEndpointInputs(ge::Graph *main, fx_converter::Node *node, const TensorEndpoint *ports, size_t count,
                        const std::vector<uint64_t> &out_syms) {
  std::vector<uint64_t> in_syms;
  in_syms.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    in_syms.push_back(ports[i].sym);
  }
  WireSymbols(node, in_syms, out_syms);
  for (size_t i = 0; i < count; ++i) {
    if (ports[i].node != nullptr) {
      AddDataEdge(main, ports[i].node, ports[i].out, node, static_cast<int32_t>(i));
    }
  }
}

fx_converter::Node *AddCalcTensorSizeNode(ge::Graph *main, const std::string &tag, const CalcTensorSizeInputs &inputs,
                                    uint64_t &sym_gen, uint64_t *s_size) {
  auto calc_desc = std::make_shared<FxNodeSpec>("calc_size_" + tag, "CalcTensorSizeFromStorage");
  AddEmptyInputDescs(calc_desc.get(), 2);
  calc_desc->AddOutputDesc(ge::TensorDesc());
  fx_converter::Node *calc = AddNode(main, calc_desc);
  const uint64_t size_sym = sym_gen++;
  if (s_size != nullptr) {
    *s_size = size_sym;
  }
  const TensorEndpoint ports[] = {inputs.dtype, inputs.shape};
  WireEndpointInputs(main, calc, ports, std::size(ports), {size_sym});
  return calc;
}

size_t EsModelOutputCapacityBytes(const std::vector<int64_t> &shape, ge::DataType dtype) {
  // Dynamic OM2 outputs have no concrete compile-time shape. Keep enough room
  // for the large dynamic matmul outputs used by the runner; GE validates the
  // runtime shape against this externally supplied output buffer before it can
  // adopt the operator's result storage.
  constexpr size_t kMinimumCapacity = 64U << 20U;
  constexpr size_t kCompileShapeMultiplier = 4U;
  int64_t elements = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim) {
      return kMinimumCapacity;
    }
    elements *= dim;
  }
  const int64_t byte_size = ge::GetSizeInBytes(elements, dtype);
  if (byte_size <= 0 || static_cast<uint64_t>(byte_size) > std::numeric_limits<size_t>::max() /
                                                              kCompileShapeMultiplier) {
    return kMinimumCapacity;
  }
  return std::max(kMinimumCapacity, static_cast<size_t>(byte_size) * kCompileShapeMultiplier);
}

fx_converter::Node *AddAllocMemoryNode(ge::Graph *main, const std::string &tag,
                                 const ge::TensorDesc &empty_td, uint64_t s_allocator,
                                 fx_converter::Node *select_l2, uint64_t s_size, fx_converter::Node *calc_size,
                                 uint64_t &sym_gen, uint64_t *s_addr) {
  auto alloc_desc = std::make_shared<FxNodeSpec>("alloc_" + tag, "AllocMemory");
  alloc_desc->AddInputDesc(empty_td);
  alloc_desc->AddInputDesc(empty_td);
  alloc_desc->AddOutputDesc(empty_td);
  fx_converter::Node *alloc_out = AddNode(main, alloc_desc);
  const uint64_t addr_sym = sym_gen++;
  if (s_addr != nullptr) {
    *s_addr = addr_sym;
  }
  WireSymbols(alloc_out, {s_allocator, s_size}, {addr_sym});
  if (select_l2 != nullptr) {
    AddDataEdge(main, select_l2, 0, alloc_out, 0);
  }
  if (calc_size != nullptr) {
    AddDataEdge(main, calc_size, 0, alloc_out, 1);
  }
  return alloc_out;
}

static TensorEndpoint AddInt32Const(ge::Graph *main, const std::string &name, int32_t value, uint64_t &sym_gen,
                                    std::vector<fx_converter::Node *> *kernel_nodes) {
  fx_converter::Node *node = AddConstNode(main, name, &value, sizeof(value), false, sym_gen,
                                          FxrtConstType::kInt32T);
  if (node == nullptr || node->GetExtendInfo() == nullptr) {
    return {};
  }
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(node);
  }
  return {node, 0, node->GetExtendInfo()->GetOutputSymbol(0)};
}

TensorFieldEndpoints AddTensorFieldConsts(ge::Graph *main, const std::string &tag, ge::DataType dtype,
                                          gert::TensorPlacement placement, ge::Format storage_format,
                                          uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes) {
  TensorFieldEndpoints fields;
  fields.dtype = AddInt32Const(main, "const_dtype_" + tag, static_cast<int32_t>(dtype), sym_gen, kernel_nodes);
  fields.placement =
      AddInt32Const(main, "const_placement_" + tag, static_cast<int32_t>(placement), sym_gen, kernel_nodes);
  fields.storage_format =
      AddInt32Const(main, "const_storage_format_" + tag, static_cast<int32_t>(storage_format), sym_gen, kernel_nodes);
  fields.origin_format = AddInt32Const(main, "const_origin_format_" + tag,
                                       static_cast<int32_t>(OriginFormatOf(storage_format)), sym_gen, kernel_nodes);
  return fields;
}

static fx_converter::Node *AddBoxedTensorNode(ge::Graph *main, const std::string &name, const char *type,
                                       const ge::TensorDesc &td_o, const BuildTensorInputs &inputs,
                                       const TensorFieldEndpoints &fields, uint64_t &sym_gen, uint64_t *s_out_tensor) {
  if (!fields.Valid()) {
    return nullptr;
  }
  auto build_desc = std::make_shared<FxNodeSpec>(name, type);
  AddEmptyInputDescs(build_desc.get(), 6);
  build_desc->AddOutputDesc(td_o);
  fx_converter::Node *out_tensor_node = AddNode(main, build_desc);
  if (out_tensor_node == nullptr) {
    return nullptr;
  }
  const uint64_t out_tensor_sym = sym_gen++;
  if (s_out_tensor != nullptr) {
    *s_out_tensor = out_tensor_sym;
  }
  const TensorEndpoint ports[] = {inputs.shape, inputs.addr, fields.dtype, fields.placement, fields.storage_format,
                                  fields.origin_format};
  WireEndpointInputs(main, out_tensor_node, ports, std::size(ports), {out_tensor_sym});
  return out_tensor_node;
}

fx_converter::Node *AddBuildTensorNode(ge::Graph *main, const std::string &tag, const ge::TensorDesc &td_o,
                                 const BuildTensorInputs &inputs, ge::DataType dtype, ge::Format storage_format,
                                 uint64_t &sym_gen, uint64_t *s_out_tensor,
                                 std::vector<fx_converter::Node *> *kernel_nodes) {
  const TensorFieldEndpoints fields =
      AddTensorFieldConsts(main, tag, dtype, gert::kOnDeviceHbm, storage_format, sym_gen, kernel_nodes);
  return AddBoxedTensorNode(main, "build_tensor_" + tag, "FxBuildTensor", td_o, inputs, fields, sym_gen, s_out_tensor);
}

fx_converter::Node *AddBuildRefTensorNode(ge::Graph *main, const std::string &tag, const ge::TensorDesc &td_o,
                                    const TensorEndpoint &shape, const TensorEndpoint &addr, ge::DataType dtype,
                                    uint64_t &sym_gen, uint64_t *s_out_tensor,
                                    std::vector<fx_converter::Node *> *kernel_nodes) {
  if (main == nullptr || shape.node == nullptr || addr.node == nullptr) {
    return nullptr;
  }
  const TensorFieldEndpoints fields =
      AddTensorFieldConsts(main, tag, dtype, gert::kOnDeviceHbm, td_o.GetFormat(), sym_gen, kernel_nodes);
  auto *node = AddBoxedTensorNode(main, "build_ref_" + tag, "FxBuildRefTensor", td_o, {shape, addr}, fields, sym_gen,
                                  s_out_tensor);
  if (node != nullptr && kernel_nodes != nullptr) {
    kernel_nodes->push_back(node);
  }
  return node;
}

static fx_converter::Node *AddBuildTensorFromPair(ge::Graph *main, const std::string &tag, const TensorRef &ref,
                                     uint64_t &sym_gen, uint64_t *s_out_tensor,
                                     std::vector<fx_converter::Node *> *kernel_nodes) {
  if (main == nullptr || !ref.HasUnpackedPair()) {
    return nullptr;
  }
  const BuildTensorInputs inputs{ref.unpacked.shape, ref.unpacked.addr};
  fx_converter::Node *node = AddBuildTensorNode(main, tag, MakeNdTensorDesc(ref.shape, ref.dtype), inputs, ref.dtype,
                                                ge::FORMAT_ND, sym_gen, s_out_tensor, kernel_nodes);
  if (node != nullptr && kernel_nodes != nullptr) {
    kernel_nodes->push_back(node);
  }
  return node;
}

bool EnsureTensorRefBoxed(TensorRef *ref, ge::Graph *main, const std::string &tag,
                          uint64_t &sym_gen,
                          std::vector<fx_converter::Node *> *kernel_nodes) {
  if (ref == nullptr) {
    return false;
  }
  // OutputData is an execution-bound shell. Its shape is populated by the
  // final Ensure, so a live consumer must use the producer's InferShape/address
  // pair instead of reading the shell before that Ensure runs.
  if (ref->producer != nullptr &&
      (std::string(ref->producer->GetType()) != "OutputData" || !ref->HasUnpackedPair())) {
    return true;
  }
  if (!ref->HasUnpackedPair()) {
    return false;
  }
  if (ref->tiling_ep.Valid()) {
    ref->producer = ref->tiling_ep.node;
    ref->symbol = ref->tiling_ep.sym;
    return true;
  }
  uint64_t symbol = 0;
  // Always BuildTensor from (InferShape, addr, attr). BuildRefTensor on AllocMemory
  // keeps the allocation's original rank (TopK indices [B, 8] → Expand sees [1000]).
  fx_converter::Node *node = AddBuildTensorFromPair(main, tag, *ref, sym_gen, &symbol, kernel_nodes);
  if (node == nullptr) {
    return false;
  }
  ref->producer = node;
  ref->symbol = symbol;
  return true;
}

bool EnsureStridedTensorRefBoxed(TensorRef *ref, ge::Graph *main, BuiltExecuteGraph *built,
                                 const std::string &tag, uint64_t &sym_gen,
                                 std::vector<fx_converter::Node *> *kernel_nodes) {
  if (ref == nullptr || main == nullptr || built == nullptr) return false;
  // Geometry-changing views clear producer. A previously boxed value already
  // carries its live layout binding; rebuilding it would freeze later consumers
  // to the Convert-time stride and duplicate the same host work.
  if (ref->producer != nullptr && ref->producer->GetType() == "FxBuildStridedTensor") return true;
  // ModelV2 passes the caller's TensorV2 directly. ES dataset Data nodes carry
  // only a TensorV1, so reconstruct the view from the runtime layout sidecars.
  if (!IsEsGraphRunnerEnabled() && ref->producer != nullptr &&
      std::string(ref->producer->GetType()) == "Data") return true;
  if (!ref->HasUnpackedPair()) return false;
  const auto &logical = LogicalShape(*ref);
  const bool is_feed = ref->producer != nullptr && std::string(ref->producer->GetType()) == "Data" &&
                       ref->feed_index >= 0;
  if (!is_feed && IsDenseTensorRef(*ref)) {
    return EnsureTensorRefBoxed(ref, main, tag, sym_gen, kernel_nodes);
  }
  const bool derive_dense_stride = ref->proven_dense &&
      std::any_of(ref->stride.begin(), ref->stride.end(), [](int64_t value) { return value < 0; });
  const bool has_symbolic_stride = ref->stride_exprs.size() == ref->stride.size() &&
      std::any_of(ref->stride_exprs.begin(), ref->stride_exprs.end(), [](const auto &expr) { return expr != nullptr; });
  if ((!ref->stride_exprs.empty() && ref->stride_exprs.size() != ref->stride.size()) ||
      ref->stride.size() != logical.size() ||
      (ref->storage_offset < 0 && ref->storage_offset_expr == nullptr) ||
      (!derive_dense_stride && !has_symbolic_stride &&
       std::any_of(ref->stride.begin(), ref->stride.end(), [](int64_t value) { return value < 0; }))) {
    return false;
  }
  if (!ref->logical_shape_exprs.empty() && ref->logical_shape_exprs.size() != logical.size()) return false;
  std::vector<int64_t> layout_values;
  std::vector<fxrt::ir::SymbolicExprPtr> layout_exprs;
  layout_values.reserve(logical.size() + ref->stride.size() + 1U);
  layout_exprs.reserve(logical.size() + ref->stride.size() + 1U);
  for (size_t i = 0; i < logical.size(); ++i) {
    const auto expr = ref->logical_shape_exprs.empty() ? fxrt::ir::SymbolicExprPtr{} : ref->logical_shape_exprs[i];
    if (logical[i] < 0 && expr == nullptr) return false;
    layout_values.push_back(expr != nullptr ? 0 : logical[i]);
    layout_exprs.push_back(expr);
  }
  for (size_t i = 0; i < ref->stride.size(); ++i) {
    const auto expr = ref->stride_exprs.empty() ? fxrt::ir::SymbolicExprPtr{} : ref->stride_exprs[i];
    layout_values.push_back(expr != nullptr ? 0 : (derive_dense_stride ? kRuntimeDenseStrideSentinel : ref->stride[i]));
    layout_exprs.push_back(expr);
  }
  layout_values.push_back(ref->storage_offset_expr != nullptr ? 0 : ref->storage_offset);
  layout_exprs.push_back(ref->storage_offset_expr);
  const TensorFieldEndpoints fields =
      AddTensorFieldConsts(main, tag, ref->dtype, gert::kOnDeviceHbm, ge::FORMAT_ND, sym_gen, kernel_nodes);
  TensorEndpoint layout_ep;
  // A direct feed retains its runtime layout contract even when Convert sampled
  // a dense tensor. Input offsets have a separate host endpoint and are added
  // exactly once by FxBuildStridedTensor.
  if (is_feed) {
    std::fill(layout_exprs.begin(), layout_exprs.end(), fxrt::ir::SymbolicExprPtr{});
    layout_values.back() = 0;
  }
  if (!fields.Valid() || !AddRuntimeLayoutTensor(main, built, tag, layout_values, layout_exprs, sym_gen, kernel_nodes,
                                                 &layout_ep)) {
    return false;
  }
  if (is_feed) {
    built->feed_layout_refreshes.push_back(
        {static_cast<size_t>(ref->feed_index), built->runtime_layout_payloads.back()});
  }
  const int64_t zero_offset = 0;
  auto *runtime_offset = ref->runtime_storage_offset.Valid()
                             ? ref->runtime_storage_offset.node
                             : AddConstNode(main, "const_runtime_offset_" + tag, &zero_offset,
                                            sizeof(zero_offset), false, sym_gen, FxrtConstType::kInt64T);
  const int32_t runtime_offset_out = ref->runtime_storage_offset.Valid() ? ref->runtime_storage_offset.out : 0;
  const uint64_t runtime_offset_symbol = ref->runtime_storage_offset.Valid()
                                             ? ref->runtime_storage_offset.sym
                                             : (runtime_offset != nullptr && runtime_offset->GetExtendInfo() != nullptr
                                                    ? runtime_offset->GetExtendInfo()->GetOutputSymbol(0)
                                                    : 0U);
  if (runtime_offset == nullptr) return false;
  auto desc = std::make_shared<FxNodeSpec>("build_strided_tensor_" + tag, "FxBuildStridedTensor");
  for (size_t i = 0; i < 7U; ++i) desc->AddInputDesc(ge::TensorDesc{});
  desc->AddOutputDesc(MakeNdTensorDesc(logical, ref->dtype));
  auto *node = AddNode(main, desc);
  if (node == nullptr) return false;
  const uint64_t symbol = sym_gen++;
  const std::vector<TensorEndpoint> inputs{ref->unpacked.addr,
                                          fields.dtype,
                                          fields.placement,
                                          fields.storage_format,
                                          fields.origin_format,
                                          layout_ep,
                                          {runtime_offset, runtime_offset_out, runtime_offset_symbol}};
  std::vector<uint64_t> symbols;
  for (size_t i = 0; i < inputs.size(); ++i) {
    symbols.push_back(inputs[i].sym);
    if (AddDataEdge(main, inputs[i].node, inputs[i].out, node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  WireSymbols(node, symbols, {symbol});
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(node);
    if (!ref->runtime_storage_offset.Valid()) kernel_nodes->push_back(runtime_offset);
  }
  ref->producer = node;
  ref->symbol = symbol;
  return true;
}

fx_converter::Node *AddAllocModelOutTensorNode(ge::Graph *main, const std::string &tag, const ge::TensorDesc &td_o,
                                         const ge::TensorDesc &empty_td, uint64_t s_allocator,
                                         fx_converter::Node *select_l2, uint64_t s_size, fx_converter::Node *calc_size,
                                         uint64_t s_out_tensor, fx_converter::Node *out_tensor_node, uint64_t &sym_gen,
                                         uint64_t *s_out_addr) {
  auto alloc_out = std::make_shared<FxNodeSpec>("alloc_model_out_" + tag, "AllocModelOutTensor");
  alloc_out->AddInputDesc(empty_td);
  alloc_out->AddInputDesc(empty_td);
  alloc_out->AddInputDesc(td_o);
  alloc_out->AddOutputDesc(empty_td);
  fx_converter::Node *node = AddNode(main, alloc_out);
  const uint64_t addr_sym = sym_gen++;
  if (s_out_addr != nullptr) {
    *s_out_addr = addr_sym;
  }
  WireSymbols(node, {s_allocator, s_size, s_out_tensor}, {addr_sym});
  if (select_l2 != nullptr) {
    AddDataEdge(main, select_l2, 0, node, 0);
  }
  if (calc_size != nullptr) {
    AddDataEdge(main, calc_size, 0, node, 1);
  }
  if (out_tensor_node != nullptr) {
    AddDataEdge(main, out_tensor_node, 0, node, 2);
  }
  return node;
}

fx_converter::Node *AddEnsureTensorAtOutMemoryNode(ge::Graph *main, const std::string &tag,
                                             const EnsureOutMemoryInputs &inputs, fx_converter::Node *launch,
                                             std::vector<fx_converter::Node *> *kernel_nodes) {
  if (!inputs.fields.Valid()) {
    return nullptr;
  }
  const ge::TensorDesc empty_td;
  auto ensure_desc = std::make_shared<FxNodeSpec>("ensure_out_" + tag, "FxEnsureTensorAtOutMemory");
  const TensorEndpoint ports[] = {inputs.shape,           inputs.tensor_data,      inputs.fields.dtype,
                                  inputs.fields.placement, inputs.fields.storage_format, inputs.fields.origin_format,
                                  inputs.stream,          inputs.output_data,      inputs.allocator};
  for (size_t i = 0; i < std::size(ports); ++i) {
    ensure_desc->AddInputDesc(empty_td);
  }
  fx_converter::Node *ensure = AddNode(main, ensure_desc);
  std::vector<uint64_t> input_symbols;
  input_symbols.reserve(std::size(ports));
  for (const TensorEndpoint &port : ports) {
    input_symbols.push_back(port.sym);
  }
  WireSymbols(ensure, input_symbols, {});
  for (size_t i = 0; i < std::size(ports); ++i) {
    if (ports[i].node != nullptr) {
      AddDataEdge(main, ports[i].node, ports[i].out, ensure, static_cast<int32_t>(i));
    }
  }
  if (launch != nullptr) {
    AddDataEdge(main, launch, -1, ensure, -1);
  }
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(ensure);
  }
  return ensure;
}

fx_converter::Node *AddFreeBatchHbmNode(ge::Graph *main, const std::string &tag, const ge::TensorDesc &empty_td,
                                  uint64_t s_ws_addr, fx_converter::Node *alloc_ws, fx_converter::Node *launch,
                                  std::vector<fx_converter::Node *> *kernel_nodes) {
  auto free_ws_desc = std::make_shared<FxNodeSpec>("free_ws_" + tag, "FreeBatchHbm");
  free_ws_desc->AddInputDesc(empty_td);
  (void)FxAttrUtils::SetInt(free_ws_desc, "ReleaseResourceIndex", 0);
  fx_converter::Node *free_ws = AddNode(main, free_ws_desc);
  WireSymbols(free_ws, {s_ws_addr}, {});
  if (alloc_ws != nullptr) {
    AddDataEdge(main, alloc_ws, 0, free_ws, 0);
  }
  if (launch != nullptr) {
    AddDataEdge(main, launch, -1, free_ws, -1);
  }
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(free_ws);
  }
  return free_ws;
}

void FillLoweringCniCommon(const OpCatalogEntry &entry, const std::vector<TensorRef> &inputs,
                           const std::vector<int64_t> &out_shape, ge::DataType out_dtype, LoweringResult *out) {
  if (out == nullptr) {
    return;
  }
  out->cni.op_type = entry.opp_type;
  out->cni.cni_io = entry.cni_io;
  out->cni.catalog = &entry;
  out->cni.behavior = entry.behavior;
  out->cni.opp_input_names = (entry.opp_input_names != nullptr) ? entry.opp_input_names : "";
  out->cni.input_shapes.clear();
  out->cni.input_dtypes.clear();
  for (const auto &ref : inputs) {
    out->cni.input_shapes.push_back(PhysicalShape(ref));
    out->cni.input_dtypes.push_back(static_cast<int32_t>(ref.dtype));
    out->cni.input_formats.push_back(static_cast<int32_t>(ref.format));
  }
  out->cni.output_shape = out_shape;
  out->cni.output_dtype = static_cast<int32_t>(out_dtype);
}

fx_converter::Node *CreateSplitDataTensor(LoweringContext &ctx, const TensorRef &ref, const std::string &tag, size_t index,
                                    uint64_t *shape_symbol, uint64_t *addr_symbol) {
  if (ctx.main == nullptr || ctx.rt == nullptr || ctx.sym_gen == nullptr || ref.producer == nullptr) {
    return nullptr;
  }
  const ge::TensorDesc empty_td;
  auto desc = std::make_shared<FxNodeSpec>("split_" + tag + "_" + std::to_string(index), "SplitDataTensor");
  desc->AddInputDesc(empty_td);
  desc->AddInputDesc(empty_td);
  desc->AddOutputDesc(empty_td);
  desc->AddOutputDesc(empty_td);
  fx_converter::Node *split = AddNode(ctx.main, desc);
  const uint64_t s_shape = (*ctx.sym_gen)++;
  const uint64_t s_addr = (*ctx.sym_gen)++;
  WireSymbols(split, {ref.symbol, ctx.rt->s_allocator}, {s_shape, s_addr});
  AddDataEdge(ctx.main, ref.producer, 0, split, 0);
  AddDataEdge(ctx.main, ctx.rt->select_l2, 0, split, 1);
  if (ctx.kernel_nodes != nullptr) {
    ctx.kernel_nodes->push_back(split);
  }
  if (shape_symbol != nullptr) {
    *shape_symbol = s_shape;
  }
  if (addr_symbol != nullptr) {
    *addr_symbol = s_addr;
  }
  return split;
}

static void PublishSplitPair(LoweringContext &ctx, const TensorRef &ref, fx_converter::Node *split, uint64_t s_shape,
                             uint64_t s_addr) {
  if (ctx.tensors == nullptr || split == nullptr) {
    return;
  }
  for (auto &kv : *ctx.tensors) {
    TensorRef &t = kv.second;
    if (t.producer == ref.producer && t.symbol == ref.symbol && !t.HasUnpackedPair()) {
      t.unpacked.shape = TensorEndpoint{split, 0, s_shape};
      t.unpacked.addr = TensorEndpoint{split, 1, s_addr};
    }
  }
}

fx_converter::Node *GetOrCreateSplitDataTensor(LoweringContext &ctx, const TensorRef &ref, const std::string &tag,
                                         size_t index, uint64_t *shape_symbol, uint64_t *addr_symbol) {
  if (ref.producer == nullptr) {
    return nullptr;
  }
  const SplitCacheKey key{ref.producer, ref.symbol};
  const auto cached = ctx.split_cache.find(key);
  if (cached != ctx.split_cache.end()) {
    if (shape_symbol != nullptr) {
      *shape_symbol = cached->second.shape_symbol;
    }
    if (addr_symbol != nullptr) {
      *addr_symbol = cached->second.addr_symbol;
    }
    PublishSplitPair(ctx, ref, cached->second.node, cached->second.shape_symbol, cached->second.addr_symbol);
    return cached->second.node;
  }
  uint64_t s_shape = 0;
  uint64_t s_addr = 0;
  fx_converter::Node *split = CreateSplitDataTensor(ctx, ref, tag, index, &s_shape, &s_addr);
  if (split == nullptr) {
    return nullptr;
  }
  ctx.split_cache.emplace(key, SplitCacheEntry{split, s_shape, s_addr});
  PublishSplitPair(ctx, ref, split, s_shape, s_addr);
  if (shape_symbol != nullptr) {
    *shape_symbol = s_shape;
  }
  if (addr_symbol != nullptr) {
    *addr_symbol = s_addr;
  }
  return split;
}

void UnpackGraphInputTensors(LoweringContext &ctx) {
  if (ctx.tensors == nullptr) {
    return;
  }
  size_t index = 0;
  for (auto &kv : *ctx.tensors) {
    TensorRef &ref = kv.second;
    if (ref.HasUnpackedPair() || ref.producer == nullptr) {
      continue;
    }
    uint64_t s_shape = 0;
    uint64_t s_addr = 0;
    fx_converter::Node *split = GetOrCreateSplitDataTensor(ctx, ref, "feed", index++, &s_shape, &s_addr);
    if (split == nullptr) {
      continue;
    }
    ref.unpacked.shape = TensorEndpoint{split, 0, s_shape};
    ref.unpacked.addr = TensorEndpoint{split, 1, s_addr};
    ref.allocation = ref.unpacked.addr;
  }
}

}  // namespace fx_converter
