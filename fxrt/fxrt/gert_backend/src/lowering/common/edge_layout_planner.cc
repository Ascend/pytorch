#include "lowering/common/edge_layout_planner.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include "common/logger.h"
#include "exe_graph/execute_graph_builder.h"
#include "ir/tensor/tensor.h"
#include "lowering/common/dense_layout_lowering.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/storage_geometry.h"
#include "lowering/common/view_metadata.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {
fxrt::ir::NodePtr IntList(const std::vector<int64_t> &values) {
  std::vector<fxrt::ir::ValuePtr> elements;
  for (int64_t value : values) {
    elements.push_back(fxrt::ir::MakeIntrusive<fxrt::ir::Value>(value));
  }
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  node->op = fxrt::ops::Op_End;
  node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
      fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elements)));
  return node;
}

fxrt::ir::NodePtr PackedViewGeometry(const fxrt::ir::NodePtr &input, const TensorRef &source,
                                    const std::vector<int64_t> &packed) {
  auto geometry = IntList(packed);
  const auto &tensor = input->output->ToTensor();
  const auto &shape = LogicalShape(source);
  std::vector<fxrt::ir::ValuePtr> elements;
  elements.reserve(packed.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    fxrt::ir::SymbolicExprPtr expr;
    if (i > 0U && i <= shape.size()) {
      const size_t axis = i - 1U;
      if (source.logical_shape_exprs.size() == shape.size()) expr = source.logical_shape_exprs[axis];
      else if (tensor->HasSymbolicShape() && ShapeOfNode(input) == shape &&
               tensor->GetSymbolicShape().size() == shape.size() && shape[axis] < 0) {
        expr = tensor->GetSymbolicShape()[axis];
      }
    } else if (i > shape.size() && i <= 2U * shape.size()) {
      if (source.stride_exprs.size() == shape.size()) expr = source.stride_exprs[i - shape.size() - 1U];
    } else if (i == packed.size() - 1U) {
      expr = source.storage_offset_expr;
    }
    elements.push_back(expr != nullptr ? fxrt::ir::MakeIntrusive<fxrt::ir::Value>(expr)
                                      : fxrt::ir::MakeIntrusive<fxrt::ir::Value>(packed[i]));
  }
  geometry->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
      fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elements)));
  return geometry;
}

std::string LayoutKey(const TensorRef &ref) {
  std::ostringstream key;
  key << ref.unpacked.addr.node << ':' << ref.unpacked.addr.out << ':' << ref.dtype
      << ':' << ref.storage_offset << ":runtime_offset@" << ref.runtime_storage_offset.node << ':'
      << ref.runtime_storage_offset.out;
  // Equal sentinels do not imply equal layouts. Distinct symbolic origins or
  // strides must never reuse one adapted address/dense snapshot.
  if (ref.storage_offset_expr != nullptr) key << ":offset=" << ref.storage_offset_expr->ToString();
  for (const auto &expr : ref.logical_shape_exprs) key << ":size=" << (expr == nullptr ? "" : expr->ToString());
  for (const auto &expr : ref.stride_exprs) key << ":stride=" << (expr == nullptr ? "" : expr->ToString());
  // Reshaping a contiguous chunk does not change the logical iteration order.
  // Use that order as the key so [K,M/2,2] and [K,M] share one dense write.
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
  const auto &shape = LogicalShape(ref);
  std::vector<int64_t> source_stride = ref.stride;
  if (source_stride.empty()) {
    ContiguousRowMajorStrides(shape, &source_stride);
  }
  // Distinct symbolic shape expressions are not equal just because both use
  // the numeric sentinel -1. Share those only at the same shape endpoint.
  if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
    const auto &endpoint = ref.logical_shape_ep.Valid() ? ref.logical_shape_ep : ref.unpacked.shape;
    key << ":shape@" << endpoint.node << ':' << endpoint.out;
  }
  if (source_stride.size() != shape.size()) {
    return key.str() + ":invalid-stride";
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) continue;
    const int64_t stride = source_stride[i];
    if (!dims.empty() && shape[i] > 0 && stride >= 0 && strides.back() / shape[i] == stride &&
        strides.back() % shape[i] == 0) {
      dims.back() *= shape[i];
      strides.back() = stride;
    } else {
      dims.push_back(shape[i]);
      strides.push_back(stride);
    }
  }
  for (size_t i = 0; i < dims.size(); ++i) key << ',' << dims[i] << '@' << strides[i];
  return key.str();
}

bool ReshapeCachedDense(const TensorRef &cached, const TensorRef &source, const std::string &tag,
                        LoweringContext &ctx, TensorRef *result, std::string *error) {
  *result = cached;
  if (LogicalShape(cached) == LogicalShape(source)) return true;
  fx_converter::Node *shape = nullptr;
  uint64_t symbol = 0;
  if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_dense_shape", LogicalShape(source),
                                   *ctx.sym_gen, ctx.kernel_nodes, &shape, &symbol, error)) return false;
  result->producer = nullptr;
  result->symbol = 0;
  result->shape = LogicalShape(source);
  result->logical_shape.clear();
  result->storage_shape.clear();
  result->logical_shape_ep = {};
  result->unpacked.shape = TensorEndpoint{shape, 0, symbol};
  ContiguousRowMajorStrides(result->shape, &result->stride);
  return true;
}

// A non-overlapping permutation, possibly after splitting/merging contiguous
// dimensions, can be written by one Transpose on a physical reshape of the base.
bool Permutation(const TensorRef &ref, std::vector<int64_t> *physical, std::vector<int64_t> *perm) {
  const auto &shape = LogicalShape(ref);
  if (ref.storage_offset != 0 || ref.stride.size() != shape.size() ||
      std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d < 0; })) return false;
  std::vector<size_t> order(shape.size());
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ref.stride[a] > ref.stride[b]; });
  std::vector<int64_t> sorted_strides;
  perm->resize(shape.size());
  for (size_t i = 0; i < order.size(); ++i) {
    physical->push_back(shape[order[i]]);
    sorted_strides.push_back(ref.stride[order[i]]);
    (*perm)[order[i]] = static_cast<int64_t>(i);
  }
  const int64_t storage_numel = NumelOfShape(PhysicalShape(ref));
  return IsContiguousRowMajor(*physical, sorted_strides) &&
         (storage_numel < 0 || NumelOfShape(*physical) == storage_numel);
}

// Broadcast may read a compact contiguous prefix of a larger allocation (for
// example x[:1].expand(...)). Re-boxing that prefix needs no device slice.
bool DenseBroadcastSource(const TensorRef &ref, std::vector<int64_t> *physical) {
  const auto &shape = LogicalShape(ref);
  if (ref.storage_offset_expr != nullptr || ref.storage_offset < 0 || ref.stride.size() != shape.size() ||
      !HasBroadcastStride(shape, ref.stride)) return false;
  *physical = shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || ref.stride[i] < 0) return false;
    if (ref.stride[i] == 0) (*physical)[i] = 1;
  }
  const int64_t capacity = NumelOfShape(PhysicalShape(ref));
  const int64_t elements = NumelOfShape(*physical);
  return IsContiguousRowMajor(*physical, ref.stride) && capacity >= 0 &&
         ref.storage_offset <= capacity && elements <= capacity - ref.storage_offset;
}

std::vector<int64_t> TargetShape(const fxrt::ir::NodePtr &node, const TensorRef &input) {
  auto shape = ShapeOfNode(node);
  if (node->op == fxrt::ops::Op_view || node->op == fxrt::ops::Op_reshape) {
    for (size_t i = 1; i < node->inputs.size(); ++i) {
      const auto &arg = node->inputs[i];
      if (arg != nullptr && arg->output != nullptr && arg->output->IsTuple()) {
        shape = ReadIntListSymbolSafe(arg->output->ToTuple());
        break;
      }
    }
  }
  const auto inferred = std::find(shape.begin(), shape.end(), -1);
  if (inferred != shape.end() && std::count(shape.begin(), shape.end(), -1) == 1) {
    const int64_t total = NumelOfShape(LogicalShape(input));
    int64_t known = 1;
    for (int64_t dim : shape) if (dim != -1) known *= dim;
    if (total >= 0 && known > 0 && total % known == 0) *inferred = total / known;
  }
  return shape;
}

// A clone's allocation can disappear only while its storage identity is not
// observable. Model outputs are copied into independent caller-owned storage;
// effects inside the graph need their own allocation before the effect executes.
bool CanElideClone(const LoweringContext &ctx) {
  if (ctx.compute_nodes == nullptr) return false;
  for (const auto &candidate : *ctx.compute_nodes) {
    const std::string name = fxrt::ops::ToStr(candidate->op);
    const auto *lowering = OpLoweringRegistry::Instance().Find(candidate, &ctx);
    if (name.rfind("inplace_", 0) == 0 || lowering == nullptr || lowering->MayWriteInputs()) return false;
  }
  return true;
}

bool CloneRequestsContiguous(const fxrt::ir::NodePtr &node) {
  return node->op == fxrt::ops::Op_clone && node->inputs.size() > 1U &&
         node->inputs[1]->output != nullptr && node->inputs[1]->output->IsInt() &&
         node->inputs[1]->output->ToInt() != 0;
}

// Tuple/argument wrappers are uses too, but do not walk through a computing
// consumer: its own input edge is the use we need to count.
bool ReferencesValue(const fxrt::ir::NodePtr &argument, const fxrt::ir::NodePtr &value) {
  if (argument == value) return true;
  if (argument == nullptr || (argument->op != fxrt::ops::Op_make_tuple && argument->op != fxrt::ops::Op_End)) {
    return false;
  }
  return std::any_of(argument->inputs.begin(), argument->inputs.end(),
                     [&](const fxrt::ir::NodePtr &input) { return ReferencesValue(input, value); });
}

bool IsReturnedValue(const fxrt::ir::NodePtr &value, const LoweringContext &ctx) {
  return ctx.return_values == nullptr ||
         std::any_of(ctx.return_values->begin(), ctx.return_values->end(),
                     [&](const fxrt::ir::NodePtr &output) { return ReferencesValue(output, value); });
}

bool HasExclusiveUser(const fxrt::ir::NodePtr &value, const fxrt::ir::NodePtr &user,
                      const LoweringContext &ctx) {
  if (ctx.compute_nodes == nullptr || ctx.return_values == nullptr) return false;
  for (const auto &output : *ctx.return_values) {
    if (ReferencesValue(output, value)) return false;
  }
  bool found_user = false;
  for (const auto &candidate : *ctx.compute_nodes) {
    for (const auto &input : candidate->inputs) {
      if (!ReferencesValue(input, value)) continue;
      if (candidate.get() != user.get()) return false;
      found_user = true;
    }
  }
  return found_user;
}

// A contiguous clone can reuse an exclusive temporary even when an unrelated
// compiled kernel mutates another allocation. Keep the proof local: every view
// in the chain has one user, none is returned, and the root is a fresh matrix
// result. Inputs, unknown allocation contracts and shared roots remain isolated.
bool CanReuseExclusiveTemporary(const fxrt::ir::NodePtr &clone, const LoweringContext &ctx) {
  if (!CloneRequestsContiguous(clone) || clone->inputs.empty()) return false;
  auto user = clone;
  auto value = clone->inputs[0];
  while (value != nullptr && value->output != nullptr && value->output->IsTensor()) {
    if (!HasExclusiveUser(value, user, ctx)) return false;
    if (value->op == fxrt::ops::Op_mm || value->op == fxrt::ops::Op_batch_matmul) return true;
    if (!IsStorageSharingViewOp(value->op) || value->inputs.empty()) return false;
    user = value;
    value = value->inputs[0];
  }
  return false;
}

bool WriteDense(const fxrt::ir::NodePtr &input, const TensorRef &source, const std::string &tag,
                LoweringContext &ctx, LoweringResult *result, std::string *error) {
  // Path C consumes storage-indexed data.  For an external non-contiguous view,
  // its base address and storage offset are separate runtime values; feeding the
  // raw storage base with a literal offset of zero would read the wrong rows.
  // Rebase the host-side data header first.  FxContiguousView only creates a
  // bounded GertTensorData header, so this is address adaptation, not a device
  // copy.  The following AsStrided lowering can then use offset zero while
  // retaining the original logical stride geometry.
  TensorRef layout_source = source;
  std::vector<int64_t> broadcast_source;
  const bool broadcast = DenseBroadcastSource(source, &broadcast_source);
  // Concrete slice origins can travel as Slice/AsStrided parameters. Only
  // runtime origins and a broadcast's raw-address ABI require a host adapter.
  if (source.runtime_storage_offset.Valid() || (broadcast && source.storage_offset != 0)) {
    TensorEndpoint source_addr = source.unpacked.addr;
    TensorEndpoint source_shape = source.unpacked.shape;
    if (!source_addr.Valid() || !source_shape.Valid()) {
      if (source.producer == nullptr) {
        if (error != nullptr) *error = "layout planner: missing source address for runtime offset";
        return false;
      }
      uint64_t shape_symbol = 0;
      uint64_t addr_symbol = 0;
      auto *split = GetOrCreateSplitDataTensor(ctx, source, tag + "_source", 0, &shape_symbol, &addr_symbol);
      if (split == nullptr) {
        if (error != nullptr) *error = "layout planner: failed to unpack source for runtime offset";
        return false;
      }
      source_shape = TensorEndpoint{split, 0, shape_symbol};
      source_addr = TensorEndpoint{split, 1, addr_symbol};
    }
    if (!source_addr.Valid() || !source_shape.Valid()) {
      if (error != nullptr) *error = "layout planner: incomplete source pair for runtime offset";
      return false;
    }
    const int32_t item_size = ge::GetSizeByDataType(source.dtype);
    if (item_size <= 0 || source.storage_offset < 0 ||
        static_cast<uint64_t>(source.storage_offset) >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(item_size)) {
      if (error != nullptr) *error = "layout planner: invalid source offset or item size";
      return false;
    }
    const size_t byte_offset = static_cast<size_t>(source.storage_offset) * static_cast<size_t>(item_size);
    const size_t element_size = static_cast<size_t>(item_size);
    const int64_t zero_offset = 0;
    const ge::TensorDesc empty_td;
    RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
    const auto c_byte_offset = builder.Const("const_layout_byte_offset_" + tag, &byte_offset,
                                             sizeof(byte_offset), false);
    const auto c_item_size = builder.Const("const_layout_item_size_" + tag, &element_size,
                                           sizeof(element_size), false);
    const auto c_zero_offset = source.runtime_storage_offset.Valid()
                                   ? source.runtime_storage_offset
                                   : builder.Const("const_layout_runtime_offset_" + tag, &zero_offset,
                                                   sizeof(zero_offset), false);
    const auto view = builder.Node("FxContiguousView", "layout_source_" + tag,
                                   {source_addr, source_shape, c_zero_offset, c_byte_offset, c_item_size}, 2U);
    if (!view.outs[0].Valid() || !view.outs[1].Valid()) {
      if (error != nullptr) *error = "layout planner: failed to create source address view";
      return false;
    }
    layout_source.unpacked.addr = view.outs[0];
    // AsStrided indexes physical storage, which can be larger than the view's
    // logical shape. The address adapter's dense logical shape is only for
    // dense consumers; reusing it here rejects valid layouts with holes.
    layout_source.unpacked.shape = source_shape;
    layout_source.storage_offset = 0;
    layout_source.runtime_storage_offset = {};
    layout_source.allocation = source.allocation.Valid() ? source.allocation : source_addr;
  }
  bool specialized = false;
  if (!TryLowerSpecializedDenseLayout(input, layout_source, tag, ctx, result, &specialized, error)) return false;
  if (specialized) return true;
  auto adapter = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  adapter->attrs["fxrt_write_dense"] = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(true);
  const auto &adapter_shape = LogicalShape(source);
  auto output = fxrt::ir::MakeIntrusive<fxrt::ir::Tensor>(
      adapter_shape, input->output->ToTensor()->Dtype(), input->output->ToTensor()->GetDevice());
  const auto &input_tensor = input->output->ToTensor();
  if (input_tensor->HasSymbolicShape() && input_tensor->Shape() == adapter_shape) {
    output->SetSymbolicShape(input_tensor->GetSymbolicShape());
  }
  adapter->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(output);
  TensorRef base = layout_source;
  base.producer = nullptr;  // Re-box from the selected physical shape, never stale logical metadata.
  base.symbol = 0;
  // A prior consumer may have boxed the logical subview. The storage-indexed
  // adapter must use the physical shape selected below for tiling as well.
  base.tiling_ep = {};
  base.logical_shape_ep = {};
  std::vector<int64_t> physical;
  std::vector<int64_t> perm;
  if (Permutation(layout_source, &physical, &perm)) {
    adapter->op = fxrt::ops::Op_permute_view;
    adapter->inputs = {input, IntList(perm)};
    base.shape = physical;
    base.logical_shape.clear();
    base.storage_shape = physical;
    ContiguousRowMajorStrides(physical, &base.stride);
    fx_converter::Node *shape_node = nullptr;
    uint64_t shape_symbol = 0;
    if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_physical", physical, *ctx.sym_gen,
                                     ctx.kernel_nodes, &shape_node, &shape_symbol, error)) return false;
    base.unpacked.shape = TensorEndpoint{shape_node, 0, shape_symbol};
  } else if (broadcast) {
    adapter->op = fxrt::ops::Op_expand;
    adapter->inputs = {input, IntList(LogicalShape(source))};
    base.shape = broadcast_source;
    base.logical_shape.clear();
    base.logical_shape_exprs.clear();
    base.storage_shape = broadcast_source;
    base.stride_exprs.clear();
    base.storage_offset_expr = nullptr;
    base.proven_dense = true;
    ContiguousRowMajorStrides(broadcast_source, &base.stride);
    fx_converter::Node *shape_node = nullptr;
    uint64_t shape_symbol = 0;
    if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_broadcast_source", broadcast_source,
                                     *ctx.sym_gen, ctx.kernel_nodes, &shape_node, &shape_symbol, error)) return false;
    base.unpacked.shape = TensorEndpoint{shape_node, 0, shape_symbol};
  } else {
    // General offsets/slices/broadcasts are explicitly storage-addressed.
    // Force Path C even when the layout normally qualifies for ViewAlias.
    adapter->op = fxrt::ops::Op_as_strided_view;
    std::vector<int64_t> packed{static_cast<int64_t>(LogicalShape(source).size())};
    packed.insert(packed.end(), LogicalShape(source).begin(), LogicalShape(source).end());
    std::vector<int64_t> source_stride = layout_source.stride;
    if (source_stride.empty() && !ContiguousRowMajorStrides(LogicalShape(layout_source), &source_stride)) return false;
    packed.insert(packed.end(), source_stride.begin(), source_stride.end());
    packed.push_back(layout_source.storage_offset);
    adapter->inputs = {input, PackedViewGeometry(input, layout_source, packed)};
    // Pack the selected TensorRef, including its expressions, after address
    // adaptation. Reusing the producer's original packed arguments here would
    // apply its offset twice after the address was already rebased.
    base.logical_shape.clear();
    base.logical_shape_exprs.clear();
    base.shape = PhysicalShape(layout_source);
    base.storage_shape.clear();
    base.storage_offset = 0;
    base.storage_offset_expr = nullptr;
    base.stride_exprs.clear();
    ContiguousRowMajorStrides(base.shape, &base.stride);
    fx_converter::Node *shape_node = nullptr;
    uint64_t shape_symbol = 0;
    if (base.feed_index < 0 &&
        std::none_of(base.shape.begin(), base.shape.end(), [](int64_t dim) { return dim < 0; })) {
      if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_base", base.shape, *ctx.sym_gen,
                                       ctx.kernel_nodes, &shape_node, &shape_symbol, error)) {
        return false;
      }
      base.unpacked.shape = TensorEndpoint{shape_node, 0, shape_symbol};
    } else if (base.unpacked.addr.Valid()) {
      const int32_t dtype_size = ge::GetSizeByDataType(base.dtype);
      if (dtype_size <= 0) {
        if (error != nullptr) *error = "layout planner: invalid element size for storage shape";
        return false;
      }
      const size_t item_size = static_cast<size_t>(dtype_size);
      const ge::TensorDesc empty_td;
      RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
      const auto size = builder.Const("const_storage_item_size_" + tag, &item_size, sizeof(item_size), false);
      const auto capacity = builder.Node("FxBuildStorageCapacityShape", "storage_capacity_" + tag,
                                         {base.unpacked.addr, size}, 1U);
      if (capacity.outs.empty() || !capacity.outs[0].Valid()) {
        if (error != nullptr) *error = "layout planner: failed to build runtime storage shape";
        return false;
      }
      // Feed storage can exceed its logical shape and can change between
      // executions. Use this address header's live capacity, including after
      // rebasing, instead of freezing a sample view's extent into tiling.
      // This endpoint describes a flat allocation capacity, not the original
      // logical rank. Keep metadata aligned so AsStrided cannot select a
      // multidimensional Slice whose input shape is actually rank one.
      base.shape = {-1};
      base.stride = {1};
      base.unpacked.shape = capacity.outs[0];
    }
  }
  const auto &name = ctx.name_of->at(input.get());
  const TensorRef saved = ctx.tensors->at(name);
  (*ctx.tensors)[name] = base;
  const auto final_flags = ctx.output_final_mask;
  ctx.output_final_mask = {false};
  auto *lowering = OpLoweringRegistry::Instance().Find(adapter, &ctx);
  const bool ok = lowering != nullptr && lowering->Lower(adapter, tag, ctx, result, error);
  ctx.output_final_mask = final_flags;
  (*ctx.tensors)[name] = saved;
  return ok;
}

}  // namespace

bool RebaseContiguousView(const TensorRef &source, const std::string &tag, LoweringContext &ctx,
                          TensorRef *result, std::string *error) {
  const int32_t item_size = ge::GetSizeByDataType(source.dtype);
  const bool contiguous = source.proven_dense || source.stride.empty() ||
                          IsContiguousRowMajor(LogicalShape(source), source.stride);
  if (!contiguous || item_size <= 0 ||
      (source.storage_offset_expr == nullptr && (source.storage_offset < 0 ||
       static_cast<uint64_t>(source.storage_offset) > std::numeric_limits<size_t>::max() / item_size))) {
    if (error != nullptr) *error = "contiguous view: invalid byte offset or element size";
    return false;
  }
  // Nothing to fold: no static offset, no symbolic offset, and no per-Execute feed
  // offset. FxContiguousView would then copy the data header (addr + same span) and
  // the shape header verbatim, i.e. it is a pure identity that still costs one host
  // kernel per Execute. Skip the node and hand back the same endpoints; the
  // bookkeeping below is what callers actually consume.
  if (source.storage_offset == 0 && source.storage_offset_expr == nullptr &&
      !source.runtime_storage_offset.Valid() && source.unpacked.Complete()) {
    *result = source;
    result->producer = nullptr;
    result->symbol = 0;
    result->tiling_ep = {};
    result->shape = LogicalShape(source);
    result->logical_shape.clear();
    result->storage_shape = result->shape;
    ConsumeStorageOffset(source, result);
    result->proven_dense = true;
    result->can_adopt_for_model_output = false;
    if (source.logical_shape_ep.Valid()) result->unpacked.shape = source.logical_shape_ep;
    return result->unpacked.Complete();
  }
  *result = source;
  result->producer = nullptr;
  result->symbol = 0;
  result->tiling_ep = {};
  result->shape = LogicalShape(source);
  result->logical_shape.clear();
  result->storage_shape = result->shape;
  result->logical_shape_ep = {};
  ConsumeStorageOffset(source, result);
  result->proven_dense = true;
  result->can_adopt_for_model_output = false;
  if (source.logical_shape_ep.Valid()) result->unpacked.shape = source.logical_shape_ep;
  if (!result->unpacked.Complete()) {
    if (error != nullptr) *error = "contiguous view: missing shape/address endpoints";
    return false;
  }
  const size_t byte_offset = source.storage_offset_expr == nullptr
                                ? static_cast<size_t>(source.storage_offset) * static_cast<size_t>(item_size)
                                : 0U;
  const int64_t zero_offset = 0;
  const size_t element_size = static_cast<size_t>(item_size);
  const ge::TensorDesc empty_td;
  RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
  const auto offset = builder.Const("const_view_offset_" + tag, &byte_offset, sizeof(byte_offset), false);
  const auto runtime_offset = source.runtime_storage_offset.Valid()
                                  ? source.runtime_storage_offset
                                  : builder.Const("const_runtime_view_offset_" + tag, &zero_offset,
                                                  sizeof(zero_offset), false);
  const auto item = builder.Const("const_view_itemsize_" + tag, &element_size, sizeof(element_size), false);
  std::vector<TensorEndpoint> inputs{source.unpacked.addr, result->unpacked.shape, runtime_offset, offset, item};
  if (source.storage_offset_expr != nullptr) {
    TensorEndpoint dynamic_offset;
    if (!AddRuntimeLayoutTensor(ctx.main, ctx.built, tag + "_offset", {0}, {source.storage_offset_expr},
                                *ctx.sym_gen, ctx.kernel_nodes, &dynamic_offset)) {
      if (error != nullptr) *error = "contiguous view: cannot bind symbolic offset";
      return false;
    }
    inputs.push_back(dynamic_offset);
  }
  const auto view = builder.Node("FxContiguousView", "contiguous_view_" + tag, inputs, 2U);
  result->unpacked.addr = view.outs[0];
  result->unpacked.shape = view.outs[1];
  return result->unpacked.Complete();
}

bool PlanInputLayouts(const fxrt::ir::NodePtr &node, const OpLowering &lowering,
                      const std::string &tag, LoweringContext &ctx, EdgeLayoutPlan *plan, std::string *error) {
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const auto &input = node->inputs[i];
    if (input == nullptr || input->output == nullptr || !input->output->IsTensor()) continue;
    TensorRef ref;
    if (!ResolveProducerTensor(input, ctx, &ref, error)) return false;
    const auto kind = lowering.InputLayout(node, i);
    const bool independent = node->op == fxrt::ops::Op_clone &&
                             (IsReturnedValue(node, ctx) ||
                              (!CanElideClone(ctx) && !CanReuseExclusiveTemporary(node, ctx)) ||
                              (ref.storage_offset != 0 && !CloneRequestsContiguous(node)));
    const bool clone_dense = CloneRequestsContiguous(node);
    bool accept = IsDenseTensorRef(ref) || kind == InputLayoutKind::kStorageIndexed ||
                  kind == InputLayoutKind::kStridedTensor;
    if (kind == InputLayoutKind::kMatrix) {
      // The matrix prepare path reads the original TensorV2 feed's live stride.
      // A symbolic stride sentinel in TensorRef must not force an AsStrided copy
      // when no graph view has changed that boxed input. Internal views still
      // require a proven last-two transpose; Data identity alone cannot prove it.
      const bool runtime_feed = ref.producer != nullptr && ref.producer->GetType() == "Data" &&
                                ref.storage_offset == 0;
      accept = accept || runtime_feed || IsLastTwoDimsTransposed(ref);
    }
    if (kind == InputLayoutKind::kMetadata) {
      // as_strided uses its own explicit geometry. Shape-only aliases must prove
      // that reinterpreting the existing storage is legal, rather than assume dense.
      std::vector<int64_t> strides;
      NativeViewLayout native;
      if (ReadNativeViewLayout(node, ref, ctx.symbol_env, &native)) {
        accept = native.aliases;
      } else if (node->attrs.count("fxrt_native_view_layouts") != 0U) {
        if (error != nullptr) *error = "native view: inferred layout does not match the selected producer";
        return false;
      } else {
        accept = accept || node->op == fxrt::ops::Op_as_strided_view ||
                 ComputeReshapeViewStrides(LogicalShape(ref), ref.stride, TargetShape(node, ref), &strides);
      }
    }
    if (node->op == fxrt::ops::Op_view && !accept) {
      if (error != nullptr) *error = "strict view: target shape cannot alias the input layout; use reshape";
      return false;
    }
    if (clone_dense) accept = IsDenseTensorRef(ref);
    if (independent) accept = false;
    if (accept) continue;
    // Storage isolation is a semantic copy, never a reusable edge-layout snapshot.
    const std::string key = independent ? "clone:" + tag : LayoutKey(ref);
    auto cached = ctx.dense_layout_cache.find(key);
    if (cached == ctx.dense_layout_cache.end()) {
      LoweringResult adapter;
      const auto adapter_tag = "layout_" + tag + "_i" + std::to_string(i);
      const bool contiguous = ref.proven_dense || ref.stride.empty() ||
                              IsContiguousRowMajor(LogicalShape(ref), ref.stride);
      if (contiguous && !independent) {
        TensorRef rebased;
        if (!RebaseContiguousView(ref, adapter_tag, ctx, &rebased, error)) return false;
        adapter.outputs.push_back(std::move(rebased));
        adapter.skip_cni = true;
      } else {
        TensorRef copy_source = ref;
        const bool preserve_layout = independent && !clone_dense;
        std::vector<int64_t> clone_physical;
        if (preserve_layout) {
          // Preserve-format clone of a non-overlapping dense permutation copies
          // its physical span, starting at the source's effective offset. Holes
          // and broadcast clones already request a dense output via their IR.
          std::vector<size_t> order(LogicalShape(ref).size());
          std::iota(order.begin(), order.end(), 0U);
          std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return ref.stride[a] > ref.stride[b];
          });
          std::vector<int64_t> sorted_stride;
          for (size_t axis : order) {
            clone_physical.push_back(LogicalShape(ref)[axis]);
            sorted_stride.push_back(ref.stride[axis]);
          }
          if (!IsContiguousRowMajor(clone_physical, sorted_stride)) {
            if (error != nullptr) *error = "clone: preserve-format layout is not a dense permutation";
            return false;
          }
          copy_source.logical_shape = clone_physical;
          copy_source.logical_shape_ep = {};
          copy_source.stride = sorted_stride;
        }
        if (!WriteDense(input, copy_source, adapter_tag, ctx, &adapter, error) || adapter.outputs.size() != 1U)
          return false;
        if (preserve_layout) {
          auto &copy = adapter.outputs[0];
          copy.shape = clone_physical;
          copy.logical_shape = LogicalShape(ref);
          copy.logical_shape_ep = ref.logical_shape_ep;
          copy.storage_shape = clone_physical;
          copy.storage_offset = 0;
          copy.stride = ref.stride;
          copy.proven_dense = ref.proven_dense;
        }
      }
      cached = ctx.dense_layout_cache.emplace(key, DenseLayoutValue{adapter.outputs[0], adapter.launch}).first;
      plan->adapters.push_back(std::move(adapter));
    }
    const std::string name = ctx.name_of->at(input.get());
    TensorRef adapted;
    if (independent) {
      adapted = cached->second.tensor;
    } else if (!ReshapeCachedDense(cached->second.tensor, ref, tag + "_i" + std::to_string(i), ctx, &adapted, error)) {
      return false;
    }
    if (independent) adapted.can_adopt_for_model_output = false;
    plan->originals.emplace_back(name, ref);
    plan->replacements.emplace_back(name, adapted);
    (*ctx.tensors)[name] = adapted;
    RT_GLOG(INFO) << "LayoutPlanner " << tag << " input=" << i << " lowering=" << lowering.Name()
                  << (independent ? " uses isolated storage" : " uses shared dense value");
  }
  return true;
}

void RestoreInputLayouts(LoweringContext &ctx, const EdgeLayoutPlan &plan) {
  for (auto it = plan.originals.rbegin(); it != plan.originals.rend(); ++it) (*ctx.tensors)[it->first] = it->second;
}
void ApplyInputLayouts(LoweringContext &ctx, const EdgeLayoutPlan &plan) {
  for (const auto &entry : plan.replacements) (*ctx.tensors)[entry.first] = entry.second;
}
}  // namespace fx_converter
