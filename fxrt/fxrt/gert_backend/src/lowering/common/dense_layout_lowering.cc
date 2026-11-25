#include "lowering/common/dense_layout_lowering.h"

#include <algorithm>
#include <limits>

#include "ir/tensor/tensor.h"
#include "lowering/aicore/aicore_binary_kernel_selector.h"
#include "lowering/aicore/aicore_binary_op_lowering.h"
#include "lowering/common/view_utils.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog.h"

namespace fx_converter {
namespace {

// Static schemas have Convert lifetime: the CNI records retain these pointers.
const OpCatalogEntry &UnitSliceEntry() {
  static const auto behavior = BehaviorBuilder().PreferHighPerformance(false).Build();
  static const auto row = CatalogRowBuilder("dense_slice").Opp("Slice").AiCore()
      .Arity(3).Inputs("x,offsets,size").ShapeInput(1).ShapeListVector()
      .ShapeListInt32().Behavior(&behavior);
  static const auto entry = row.Build();
  return entry;
}

fxrt::ir::NodePtr Integer(int64_t value) {
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  node->op = fxrt::ops::Op_End;
  node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(value);
  return node;
}

fxrt::ir::NodePtr Integers(const std::vector<int64_t> &values) {
  auto node = Integer(0);
  std::vector<fxrt::ir::ValuePtr> elements;
  for (const auto value : values) {
    elements.push_back(fxrt::ir::MakeIntrusive<fxrt::ir::Value>(value));
  }
  node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
      fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elements)));
  return node;
}

struct DenseCandidate {
  const OpCatalogEntry *entry = nullptr;
  std::vector<fxrt::ir::NodePtr> arguments;
  std::vector<std::vector<int64_t>> shapes;
};

bool ConcreteGeometry(const TensorRef &ref, std::vector<int64_t> *base_stride) {
  const auto &physical = PhysicalShape(ref);
  if (ref.format != ge::FORMAT_ND || physical.empty() || ref.storage_offset < 0 || ref.storage_offset_expr != nullptr ||
      std::any_of(physical.begin(), physical.end(), [](int64_t value) {
        return value <= 0 || value > std::numeric_limits<int32_t>::max();
      }) || std::any_of(ref.stride_exprs.begin(), ref.stride_exprs.end(),
                       [](const auto &expr) { return expr != nullptr; }) ||
      std::any_of(ref.logical_shape_exprs.begin(), ref.logical_shape_exprs.end(),
                  [](const auto &expr) { return expr != nullptr; })) {
    return false;
  }
  int64_t capacity = 1;
  for (const auto dim : physical) {
    if (capacity > std::numeric_limits<int64_t>::max() / dim) return false;
    capacity *= dim;
  }
  return ref.storage_offset < capacity && ContiguousRowMajorStrides(physical, base_stride);
}

// Recover coordinates only when every affine address equals an ordinary
// unit-step slice of the physical dense allocation. Other geometries are
// handled separately or retain the existing general layout path.
bool SliceCandidate(const TensorRef &ref, const std::vector<int64_t> &base_stride,
                    DenseCandidate *candidate) {
  const auto &physical = PhysicalShape(ref);
  const auto &shape = LogicalShape(ref);
  if (shape.size() != physical.size() || ref.stride.size() != physical.size()) return false;
  std::vector<int64_t> begin(shape.size()), step(shape.size());
  int64_t offset = ref.storage_offset;
  size_t changed = 0;
  bool unit_step = true;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (base_stride[i] <= 0 || shape[i] <= 0 || ref.stride[i] <= 0 ||
        ref.stride[i] % base_stride[i] != 0) return false;
    begin[i] = offset / base_stride[i];
    offset %= base_stride[i];
    step[i] = ref.stride[i] / base_stride[i];
    if (begin[i] >= physical[i] || step[i] > std::numeric_limits<int32_t>::max() ||
        shape[i] - 1 > (physical[i] - 1 - begin[i]) / step[i]) return false;
    if (begin[i] != 0 || shape[i] != physical[i] || step[i] != 1) {
      ++changed;
    }
    unit_step = unit_step && step[i] == 1;
  }
  if (offset != 0 || changed == 0) return false;
  if (unit_step) {
    candidate->entry = &UnitSliceEntry();
    candidate->arguments = {Integers(begin), Integers(shape)};
    candidate->shapes = {shape};
  } else {
    // The available StridedSlice binaries / SliceV2 API regressed the measured
    // 910B cases. Keep the general AsStrided path until a faster ABI is validated.
    return false;
  }
  return true;
}

bool SingleAxisSelectionCandidate(const TensorRef &ref, const std::vector<int64_t> &base_stride,
                                  DenseCandidate *candidate) {
  const auto &physical = PhysicalShape(ref);
  const auto &shape = LogicalShape(ref);
  if (shape.size() + 1U != physical.size() || ref.stride.size() != shape.size()) return false;
  for (size_t axis = 0; axis < physical.size(); ++axis) {
    const int64_t index = ref.storage_offset / base_stride[axis];
    if (index >= physical[axis] || ref.storage_offset % base_stride[axis] != 0) continue;
    bool matches = true;
    for (size_t i = 0; i < shape.size(); ++i) {
      const size_t source_axis = i < axis ? i : i + 1U;
      matches = matches && shape[i] == physical[source_axis] && ref.stride[i] == base_stride[source_axis];
    }
    if (!matches) continue;
    std::vector<int64_t> begin(physical.size(), 0);
    begin[axis] = index;
    auto slice_shape = physical;
    slice_shape[axis] = 1;
    candidate->entry = &UnitSliceEntry();
    candidate->arguments = {Integers(begin), Integers(slice_shape)};
    // Slice keeps the selected axis at size one; the TensorRef being
    // materialized has already removed that axis with select semantics.
    candidate->shapes = {shape};
    return true;
  }
  return false;
}

bool HasImplementation(const DenseCandidate &candidate, ge::DataType dtype) {
  BinarySelectRequest request;
  request.opp_type = candidate.entry->opp_type;
  request.inputs.push_back({dtype, ge::FORMAT_ND});
  for (size_t i = 1; i < candidate.entry->arity; ++i) {
    request.inputs.push_back({ge::DT_INT32, ge::FORMAT_ND});
  }
  request.outputs.push_back({dtype, ge::FORMAT_ND});
  request.prefer_high_performance = candidate.entry->behavior == nullptr ||
                                    candidate.entry->behavior->prefer_high_performance;
  if (candidate.entry->behavior != nullptr && candidate.entry->behavior->select_suffix != nullptr) {
    request.attr_key_suffix = candidate.entry->behavior->select_suffix;
  }
  SelectedBinaryKernel binary;
  std::string error;
  return SelectBinaryKernel(request, binary, &error);
}

}  // namespace

bool TryLowerSpecializedDenseLayout(const fxrt::ir::NodePtr &input, const TensorRef &source,
                                   const std::string &tag, LoweringContext &ctx,
                                   LoweringResult *result, bool *matched, std::string *error) {
  *matched = false;
  std::vector<int64_t> base_stride;
  DenseCandidate candidate;
  if (!ConcreteGeometry(source, &base_stride) ||
      (!SingleAxisSelectionCandidate(source, base_stride, &candidate) &&
       !SliceCandidate(source, base_stride, &candidate)) ||
      !HasImplementation(candidate, source.dtype)) return true;
  *matched = true;
  auto adapter = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  adapter->op = fxrt::ops::Op_End;
  adapter->inputs = {input};
  adapter->inputs.insert(adapter->inputs.end(), candidate.arguments.begin(), candidate.arguments.end());
  std::vector<fxrt::ir::ValuePtr> outputs;
  for (const auto &shape : candidate.shapes) {
    auto tensor = fxrt::ir::MakeIntrusive<fxrt::ir::Tensor>(shape, input->output->ToTensor()->Dtype(),
                                                         input->output->ToTensor()->GetDevice());
    outputs.push_back(fxrt::ir::MakeIntrusive<fxrt::ir::Value>(tensor));
  }
  adapter->output = outputs.size() == 1U ? outputs[0] : fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
      fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(outputs)));
  TensorRef base = source;
  base.producer = nullptr;
  base.symbol = 0;
  base.shape = PhysicalShape(source);
  base.logical_shape.clear();
  base.logical_shape_exprs.clear();
  base.logical_shape_ep = {};
  base.storage_shape.clear();
  ContiguousRowMajorStrides(base.shape, &base.stride);
  base.stride_exprs.clear();
  base.storage_offset = 0;
  base.storage_offset_expr = nullptr;
  base.proven_dense = true;
  // A preceding dense reshape can retain a logical shape endpoint with another
  // rank. The specialized kernel addresses the physical allocation geometry.
  base.tiling_ep = {};
  if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_dense_base", base.shape,
                                    *ctx.sym_gen, ctx.kernel_nodes, &base.unpacked.shape.node,
                                    &base.unpacked.shape.sym, error)) {
    return false;
  }
  base.unpacked.shape.out = 0;
  const auto &name = ctx.name_of->at(input.get());
  const auto saved = ctx.tensors->at(name);
  const auto final_mask = ctx.output_final_mask;
  (*ctx.tensors)[name] = std::move(base);
  ctx.output_final_mask.assign(candidate.shapes.size(), false);
  const bool ok = LowerAiCoreBinaryWithCatalog(adapter, *candidate.entry, tag, ctx, result, error);
  (*ctx.tensors)[name] = saved;
  ctx.output_final_mask = final_mask;
  if (ok && result->outputs.size() == candidate.shapes.size()) {
    const auto selected = result->outputs[0];
    result->outputs = {selected};
    return true;
  }
  if (ok && error != nullptr) *error = "layout specialization: unexpected output count for " + tag;
  return false;
}

}  // namespace fx_converter
