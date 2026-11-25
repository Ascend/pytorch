#include "lowering/common/output_layout_planner.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "lowering/common/view_utils.h"
#include "exe_graph/execute_graph_builder.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {
namespace {
bool Product(const std::vector<int64_t> &shape, int64_t *result) {
  *result = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || *result > std::numeric_limits<int64_t>::max() / dim) {
      return false;
    }
    *result *= dim;
  }
  return true;
}

bool IsFullStorageView(fxrt::ops::Op op) {
  switch (op) {
    case fxrt::ops::Op_view:
    case fxrt::ops::Op_reshape:
    case fxrt::ops::Op_alias:
    case fxrt::ops::Op_flatten_view:
    case fxrt::ops::Op_permute_view:
    case fxrt::ops::Op_squeeze:
    case fxrt::ops::Op_squeeze_dim:
    case fxrt::ops::Op_squeeze_view:
    case fxrt::ops::Op_unsqueeze:
    case fxrt::ops::Op_unsqueeze_view:
      return true;
    default:
      return false;
  }
}

bool IsCompleteStorageRoot(const fxrt::ir::NodePtr &node, const TensorRef &ref) {
  if (LogicalShape(ref) == PhysicalShape(ref)) return true;
  auto origin = node;
  bool preserves_elements = false;
  while (origin != nullptr && IsFullStorageView(origin->op) && !origin->inputs.empty()) {
    preserves_elements = true;
    origin = origin->inputs[0];
  }
  return preserves_elements && origin != nullptr && !IsStorageSharingViewOp(origin->op);
}

}  // namespace

bool PlanOutputStorage(const fxrt::ir::NodePtr &node, const TensorRef &ref,
                       OutputStorageCandidate *candidate) {
  if (!IsEsGraphRunnerEnabled() || !ref.HasUnpackedPair() || ref.storage_offset != 0 ||
      ref.can_adopt_for_model_output) {
    return false;
  }
  const auto owner = ref.allocation.Valid() ? ref.allocation : ref.unpacked.addr;
  // Borrowed feeds/fallback pointers and rebased addresses do not establish a
  // physical capacity. Preserve their existing output path.
  if (owner.node != ref.unpacked.addr.node || owner.out != ref.unpacked.addr.out ||
      (std::string(owner.node->GetType()) != "AllocMemory" &&
       std::string(owner.node->GetType()) != "AllocModelOutTensor")) {
    return false;
  }
  OutputStorageCandidate plan;
  plan.source = ref;
  plan.shape = LogicalShape(ref);
  plan.stride = ref.stride;
  if (plan.stride.empty() && !ContiguousRowMajorStrides(plan.shape, &plan.stride)) {
    return false;
  }
  int64_t logical_count = 0;
  int64_t capacity = 0;
  if (plan.shape.size() != plan.stride.size() || !Product(plan.shape, &logical_count) ||
      !Product(PhysicalShape(ref), &capacity)) {
    return false;
  }
  // Remove broadcast and singleton axes, then prove dense physical coverage.
  // This excludes holes, negative strides and overlapping non-broadcast layouts.
  std::vector<std::pair<int64_t, int64_t>> axes;
  for (size_t i = 0; i < plan.shape.size(); ++i) {
    if (plan.stride[i] < 0) {
      return false;
    }
    if (plan.shape[i] > 1 && plan.stride[i] != 0) {
      axes.emplace_back(plan.stride[i], plan.shape[i]);
    }
  }
  std::sort(axes.begin(), axes.end());
  int64_t span = 1;
  for (const auto &axis : axes) {
    if (axis.first != span || span > std::numeric_limits<int64_t>::max() / axis.second) {
      return false;
    }
    span *= axis.second;
  }
  const auto item_size = ge::GetSizeByDataType(ref.dtype);
  if (span > capacity || item_size <= 0 ||
      static_cast<uint64_t>(span) > std::numeric_limits<size_t>::max() / item_size) {
    return false;
  }
  plan.elements = span;
  plan.noncontiguous = !IsContiguousRowMajor(plan.shape, plan.stride);
  auto origin = node;
  while (origin != nullptr) {
    if (origin->op == fxrt::ops::Op_clone) {
      plan.isolation = origin.get();
      break;
    }
    if (!IsStorageSharingViewOp(origin->op) || origin->inputs.empty()) {
      break;
    }
    origin = origin->inputs[0];
  }
  *candidate = std::move(plan);
  return true;
}

bool SameOutputStorage(const OutputStorageCandidate &a, const OutputStorageCandidate &b) {
  return a.source.unpacked.addr.node == b.source.unpacked.addr.node &&
         a.source.unpacked.addr.out == b.source.unpacked.addr.out &&
         a.source.dtype == b.source.dtype && a.isolation == b.isolation;
}

bool PlanReturnedRootView(const fxrt::ir::NodePtr &root_node, const TensorRef &root,
                          const fxrt::ir::NodePtr &node, const TensorRef &ref,
                          OutputViewSpec *view) {
  const auto root_shape = LogicalShape(root);
  const auto ref_shape = LogicalShape(ref);
  if (view == nullptr || root_node == nullptr || node == nullptr || root.feed_index >= 0 ||
      root.can_adopt_for_model_output || ref.can_adopt_for_model_output ||
      root.format != ge::FORMAT_ND || ref.format != root.format || ref.dtype != root.dtype ||
      std::find(root_shape.begin(), root_shape.end(), 0) != root_shape.end() ||
      std::find(ref_shape.begin(), ref_shape.end(), 0) != ref_shape.end() ||
      !root.HasUnpackedPair() || !ref.HasUnpackedPair() || !IsDenseTensorRef(root) ||
      !IsCompleteStorageRoot(root_node, root) || root.storage_offset != 0 ||
      root.storage_offset_expr != nullptr || root.runtime_storage_offset.Valid() ||
      ref.runtime_storage_offset.Valid()) {
    return false;
  }
  const auto owner = root.allocation.Valid() ? root.allocation : root.unpacked.addr;
  const auto ref_owner = ref.allocation.Valid() ? ref.allocation : ref.unpacked.addr;
  if (!owner.Valid() || owner.node != root.unpacked.addr.node || owner.out != root.unpacked.addr.out ||
      owner.node != ref_owner.node || owner.out != ref_owner.out ||
      owner.node != ref.unpacked.addr.node || owner.out != ref.unpacked.addr.out) {
    return false;
  }
  const bool direct = root.producer != nullptr && root.producer->GetType() == "OutputData";
  if (!direct) {
    if (owner.node->GetType() != "AllocMemory") return false;
  }
  // Multiple output wrappers may be sibling views of the same allocation.
  // Require each non-root wrapper to be a view chain; the owner equality above
  // proves that it reaches this root's storage without relying on node names.
  if (node.get() != root_node.get()) {
    auto origin = node;
    bool has_view = false;
    while (origin != nullptr && IsStorageSharingViewOp(origin->op) && !origin->inputs.empty()) {
      has_view = true;
      origin = origin->inputs[0];
    }
    if (!has_view || origin == nullptr) return false;
  }
  OutputViewSpec result;
  result.preserve_layout = true;
  result.shape = LogicalShape(ref);
  result.stride = ref.stride;
  result.storage_offset = ref.storage_offset;
  result.shape_exprs = ref.logical_shape_exprs;
  result.stride_exprs = ref.stride_exprs;
  result.offset_expr = ref.storage_offset_expr;
  result.derive_dense_stride = ref.proven_dense || ref.stride.empty();
  if ((!result.shape_exprs.empty() && result.shape_exprs.size() != result.shape.size()) ||
      (!result.stride_exprs.empty() && result.stride_exprs.size() != result.stride.size()) ||
      (!result.derive_dense_stride && result.stride.size() != result.shape.size())) return false;
  for (size_t i = 0; i < result.shape.size(); ++i) {
    if (result.shape[i] < 0 && (result.shape_exprs.empty() || result.shape_exprs[i] == nullptr)) return false;
  }
  for (size_t i = 0; i < result.stride.size(); ++i) {
    if (!result.derive_dense_stride && result.stride[i] < 0 &&
        (result.stride_exprs.empty() || result.stride_exprs[i] == nullptr)) return false;
  }
  if (result.storage_offset < 0 && result.offset_expr == nullptr) return false;
  *view = std::move(result);
  return true;
}
}  // namespace fx_converter
