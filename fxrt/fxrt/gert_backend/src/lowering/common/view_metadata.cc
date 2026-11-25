#include "lowering/common/view_metadata.h"

#include <algorithm>

#include "lowering/common/view_utils.h"
#include "shape/fx_sym_expr_parser.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {
bool HasExpression(const std::vector<fxrt::ir::SymbolicExprPtr> &expressions) {
  return std::any_of(expressions.begin(), expressions.end(), [](const auto &expr) { return expr != nullptr; });
}

bool SameValues(const std::vector<int64_t> &left, const std::vector<fxrt::ir::SymbolicExprPtr> &left_exprs,
                const std::vector<int64_t> &right, const std::vector<fxrt::ir::SymbolicExprPtr> &right_exprs) {
  if (left.size() != right.size()) return false;
  if ((!left_exprs.empty() && left_exprs.size() != left.size()) ||
      (!right_exprs.empty() && right_exprs.size() != right.size())) return false;
  for (size_t i = 0; i < left.size(); ++i) {
    const auto lhs = left_exprs.empty() ? fxrt::ir::SymbolicExprPtr{} : left_exprs[i];
    const auto rhs = right_exprs.empty() ? fxrt::ir::SymbolicExprPtr{} : right_exprs[i];
    if (lhs != nullptr || rhs != nullptr) {
      if (lhs == nullptr || rhs == nullptr || lhs->ToString() != rhs->ToString()) return false;
    } else if (left[i] < 0 || left[i] != right[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

bool ReadNativeViewLayout(const fxrt::ir::NodePtr &node, const TensorRef &source, FxSymbolEnv *env,
                           NativeViewLayout *layout) {
  if (node == nullptr || env == nullptr || layout == nullptr) return false;
  const auto attribute = node->attrs.find("fxrt_native_view_layouts");
  if (attribute == node->attrs.end() || attribute->second == nullptr || !attribute->second->IsTuple()) return false;
  const auto &candidates = attribute->second->ToTuple();
  if (candidates == nullptr || candidates->Size() < 2U) return false;
  const bool dense = source.proven_dense || source.stride.empty() ||
                     (!HasExpression(source.stride_exprs) &&
                      IsContiguousRowMajor(LogicalShape(source), source.stride));
  auto source_shape_exprs = source.logical_shape_exprs;
  if (source_shape_exprs.empty() && !node->inputs.empty()) {
    (void)SymbolicShapeExprsOfNode(node->inputs[0], *env, &source_shape_exprs);
  }
  const auto read_candidate = [&](const fxrt::ir::ValuePtr &value, NativeViewLayout *candidate) {
    if (value == nullptr || !value->IsTuple()) return false;
    const auto &record = value->ToTuple();
    if (record == nullptr || record->Size() != 7U) return false;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_stride;
    std::vector<fxrt::ir::SymbolicExprPtr> input_shape_exprs;
    std::vector<fxrt::ir::SymbolicExprPtr> input_stride_exprs;
    for (size_t i = 0; i < 4U; ++i) if ((*record)[i] == nullptr || !(*record)[i]->IsTuple()) return false;
    if (!ReadIntListWithSymbols((*record)[0]->ToTuple(), *env, &input_shape, &input_shape_exprs) ||
        !ReadIntListWithSymbols((*record)[1]->ToTuple(), *env, &input_stride, &input_stride_exprs) ||
        !SameValues(LogicalShape(source), source_shape_exprs, input_shape, input_shape_exprs) ||
        (!dense && !SameValues(source.stride, source.stride_exprs, input_stride, input_stride_exprs)) ||
        !ReadIntListWithSymbols((*record)[2]->ToTuple(), *env, &candidate->size, &candidate->size_exprs) ||
        !ReadIntListWithSymbols((*record)[3]->ToTuple(), *env, &candidate->stride, &candidate->stride_exprs) ||
        candidate->size.size() != candidate->stride.size()) return false;
    const auto &offset = (*record)[4];
    if (offset == nullptr) return false;
    if (offset->IsInt()) {
      candidate->relative_offset = offset->ToInt();
    } else if (offset->IsSymbol()) {
      candidate->relative_offset_expr = ParseFxSymExpr(offset->ToSymbol(), *env);
      if (candidate->relative_offset_expr == nullptr) return false;
    } else {
      return false;
    }
    if ((*record)[5] == nullptr || !(*record)[5]->IsBool() ||
        (*record)[6] == nullptr || !(*record)[6]->IsBool()) return false;
    candidate->aliases = (*record)[5]->ToBool();
    candidate->contiguous = (*record)[6]->ToBool();
    return true;
  };
  for (size_t index = 0; index < candidates->Size(); ++index) {
    if (dense && index != 1U) continue;
    NativeViewLayout candidate;
    if (read_candidate((*candidates)[index], &candidate)) {
      *layout = std::move(candidate);
      return true;
    }
  }
  return false;
}
}  // namespace fx_converter
