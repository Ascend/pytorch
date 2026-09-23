#include "ops/ascend/aclnn/permute_view.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
std::vector<int64_t> NormalizeDims(const std::vector<int64_t>& dims, int64_t dimSize) {
  CHECK_IF_FAIL_MSG(
      SizeToLong(dims.size()) == dimSize,
      "permute expects dims length equal to input rank, but got dims size " + std::to_string(dims.size()) +
          " and input rank " + std::to_string(dimSize));

  std::vector<int64_t> normalized;
  normalized.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(normalized), [dimSize](int64_t dim) {
    return DynamicDimWrap(dim, dimSize);
  });

  auto sortedDims = normalized;
  std::sort(sortedDims.begin(), sortedDims.end());
  auto uniqueEnd = std::unique(sortedDims.begin(), sortedDims.end());
  CHECK_IF_FAIL_MSG(uniqueEnd == sortedDims.end(), "permute dims must be unique");
  return normalized;
}

void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& dims) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto dimSize = SizeToLong(curShape.size());
  const auto normalizedDims = NormalizeDims(dims, dimSize);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "Permute output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  CHECK_IF_FAIL_MSG(
      inferredShape.size() == normalizedDims.size(),
      "Permute inferred output rank " + std::to_string(inferredShape.size()) + " does not match dims size " +
          std::to_string(normalizedDims.size()));
  const auto& curStrides = GetTensorStrides(inputTensorPtr);

  std::vector<int64_t> newStrides;
  newStrides.reserve(curStrides.size());
  std::transform(
      normalizedDims.begin(), normalizedDims.end(), std::back_inserter(newStrides), [&curStrides](int64_t dim) {
        return curStrides[LongToSize(dim)];
      });

  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides, inputTensorPtr->StorageOffset());
}
} // namespace

OpsErrorCode AclnnPermuteView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto& dims = input[kIndex1]->ToTuple()->ToIntList();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dims);
  CheckStorageMatch(input, output);
  *workspaceSize = 0;
  return SUCCESS;
}

FXRT_REG_OP(permute_view, AclnnPermuteView, Ascend);
} // namespace ops
} // namespace fxrt
