#include "ops/ascend/aclnn/squeeze_view.h"

#include <algorithm>
#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
std::vector<int64_t> NormalizeDims(const std::vector<int64_t>& dims, int64_t dimSize) {
  std::vector<int64_t> normalized;
  normalized.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(normalized), [dimSize](int64_t dim) {
    return DynamicDimWrap(dim, dimSize, true);
  });

  auto sortedDims = normalized;
  std::sort(sortedDims.begin(), sortedDims.end());
  auto uniqueEnd = std::unique(sortedDims.begin(), sortedDims.end());
  CHECK_IF_FAIL_MSG(uniqueEnd == sortedDims.end(), "dim appears multiple times in the list of dims");
  return normalized;
}

void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& dims) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "Squeeze output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");

  const auto normalizedDims = NormalizeDims(dims, SizeToLong(curShape.size()));
  std::vector<bool> squeezeMask(curShape.size(), false);
  for (auto dim : normalizedDims) {
    if (LongToSize(dim) < squeezeMask.size()) {
      squeezeMask[LongToSize(dim)] = true;
    }
  }

  std::vector<int64_t> newStrides;
  newStrides.reserve(curStrides.size());
  for (size_t i = 0; i < curShape.size(); ++i) {
    if (squeezeMask[i] && curShape[i] == 1) {
      continue;
    }
    newStrides.emplace_back(curStrides[i]);
  }

  CHECK_IF_FAIL_MSG(
      inferredShape.size() == newStrides.size(),
      "Squeeze inferred output rank " + std::to_string(inferredShape.size()) + " does not match computed stride rank " +
          std::to_string(newStrides.size()));
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides, inputTensorPtr->StorageOffset());
}
} // namespace

OpsErrorCode AclnnSqueezeView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto& dims = input[kIndex1]->ToTuple()->ToIntList();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dims);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(squeeze_view, AclnnSqueezeView, Ascend);
} // namespace ops
} // namespace fxrt
