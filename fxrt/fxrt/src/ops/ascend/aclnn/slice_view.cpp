#include "ops/ascend/aclnn/slice_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const int64_t oriDim,
    const int64_t oriStart,
    const int64_t oriEnd,
    const int64_t step) {
  (void)oriEnd;
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "Slice output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  const auto dimSize = curShape.size();
  CHECK_IF_FAIL_MSG(dimSize > 0, "slice can not be applied to a 0-dim tensor.");
  const auto dim = DynamicDimWrap(oriDim, dimSize);
  const auto dimValue = curShape[dim];

  auto start = oriStart < 0 ? oriStart + dimValue : oriStart;
  if (start < 0) {
    start = 0;
  } else if (start > dimValue) {
    start = dimValue;
  }

  auto newStrides = curStrides;
  newStrides[dim] *= step;
  const auto storageOffset = inputTensorPtr->StorageOffset();
  const size_t newStorageOffset = storageOffset + LongToSize(start * curStrides[dim]);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides, newStorageOffset);
}
} // namespace

OpsErrorCode AclnnSliceView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto start = input[kIndex2]->ToInt();
  const auto end = input[kIndex3]->ToInt();
  const auto step = input[kIndex4]->ToInt();
  CHECK_IF_FAIL_MSG(step > 0, "step must be positive");
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim, start, end, step);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(slice_view, AclnnSliceView, Ascend);
} // namespace ops
} // namespace fxrt
