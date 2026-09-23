#include "ops/ascend/aclnn/unsqueeze_view.h"

#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const int64_t oriDim) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "Unsqueeze output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  CHECK_IF_FAIL_MSG(
      inferredShape.size() == curShape.size() + 1,
      "Unsqueeze inferred output rank " + std::to_string(inferredShape.size()) + " does not match input rank " +
          std::to_string(curShape.size()) + " plus one");

  const auto dim = DynamicDimWrap(oriDim, SizeToLong(curShape.size()) + 1);
  auto newStrides = curStrides;
  const auto newStride =
      LongToSize(dim) >= curShape.size() ? 1 : curShape[LongToSize(dim)] * curStrides[LongToSize(dim)];
  newStrides.insert(newStrides.begin() + dim, newStride);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides, inputTensorPtr->StorageOffset());
}
} // namespace

OpsErrorCode AclnnUnsqueezeView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(unsqueeze_view, AclnnUnsqueezeView, Ascend);
} // namespace ops
} // namespace fxrt
