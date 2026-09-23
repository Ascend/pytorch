#include "ops/ascend/aclnn/as_strided_view.h"

#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
constexpr size_t kAsStridedMinInputNum = 3;
constexpr size_t kAsStridedMaxInputNum = 4;

void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    int64_t storageOffset) {
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "AsStrided output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  CHECK_IF_FAIL_MSG(
      size.size() == stride.size(),
      "as_strided size and stride must have the same length, but got size length " + std::to_string(size.size()) +
          " and stride length " + std::to_string(stride.size()));
  CHECK_IF_FAIL_MSG(
      inferredShape.size() == size.size(),
      "AsStrided inferred output rank " + std::to_string(inferredShape.size()) + " does not match size length " +
          std::to_string(size.size()));

  for (size_t i = 0; i < size.size(); ++i) {
    CHECK_IF_FAIL_MSG(
        size[i] >= 0,
        "as_strided size must be non-negative, but got " + std::to_string(size[i]) + " at dim " + std::to_string(i));
    CHECK_IF_FAIL_MSG(
        stride[i] >= 0,
        "as_strided stride must be non-negative, but got " + std::to_string(stride[i]) + " at dim " +
            std::to_string(i));
    CHECK_IF_FAIL_MSG(
        inferredShape[i] == size[i],
        "AsStrided inferred output shape mismatch at dim " + std::to_string(i) + ": inferred " +
            std::to_string(inferredShape[i]) + ", size arg " + std::to_string(size[i]));
  }

  CHECK_IF_FAIL_MSG(
      storageOffset >= 0, "as_strided storage_offset must be non-negative, but got " + std::to_string(storageOffset));
  const auto newStorageOffset = inputTensorPtr->StorageOffset() + LongToSize(storageOffset);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, stride, newStorageOffset);
}
} // namespace

OpsErrorCode AclnnAsStridedView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_FAIL_MSG(
      input.size() >= kAsStridedMinInputNum && input.size() <= kAsStridedMaxInputNum,
      "as_strided_view expects 3 or 4 inputs, but got " + std::to_string(input.size()));
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto size = input[kIndex1]->ToTuple()->ToIntList();
  const auto stride = input[kIndex2]->ToTuple()->ToIntList();
  const auto storageOffset = input.size() > kIndex3 ? input[kIndex3]->ToInt() : 0;
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), size, stride, storageOffset);
  CheckStorageMatch(input, output);
  *workspaceSize = 0;
  return SUCCESS;
}

FXRT_REG_OP(as_strided_view, AclnnAsStridedView, Ascend);
} // namespace ops
} // namespace fxrt
