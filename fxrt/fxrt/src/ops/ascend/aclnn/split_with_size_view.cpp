#include "ops/ascend/aclnn/split_with_size_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
void SplitSizeInputsCheck(
    const std::vector<int64_t>& splitSize,
    const int64_t& axis,
    const std::vector<int64_t>& tensorShape) {
  CHECK_IF_FAIL_MSG(
      splitSize.size() > 0,
      "For SplitWithSize, the size of splitSize should > 0, but got" + std::to_string(splitSize.size()));
  const int64_t sumSplitSize = std::accumulate(splitSize.begin(), splitSize.end(), 0);
  if (sumSplitSize != tensorShape[axis]) {
    RT_GLOG(EXCEPTION) << "For 'SplitWithSize',  the sum of splitSize should be equal to " << tensorShape[axis]
                       << "(input.shape[" << axis << "]), but got splitSize: " << splitSize;
  }
}

void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const std::vector<ir::TensorPtr>& outputTensorVector,
    const std::vector<int64_t>& splitSize,
    int64_t dim) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  auto curOffset = inputTensorPtr->StorageOffset();
  const auto rank = SizeToLong(curShape.size());
  CHECK_IF_FAIL_MSG(rank > 0, "For SplitWithSize, rank should > 0, but got" + std::to_string(rank));
  const auto ndim = curShape.size();
  const auto wrapDim = DynamicDimWrap(dim, ndim);
  SplitSizeInputsCheck(splitSize, wrapDim, curShape);

  CHECK_IF_FAIL_MSG(
      splitSize.size() == outputTensorVector.size(),
      "For SplitWithSize, the size of splitSize is " + std::to_string(splitSize.size()) +
          " and the size of outputTensorVector is " + std::to_string(outputTensorVector.size()));
  for (size_t i = 0; i < splitSize.size(); ++i) {
    const auto splitIter = splitSize[i];
    const auto& inferredShape = outputTensorVector[i]->Shape();
    CHECK_IF_FAIL_MSG(
        !outputTensorVector[i]->HasDynamicShape(),
        "SplitWithSize output shape should have been inferred before CalcWorkspace, but got " +
            std::to_string(inferredShape.size()) + " dimensions with unresolved values");
    CHECK_IF_FAIL_MSG(
        inferredShape.size() == curShape.size(),
        "SplitWithSize inferred output rank " + std::to_string(inferredShape.size()) + " does not match input rank " +
            std::to_string(curShape.size()));
    CHECK_IF_FAIL_MSG(
        inferredShape[wrapDim] == splitIter,
        "SplitWithSize inferred output shape " + ir::ShapeToString(inferredShape) + " does not match split size " +
            std::to_string(splitIter) + " at dim " + std::to_string(wrapDim));
    UpdateTensorViewInfo(inputTensorPtr, outputTensorVector[i], inferredShape, curStrides, curOffset);
    curOffset += LongToSize(splitIter * curStrides[wrapDim]);
  }
}
} // namespace

OpsErrorCode AclnnSplitWithSizeView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto splitSize = input[kIndex1]->ToTuple()->ToIntList();
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple()->ToTensorList(), splitSize, dim);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(split_with_size_view, AclnnSplitWithSizeView, Ascend);
} // namespace ops
} // namespace fxrt
