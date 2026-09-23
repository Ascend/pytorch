#include "ops/ascend/aclnn/unbind_view.h"

#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr& inputTensorPtr, const ir::TuplePtr& outputTuple, const int64_t oriDim) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  CHECK_IF_FAIL_MSG(
      !curShape.empty(), "Dimension specified as " + std::to_string(oriDim) + " but tensor has no dimensions");

  const auto dim = DynamicDimWrap(oriDim, SizeToLong(curShape.size()));
  const auto dimValue = curShape[LongToSize(dim)];
  CHECK_IF_FAIL_MSG(
      outputTuple->Size() == LongToSize(dimValue),
      "Unbind output tuple size " + std::to_string(outputTuple->Size()) + " does not match input dim size " +
          std::to_string(dimValue));

  auto newStrides = curStrides;
  newStrides.erase(newStrides.begin() + dim);
  const auto storageOffset = inputTensorPtr->StorageOffset();
  for (size_t i = 0; i < outputTuple->Size(); ++i) {
    const auto outputTensorPtr = (*outputTuple)[i]->ToTensor();
    const auto& inferredShape = outputTensorPtr->Shape();
    CHECK_IF_FAIL_MSG(
        !outputTensorPtr->HasDynamicShape(),
        "Unbind output shape should have been inferred before CalcWorkspace, but got " +
            std::to_string(inferredShape.size()) + " dimensions with unresolved values");
    CHECK_IF_FAIL_MSG(
        inferredShape.size() + 1 == curShape.size(),
        "Unbind inferred output rank " + std::to_string(inferredShape.size()) + " does not match input rank " +
            std::to_string(curShape.size()) + " minus one");
    const auto newStorageOffset = storageOffset + LongToSize(SizeToLong(i) * curStrides[LongToSize(dim)]);
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides, newStorageOffset);
  }
}
} // namespace

OpsErrorCode AclnnUnbindView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple(), dim);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(unbind_view, AclnnUnbindView, Ascend);
} // namespace ops
} // namespace fxrt
