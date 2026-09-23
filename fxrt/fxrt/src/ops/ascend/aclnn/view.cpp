#include "ops/ascend/aclnn/view.h"

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
    const std::vector<int64_t>& viewShapeArg) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "View output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");

  const auto strides = CalculateViewStrides(curShape, curStrides, inferredShape);
  if (strides.has_value()) {
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, strides.value());
    return;
  }
  RT_GLOG(EXCEPTION) << "View encountered unsupported non-contiguous input tensor. output shape: " << viewShapeArg
                     << " (inferred as " << inferredShape << "), input shape: " << curShape
                     << ", input stride: " << curStrides
                     << ". Consider calling .contiguous() on the input tensor at the corresponding operator call site.";
}
} // namespace

OpsErrorCode AclnnView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto& shape = input[kIndex1]->ToTuple()->ToIntList();
  if (std::any_of(shape.begin(), shape.end(), [](const int& shapeI) { return shapeI < -1; })) {
    RT_GLOG(EXCEPTION) << "For View the component of shape can't be less than -1";
  }
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), shape);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(view, AclnnView, Ascend);
} // namespace ops
} // namespace fxrt
