#include "ops/ascend/aclnn/narrow_view.h"

#include <vector>

#include "common/common.h"
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
    const int64_t length) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  const auto& inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(
      !outputTensorPtr->HasDynamicShape(),
      "Narrow output shape should have been inferred before CalcWorkspace, but got " +
          std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  CHECK_IF_FAIL_MSG(!curShape.empty(), "narrow() cannot be applied to a 0-dim tensor.");
  CHECK_IF_FAIL_MSG(length >= 0, "narrow(): length must be non-negative.");

  const auto dim = DynamicDimWrap(oriDim, SizeToLong(curShape.size()));
  const auto dimValue = curShape[LongToSize(dim)];
  CHECK_IF_FAIL_MSG(
      oriStart >= -dimValue && oriStart <= dimValue,
      "start out of range (expected to be in range of [" + std::to_string(-dimValue) + ", " + std::to_string(dimValue) +
          "], but got " + std::to_string(oriStart) + ")");

  const auto start = oriStart < 0 ? oriStart + dimValue : oriStart;
  CHECK_IF_FAIL_MSG(
      start <= dimValue - length,
      "start (" + std::to_string(start) + ") + length (" + std::to_string(length) + ") exceeds dimension size (" +
          std::to_string(dimValue) + ").");
  CHECK_IF_FAIL_MSG(
      inferredShape.size() == curShape.size(),
      "Narrow inferred output rank " + std::to_string(inferredShape.size()) + " does not match input rank " +
          std::to_string(curShape.size()));
  CHECK_IF_FAIL_MSG(
      inferredShape[LongToSize(dim)] == length,
      "Narrow inferred output shape has unexpected length at dim " + std::to_string(dim));

  const auto newStorageOffset = inputTensorPtr->StorageOffset() + LongToSize(start * curStrides[LongToSize(dim)]);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, curStrides, newStorageOffset);
}
} // namespace

OpsErrorCode AclnnNarrowView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto start = input[kIndex2]->ToInt();
  const auto length = input[kIndex3]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim, start, length);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

FXRT_REG_OP(narrow_view, AclnnNarrowView, Ascend);
} // namespace ops
} // namespace fxrt
