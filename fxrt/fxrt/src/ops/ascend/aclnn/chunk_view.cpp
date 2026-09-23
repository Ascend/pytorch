#include "ops/ascend/aclnn/chunk_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const std::vector<ir::TensorPtr>& outputTensorVector,
    int64_t chunks,
    int64_t dim) {
  const auto& curShape = inputTensorPtr->Shape();
  const auto& curStrides = GetTensorStrides(inputTensorPtr);
  auto curOffset = inputTensorPtr->StorageOffset();
  const auto ndim = curShape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For 'Chunk', input's rank should be greater than 0, but got " + std::to_string(ndim));
  CHECK_IF_FAIL_MSG(chunks > 0, "For 'Chunk', chunks should be greater than 0, but got " + std::to_string(chunks));

  const auto wrapDim = DynamicDimWrap(dim, ndim);
  CHECK_IF_FAIL_MSG(!outputTensorVector.empty(), "For 'Chunk', output tensor size should be greater than 0");
  for (const auto& outputTensor : outputTensorVector) {
    const auto& inferredShape = outputTensor->Shape();
    CHECK_IF_FAIL_MSG(
        !outputTensor->HasDynamicShape(),
        "Chunk output shape should have been inferred before CalcWorkspace, but got " +
            std::to_string(inferredShape.size()) + " dimensions with unresolved values");
    CHECK_IF_FAIL_MSG(
        inferredShape.size() == ndim,
        "Chunk inferred output rank " + std::to_string(inferredShape.size()) + " does not match input rank " +
            std::to_string(ndim));
    UpdateTensorViewInfo(inputTensorPtr, outputTensor, inferredShape, curStrides, curOffset);
    curOffset += LongToSize(inferredShape[wrapDim] * curStrides[wrapDim]);
  }
}
} // namespace

OpsErrorCode AclnnChunkView::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto chunks = input[kIndex1]->ToInt();
  CHECK_IF_FAIL_MSG(chunks >= 0, "chunks must be positive, but got " + std::to_string(chunks));
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple()->ToTensorList(), chunks, dim);
  return SUCCESS;
}

FXRT_REG_OP(chunk_view, AclnnChunkView, Ascend);
} // namespace ops
} // namespace fxrt
