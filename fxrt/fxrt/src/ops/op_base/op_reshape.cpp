#include <vector>

#include "ops/op_base/op_reshape.h"
#include "ir/value/value.h"
#include "common/logger.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpReshape::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  if (input.size() != kInputSize2) {
    RT_GLOG(ERROR) << "Expect input size is 2, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  if (!input[kIndex1]->IsTuple()) {
    RT_GLOG(ERROR) << "Input types are invalid, expect tuple on second input.";
    return INVALID_PARAM;
  }
  auto& shapeTuple = input[kIndex1]->ToTuple();

  auto& outputTensor = output->ToTensor();
  auto& outputShape = outputTensor->Shape();
  outputShape.clear();

  bool hasNegativeDim = false;
  int64_t knownDimProduct = 1;
  for (auto& dimValue : *shapeTuple) {
    int64_t dim = dimValue->ToInt();
    if (dim < 0) {
      if (hasNegativeDim) {
        RT_GLOG(EXCEPTION) << "Input shape tuple has more than one negative dimension.";
      }
      hasNegativeDim = true;
    } else {
      knownDimProduct *= dim;
    }
    (void)outputShape.emplace_back(dim);
  }

  if (hasNegativeDim) {
    int64_t inputNumel = input[kIndex0]->ToTensor()->Numel();
    if (inputNumel % knownDimProduct != 0) {
      RT_GLOG(EXCEPTION) << "Input tensor size is invalid for inferring the negative dimension.";
    }
    for (auto& dim : outputShape) {
      // cppcheck-suppress useStlAlgorithm
      if (dim < 0) {
        dim = inputNumel / knownDimProduct;
        break;
      }
    }
  }

  outputTensor->Resize();
  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
