#include <vector>

#include "ops/op_base/op_matmul.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpMatMul::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  if (input.size() != kInputSize2) {
    RT_GLOG(ERROR) << "Expect input size is 2 for AtenMatMul, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto& input1Shape = input[kIndex1]->ToTensor()->Shape();
  auto& outputTensor = output->ToTensor();
  auto& outputShape = outputTensor->Shape();
  outputShape.resize(2);
  outputShape[0] = input0Shape[0];
  outputShape[1] = input1Shape[1];
  outputTensor->Resize();
  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
