#include <vector>

#include "ops/op_base/op_all_to_all.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpAllToAll::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  RT_VLOG(VL_OPS) << "HcclAllToAll InferShape";

  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto outputShape = input0Shape;
  auto outputSplitSizes = input[kIndex1]->ToTuple();
  int64_t outputSize = 0;
  for (size_t i = 0; i < outputSplitSizes->Size(); ++i) {
    outputSize += outputSplitSizes->operator[](i)->ToInt();
  }
  outputShape[0] = outputSize;
  auto outputTensor = output->ToTensor();
  CHECK_IF_NULL(outputTensor);
  outputTensor->SetShape(outputShape);
  auto outputDtype = input[kIndex0]->ToTensor()->Dtype();
  outputTensor->SetDtype(outputDtype);
  outputTensor->Resize();

  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
