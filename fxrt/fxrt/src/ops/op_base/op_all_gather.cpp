#include <vector>

#include "ops/op_base/op_all_gather.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpAllGather::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  RT_VLOG(VL_OPS) << "HcclAllGather InferShape";

  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto rankSize = input[kIndex1]->ToInt();

  auto outputShape = input0Shape;
  outputShape[0] *= rankSize;

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
