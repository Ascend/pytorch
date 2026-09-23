#include <vector>

#include "ops/op_base/op_reduce_scatter.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpReduceScatter::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  RT_VLOG(VL_OPS) << "HcclReduceScatter InferShape";

  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto rankSize = input[kIndex2]->ToInt();
  if (rankSize <= 0) {
    RT_GLOG(ERROR) << "HcclReduceScatter got invalid rank size: " << rankSize;
    return INVALID_PARAM;
  }
  auto outputShape = input0Shape;
  outputShape[kIndex0] /= rankSize;

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
