#include "ops/operator.h"
#include <vector>

namespace fxrt {
namespace ops {

OpsErrorCode Operator::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  ir::VisitAllTensors(output, [](const ir::TensorPtr& tensor) {
    if (tensor->HasSymbolicShape()) {
      tensor->EvalSymbolicShape();
    }
    if (tensor->HasDynamicShape()) {
      RT_GLOG(EXCEPTION) << "Tensor shape still unknown before launch: " << tensor;
    }
  });
  RT_VLOG(VL_OPS) << "Operator output shape inferred: " << *output;
  return SUCCESS;
}

} // namespace ops
} // namespace fxrt
