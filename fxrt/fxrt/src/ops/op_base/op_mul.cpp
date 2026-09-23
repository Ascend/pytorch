#include <vector>

#include "ops/op_base/op_mul.h"
#include "ops/utils/utils.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpMul::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  CHECK_IF_FAIL(input.size() == kInputSize2);
  const auto& input0 = input[kIndex0]->ToTensor();
  const auto& input1 = input[kIndex1]->ToTensor();
  CalBroadCastShape(input0->Shape(), input1->Shape(), &(output->ToTensor()->Shape()));
  output->ToTensor()->Resize();
  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
