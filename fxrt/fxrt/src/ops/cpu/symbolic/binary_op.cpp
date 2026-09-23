#include <vector>

#include "ops/cpu/symbolic/binary_op.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode BinaryOp::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  Operator::InferShape(input, output);
  if (!output->IsSymbol() && !output->IsInt() && !output->IsDouble()) {
    RT_GLOG(EXCEPTION) << "BinaryOp: output must be symbol/int/double";
  }
  return SUCCESS;
}

OpsErrorCode BinaryOp::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool BinaryOp::NeedLaunch() {
  return false;
}

FXRT_REG_OP(add_scalar, BinaryAdd, CPU);
FXRT_REG_OP(sub_scalar, BinarySub, CPU);
FXRT_REG_OP(mul_scalar, BinaryMul, CPU);
FXRT_REG_OP(div_scalar, BinaryDiv, CPU);
FXRT_REG_OP(div_mod_scalar, BinaryFloorDiv, CPU);
} // namespace ops
} // namespace fxrt
