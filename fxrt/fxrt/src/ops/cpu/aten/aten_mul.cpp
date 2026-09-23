#include <vector>

#include "ops/cpu/aten/aten_mul.h"
#include "ops/utils/data_convert.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AtenMul::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto atenInput0 = ToTorchTensor(input[kIndex0]->ToTensor());
  auto atenInput1 = ToTorchTensor(input[kIndex1]->ToTensor());
  auto atenOutput = ToTorchTensor(output->ToTensor());
  at::mul_out(atenOutput, atenInput0, atenInput1);
  return SUCCESS;
}

FXRT_REG_OP(mul, AtenMul, CPU);
} // namespace ops
} // namespace fxrt
