#include <vector>

#include "ops/cpu/aten/aten_matmul.h"
#include "ops/utils/data_convert.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AtenMatMul::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto atenInput0 = ToTorchTensor(input[kIndex0]->ToTensor());
  auto atenInput1 = ToTorchTensor(input[kIndex1]->ToTensor());
  auto atenOutput = ToTorchTensor(output->ToTensor());
  at::matmul_out(atenOutput, atenInput0, atenInput1);
  return SUCCESS;
}

FXRT_REG_OP(matmul, AtenMatMul, CPU);
} // namespace ops
} // namespace fxrt
