#include <vector>

#include "ops/cpu/aten/aten_reshape.h"
#include "ops/utils/data_convert.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AtenReshape::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto atenInput0 = ToTorchTensor(input[kIndex0]->ToTensor());
  auto atenOutput = ToTorchTensor(output->ToTensor());
  auto outputShape = output->ToTensor()->Shape();
  at::resize_out(atenOutput, atenInput0, outputShape);
  return SUCCESS;
}

FXRT_REG_OP(reshape, AtenReshape, CPU);
} // namespace ops
} // namespace fxrt
