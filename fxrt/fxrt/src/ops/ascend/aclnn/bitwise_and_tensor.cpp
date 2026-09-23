#include <vector>

#include "ops/ascend/aclnn/bitwise_and_tensor.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnBitwiseAndTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnBitwiseAndTensor::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(bitwise_and_tensor, AclnnBitwiseAndTensor, Ascend);
FXRT_REG_OP_PROTOTYPE(bitwise_and_tensor, 2);
} // namespace ops
} // namespace fxrt
