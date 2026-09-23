#include <vector>

#include "ops/ascend/aclnn/div_mod_scalar.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnDivModScalar::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1], input[2]->ToInt(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnDivModScalar::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(
      workspace, workspaceSize, stream, input[0]->ToTensor(), input[1], input[2]->ToInt(), output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(div_mod_scalar, AclnnDivModScalar, Ascend);
} // namespace ops
} // namespace fxrt
