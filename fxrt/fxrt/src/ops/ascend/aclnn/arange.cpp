#include <vector>

#include "ops/ascend/aclnn/arange.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnArange::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), input[0], input[1], input[2], output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnArange::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0], input[1], input[2], output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(arange, AclnnArange, Ascend);
} // namespace ops
} // namespace fxrt
