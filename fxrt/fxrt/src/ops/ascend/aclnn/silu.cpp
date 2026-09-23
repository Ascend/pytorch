#include <vector>

#include "ops/ascend/aclnn/silu.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnSilu::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnSilu::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(silu, AclnnSilu, Ascend);
FXRT_REG_OP_PROTOTYPE(silu, 1);
} // namespace ops
} // namespace fxrt
