#include <vector>

#include "ops/ascend/aclnn/aclnn_zeros.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnZeros::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnZeros::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(zeros, AclnnZeros, Ascend);
} // namespace ops
} // namespace fxrt
