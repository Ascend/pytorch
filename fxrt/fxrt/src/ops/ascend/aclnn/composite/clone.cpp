#include <vector>
#include "ops/ascend/aclnn/composite/clone.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnClone::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), output->ToTensor(), input[kIndex0]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnClone::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, output->ToTensor(), input[kIndex0]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(clone, AclnnClone, Ascend);
FXRT_REG_OP_PROTOTYPE(clone, 1);
} // namespace ops
} // namespace fxrt
