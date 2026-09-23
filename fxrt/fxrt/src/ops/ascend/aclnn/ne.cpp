#include <vector>

#include "ops/ascend/aclnn/ne.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnNe::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnNe::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(ne, AclnnNe, Ascend);
FXRT_REG_OP_PROTOTYPE(ne, 2);
} // namespace ops
} // namespace fxrt
