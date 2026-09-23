#include <vector>

#include "ops/ascend/aclnn/mm.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnMm::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  cubeMathType_ = GetCubeMathType();
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kIndex0]->ToTensor(),
      input[kIndex1]->ToTensor(),
      output->ToTensor(),
      cubeMathType_);
  return SUCCESS;
}

OpsErrorCode AclnnMm::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kIndex0]->ToTensor(),
      input[kIndex1]->ToTensor(),
      output->ToTensor(),
      cubeMathType_);
  return SUCCESS;
}

FXRT_REG_OP(mm, AclnnMm, Ascend);
FXRT_REG_OP_PROTOTYPE(mm, 2);
} // namespace ops
} // namespace fxrt
