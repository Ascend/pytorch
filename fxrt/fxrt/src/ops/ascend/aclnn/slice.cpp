#include <vector>

#include "ops/ascend/aclnn/slice.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnSlice::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToInt(),
      input[3]->ToInt(),
      input[4]->ToInt(),
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnSlice::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToInt(),
      input[3]->ToInt(),
      input[4]->ToInt(),
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(slice, AclnnSlice, Ascend);
} // namespace ops
} // namespace fxrt
