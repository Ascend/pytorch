#include <vector>

#include "ops/ascend/aclnn/index_select.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnIndexSelect::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToTensor(),
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnIndexSelect::Launch(
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
      input[2]->ToTensor(),
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(index_select, AclnnIndexSelect, Ascend);
} // namespace ops
} // namespace fxrt
