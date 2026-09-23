#include <vector>

#include "ops/ascend/aclnn/topk.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnTopk::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToInt(),
      input[3]->ToBool(),
      input[4]->ToBool(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnTopk::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto& output_tuple = output->ToTuple();

  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToInt(),
      input[3]->ToBool(),
      input[4]->ToBool(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(topk, AclnnTopk, Ascend);
} // namespace ops
} // namespace fxrt
