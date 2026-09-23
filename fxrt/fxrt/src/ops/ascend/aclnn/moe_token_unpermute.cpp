#include <vector>

#include "ops/ascend/aclnn/moe_token_unpermute.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnMoeTokenUnpermute::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTensor(),
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->ToBool(),
      input[4]->IsTuple() ? std::optional(input[4]->ToTuple()->ToIntList()) : std::nullopt,
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeTokenUnpermute::Launch(
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
      input[1]->ToTensor(),
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->ToBool(),
      input[4]->IsTuple() ? std::optional(input[4]->ToTuple()->ToIntList()) : std::nullopt,
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_token_unpermute, AclnnMoeTokenUnpermute, Ascend);
} // namespace ops
} // namespace fxrt
