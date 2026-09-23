#include <vector>

#include "ops/ascend/aclnn/moe_re_routing.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnMoeReRouting::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTensor(),
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->ToInt(),
      input[4]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor(),
      (*output_tuple)[3]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeReRouting::Launch(
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
      input[1]->ToTensor(),
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->ToInt(),
      input[4]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor(),
      (*output_tuple)[3]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_re_routing, AclnnMoeReRouting, Ascend);
} // namespace ops
} // namespace fxrt
