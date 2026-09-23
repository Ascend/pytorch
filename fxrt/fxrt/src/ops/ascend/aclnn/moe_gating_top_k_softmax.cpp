#include <vector>

#include "ops/ascend/aclnn/moe_gating_top_k_softmax.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnMoeGatingTopKSoftmax::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->IsTensor() ? std::optional(input[1]->ToTensor()) : std::nullopt,
      input[2]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeGatingTopKSoftmax::Launch(
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
      input[1]->IsTensor() ? std::optional(input[1]->ToTensor()) : std::nullopt,
      input[2]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_gating_top_k_softmax, AclnnMoeGatingTopKSoftmax, Ascend);
} // namespace ops
} // namespace fxrt
