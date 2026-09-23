#include <vector>

#include "ops/ascend/aclnn/moe_init_routing_v3.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnMoeInitRoutingV3::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTensor(),
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->IsTensor() ? std::optional(input[3]->ToTensor()) : std::nullopt,
      input[4]->ToInt(),
      input[5]->ToInt(),
      input[6]->ToInt(),
      input[7]->ToInt(),
      input[8]->ToInt(),
      input[9]->ToBool(),
      input[10]->ToInt(),
      input[11]->ToTuple()->ToIntList(),
      input[12]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor(),
      (*output_tuple)[3]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeInitRoutingV3::Launch(
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
      input[3]->IsTensor() ? std::optional(input[3]->ToTensor()) : std::nullopt,
      input[4]->ToInt(),
      input[5]->ToInt(),
      input[6]->ToInt(),
      input[7]->ToInt(),
      input[8]->ToInt(),
      input[9]->ToBool(),
      input[10]->ToInt(),
      input[11]->ToTuple()->ToIntList(),
      input[12]->ToInt(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor(),
      (*output_tuple)[2]->ToTensor(),
      (*output_tuple)[3]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_init_routing_v3, AclnnMoeInitRoutingV3, Ascend);
} // namespace ops
} // namespace fxrt
