#include <vector>

#include "ops/ascend/aclnn/dequant_swiglu_quant.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnDequantSwigluQuant::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->IsTensor() ? std::optional(input[1]->ToTensor()) : std::nullopt,
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->IsTensor() ? std::optional(input[3]->ToTensor()) : std::nullopt,
      input[4]->IsTensor() ? std::optional(input[4]->ToTensor()) : std::nullopt,
      input[5]->IsTensor() ? std::optional(input[5]->ToTensor()) : std::nullopt,
      input[6]->IsTensor() ? std::optional(input[6]->ToTensor()) : std::nullopt,
      input[7]->ToBool(),
      input[8]->ToString(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnDequantSwigluQuant::Launch(
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
      input[2]->IsTensor() ? std::optional(input[2]->ToTensor()) : std::nullopt,
      input[3]->IsTensor() ? std::optional(input[3]->ToTensor()) : std::nullopt,
      input[4]->IsTensor() ? std::optional(input[4]->ToTensor()) : std::nullopt,
      input[5]->IsTensor() ? std::optional(input[5]->ToTensor()) : std::nullopt,
      input[6]->IsTensor() ? std::optional(input[6]->ToTensor()) : std::nullopt,
      input[7]->ToBool(),
      input[8]->ToString(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(dequant_swiglu_quant, AclnnDequantSwigluQuant, Ascend);
} // namespace ops
} // namespace fxrt
