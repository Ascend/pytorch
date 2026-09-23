#include <vector>

#include "ops/ascend/aclnn/rms_norm.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnRmsNorm::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTensor(),
      input[2]->ToDouble(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnRmsNorm::Launch(
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
      input[2]->ToDouble(),
      (*output_tuple)[0]->ToTensor(),
      (*output_tuple)[1]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(rms_norm, AclnnRmsNorm, Ascend);
} // namespace ops
} // namespace fxrt
