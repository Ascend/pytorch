#include <vector>

#include "ops/ascend/aclnn/split_tensor.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnSplitTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToInt(),
      output->IsTuple() ? output->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>());
  return SUCCESS;
}

OpsErrorCode AclnnSplitTensor::Launch(
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
      output->IsTuple() ? output->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>());
  return SUCCESS;
}

FXRT_REG_OP(split_tensor, AclnnSplitTensor, Ascend);
} // namespace ops
} // namespace fxrt
