#include <vector>

#include "ops/ascend/aclnn/index_tensor.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnIndexTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->IsTuple() ? input[1]->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>(),
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnIndexTensor::Launch(
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
      input[1]->IsTuple() ? input[1]->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>(),
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(index_tensor, AclnnIndexTensor, Ascend);
FXRT_REG_OP_PROTOTYPE(index_tensor, 2);
} // namespace ops
} // namespace fxrt
