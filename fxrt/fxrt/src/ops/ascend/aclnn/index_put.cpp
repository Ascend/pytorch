#include <vector>

#include "ops/ascend/aclnn/index_put.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnIndexPut::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->IsTuple() ? input[1]->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>(),
      input[2]->ToTensor(),
      input[3]->ToBool(),
      input[4]->ToBool());
  return SUCCESS;
}

OpsErrorCode AclnnIndexPut::Launch(
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
      input[2]->ToTensor(),
      input[3]->ToBool(),
      input[4]->ToBool());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnIndexPut::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(index_put, AclnnIndexPut, Ascend);
} // namespace ops
} // namespace fxrt
