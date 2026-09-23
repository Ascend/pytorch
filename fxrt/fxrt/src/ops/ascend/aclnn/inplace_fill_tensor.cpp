#include <vector>

#include "ops/ascend/aclnn/inplace_fill_tensor.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnInplaceFillTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnInplaceFillTensor::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnInplaceFillTensor::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(inplace_fill_tensor, AclnnInplaceFillTensor, Ascend);
FXRT_REG_OP_PROTOTYPE(inplace_fill_tensor, 2);
} // namespace ops
} // namespace fxrt
