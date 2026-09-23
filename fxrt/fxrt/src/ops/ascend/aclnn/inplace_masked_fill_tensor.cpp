#include <vector>

#include "ops/ascend/aclnn/inplace_masked_fill_tensor.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnInplaceMaskedFillTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor(), input[2]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnInplaceMaskedFillTensor::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), input[2]->ToTensor());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnInplaceMaskedFillTensor::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(inplace_masked_fill_tensor, AclnnInplaceMaskedFillTensor, Ascend);
FXRT_REG_OP_PROTOTYPE(inplace_masked_fill_tensor, 3);
} // namespace ops
} // namespace fxrt
