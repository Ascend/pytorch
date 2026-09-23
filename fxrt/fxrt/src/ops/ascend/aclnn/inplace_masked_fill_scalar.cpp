#include <vector>

#include "ops/ascend/aclnn/inplace_masked_fill_scalar.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnInplaceMaskedFillScalar::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor(), input[2]);
  return SUCCESS;
}

OpsErrorCode AclnnInplaceMaskedFillScalar::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), input[2]);
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnInplaceMaskedFillScalar::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(inplace_masked_fill_scalar, AclnnInplaceMaskedFillScalar, Ascend);
} // namespace ops
} // namespace fxrt
