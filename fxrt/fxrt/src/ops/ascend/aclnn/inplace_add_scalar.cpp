#include <vector>

#include "ops/ascend/aclnn/inplace_add_scalar.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnInplaceAddScalar::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1], input[2]);
  return SUCCESS;
}

OpsErrorCode AclnnInplaceAddScalar::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1], input[2]);
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnInplaceAddScalar::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(inplace_add_scalar, AclnnInplaceAddScalar, Ascend);
} // namespace ops
} // namespace fxrt
