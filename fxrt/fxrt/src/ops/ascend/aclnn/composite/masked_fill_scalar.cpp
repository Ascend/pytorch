#include "ops/ascend/aclnn/composite/masked_fill_scalar.h"
#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnMaskedFillScalar::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executorCopy_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), output->ToTensor(), input[kIndex0]->ToTensor());
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), output->ToTensor(), input[kIndex1]->ToTensor(), input[kIndex2]);
  return SUCCESS;
}

OpsErrorCode AclnnMaskedFillScalar::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executorCopy_->Launch(workspace, workspaceSize, stream, output->ToTensor(), input[kIndex0]->ToTensor());
  executor_->Launch(workspace, workspaceSize, stream, output->ToTensor(), input[kIndex1]->ToTensor(), input[kIndex2]);
  return SUCCESS;
}

FXRT_REG_OP(masked_fill_scalar, AclnnMaskedFillScalar, Ascend);
} // namespace ops
} // namespace fxrt
