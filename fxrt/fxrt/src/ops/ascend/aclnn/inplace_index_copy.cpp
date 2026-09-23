#include <vector>

#include "ops/ascend/aclnn/inplace_index_copy.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnInplaceIndexCopy::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToInt(),
      input[2]->ToTensor(),
      input[3]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnInplaceIndexCopy::Launch(
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
      input[2]->ToTensor(),
      input[3]->ToTensor());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnInplaceIndexCopy::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(inplace_index_copy, AclnnInplaceIndexCopy, Ascend);
} // namespace ops
} // namespace fxrt
