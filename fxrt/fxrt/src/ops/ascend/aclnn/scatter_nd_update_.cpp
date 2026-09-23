#include <vector>

#include "ops/ascend/aclnn/scatter_nd_update_.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnScatterNdUpdate::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize), input[0]->ToTensor(), input[1]->ToTensor(), input[2]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnScatterNdUpdate::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), input[2]->ToTensor());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnScatterNdUpdate::GetOutputInputRefPairs() const {
  return {{0, 0}};
}

FXRT_REG_OP(scatter_nd_update_, AclnnScatterNdUpdate, Ascend);
FXRT_REG_OP_PROTOTYPE(scatter_nd_update_, 3);
} // namespace ops
} // namespace fxrt
