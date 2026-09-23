#include <vector>

#include "ops/ascend/aclnn/apply_rotary_pos_emb.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnApplyRotaryPosEmb::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTensor(),
      input[2]->ToTensor(),
      input[3]->ToTensor(),
      input[4]->ToInt());
  return SUCCESS;
}

OpsErrorCode AclnnApplyRotaryPosEmb::Launch(
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
      input[1]->ToTensor(),
      input[2]->ToTensor(),
      input[3]->ToTensor(),
      input[4]->ToInt());
  return SUCCESS;
}
std::vector<std::pair<uint32_t, uint32_t>> AclnnApplyRotaryPosEmb::GetOutputInputRefPairs() const {
  return {{0, 0}, {1, 1}};
}

FXRT_REG_OP(apply_rotary_pos_emb, AclnnApplyRotaryPosEmb, Ascend);
} // namespace ops
} // namespace fxrt
