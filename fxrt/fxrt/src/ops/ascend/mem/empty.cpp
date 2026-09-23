#include <vector>

#include "ops/ascend/mem/empty.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode Empty::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

OpsErrorCode Empty::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

FXRT_REG_OP(empty, Empty, Ascend);
FXRT_REG_OP(empty_like, Empty, Ascend);
FXRT_REG_OP(new_empty, Empty, Ascend);
} // namespace ops
} // namespace fxrt
