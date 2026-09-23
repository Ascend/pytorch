#include <vector>

#include "ops/cpu/aten/aten_empty.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AtenEmpty::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

OpsErrorCode AtenEmpty::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

FXRT_REG_OP(empty, AtenEmpty, CPU);
FXRT_REG_OP(empty_like, AtenEmpty, CPU);
FXRT_REG_OP(new_empty, AtenEmpty, CPU);
} // namespace ops
} // namespace fxrt
