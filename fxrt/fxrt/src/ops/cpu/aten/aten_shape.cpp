#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"

#include "ops/cpu/aten/aten_shape.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AtenShape::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  // The output of Shape is a tuple, which does not have a shape in the tensor sense.
  // The tuple will be constructed in the InferShape method, only need input tensor shape information.
  // Here we just skip launch.

  return SUCCESS;
}

bool AtenShape::NeedLaunch() {
  return false;
}

FXRT_REG_OP(shape, AtenShape, CPU);
FXRT_REG_OP_PROTOTYPE(shape, 1);
} // namespace ops
} // namespace fxrt
