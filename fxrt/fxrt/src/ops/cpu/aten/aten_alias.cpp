#include <vector>

#include "ops/cpu/aten/aten_alias.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
FXRT_REG_OP(alias, AtenAlias, CPU);
} // namespace ops
} // namespace fxrt
