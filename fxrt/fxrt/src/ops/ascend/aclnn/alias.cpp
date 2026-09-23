#include <vector>

#include "ops/ascend/aclnn/alias.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
FXRT_REG_OP(alias, AclnnAlias, Ascend);
} // namespace ops
} // namespace fxrt
