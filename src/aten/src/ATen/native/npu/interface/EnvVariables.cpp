

#include "c10/npu/register/OptionRegister.h"

namespace at {
namespace native {
namespace npu {
namespace env {

REGISTER_OPTION(autotune)
REGISTER_OPTION_BOOL_FUNCTION(AutoTuneEnabled, autotune, "disable", "enable")

REGISTER_OPTION(ACL_OP_DEBUG_LEVEL)
REGISTER_OPTION(ACL_DEBUG_DIR)
REGISTER_OPTION(ACL_OP_COMPILER_CACHE_MODE)
REGISTER_OPTION(ACL_OP_COMPILER_CACHE_DIR)
REGISTER_OPTION(NPU_FUZZY_COMPILE_BLACKLIST)

} // namespace env
} // namespace npu
} // namespace native
} // namespace at
