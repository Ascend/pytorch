

#include "c10/npu/register/OptionRegister.h"
#include "c10/util/Exception.h"
#include <third_party/acl/inc/acl/acl_mdl.h>

namespace at {
namespace native {
namespace npu {
namespace env {

REGISTER_OPTION(autotune)
REGISTER_OPTION_BOOL_FUNCTION(AutoTuneEnabled, autotune, "disable", "enable")

REGISTER_OPTION_INIT_BY_ENV(bmmv2_enable)
REGISTER_OPTION_BOOL_FUNCTION(CheckBmmV2Enable, bmmv2_enable, "0", "1")

REGISTER_OPTION(ACL_OP_DEBUG_LEVEL)
REGISTER_OPTION(ACL_DEBUG_DIR)
REGISTER_OPTION(ACL_OP_COMPILER_CACHE_MODE)
REGISTER_OPTION(ACL_OP_COMPILER_CACHE_DIR)
REGISTER_OPTION(NPU_FUZZY_COMPILE_BLACKLIST)

REGISTER_OPTION_HOOK(mdldumpswitch, [](const std::string& val) { 
  if (val == "init") { aclmdlInitDump(); }
  else if (val == "finalize") { aclmdlFinalizeDump(); }
  else { TORCH_CHECK(0, "set initdump value only support init or finalize, but got ", val); }
  })
REGISTER_OPTION_HOOK(mdldumpconfigpath, [](const std::string& val) { aclmdlSetDump(val.c_str()); })
} // namespace env
} // namespace npu
} // namespace native
} // namespace at
