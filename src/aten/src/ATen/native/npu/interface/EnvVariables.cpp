

#include "c10/npu/register/OptionRegister.h"
#include "c10/util/Exception.h"
#include "ATen/native/npu/utils/NpuFuzzyBlacklist.h"
#include "ATen/native/npu/utils/NpuProfilingDispatch.h"
#include <third_party/acl/inc/acl/acl_mdl.h>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
namespace at {
namespace native {
namespace npu {
namespace env {

REGISTER_OPTION(autotune)
REGISTER_OPTION_BOOL_FUNCTION(AutoTuneEnabled, autotune, "disable", "enable")

REGISTER_OPTION_INIT_BY_ENV(bmmv2_enable)
REGISTER_OPTION_BOOL_FUNCTION(CheckBmmV2Enable, bmmv2_enable, "0", "1")

REGISTER_OPTION_HOOK(mdldumpswitch, [](const std::string& val) { 
  if (val == "enable") { aclmdlInitDump(); }
  else { aclmdlFinalizeDump(); }
  })
REGISTER_OPTION_HOOK(mdldumpconfigpath, [](const std::string& val) { aclmdlSetDump(val.c_str()); })

REGISTER_OPTION_HOOK(fuzzycompileswitch, [](const std::string& val) {
  if (val == "enable") { aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ); }
  else { aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT); }
 })
REGISTER_OPTION_BOOL_FUNCTION(CheckFuzzyEnable, fuzzycompileswitch, "disable", "enable")

REGISTER_OPTION_HOOK(ACL_OP_DEBUG_LEVEL, [](const std::string& val) { 
  aclSetCompileopt(aclCompileOpt::ACL_OP_DEBUG_LEVEL, val.c_str());
 })
REGISTER_OPTION_HOOK(ACL_DEBUG_DIR, [](const std::string& val) { 
  aclSetCompileopt(aclCompileOpt::ACL_DEBUG_DIR, val.c_str());
 })
REGISTER_OPTION_HOOK(ACL_OP_COMPILER_CACHE_MODE, [](const std::string& val) { 
  aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_MODE, val.c_str());
 })
REGISTER_OPTION_HOOK(ACL_OP_COMPILER_CACHE_DIR, [](const std::string& val) { 
  aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_DIR, val.c_str());
 })
REGISTER_OPTION_HOOK(NPU_FUZZY_COMPILE_BLACKLIST, [](const std::string& val) { 
  if (CheckFuzzyEnable()) {
    FuzzyCompileBlacklist::GetInstance().RegisterBlacklist(val);
  }
 })

 REGISTER_OPTION_INIT_BY_ENV(PROFILING_MODE)
 REGISTER_OPTION_BOOL_FUNCTION(CheckProfilingEnable, PROFILING_MODE, "false", "true");

 REGISTER_OPTION_HOOK(deliverswitch, [](const std::string& val) {
   TORCH_CHECK(
       CheckProfilingEnable(), 
       "before you prepare to deliver op, ",
       "you should be enture profiling mode is on correctly!");
   if (val == "enable"){
     at::native::npu::NpuProfilingDispatch::Instance().start();
   } else {
     at::native::npu::NpuProfilingDispatch::Instance().stop();
   }
 })

} // namespace env
} // namespace npu
} // namespace native
} // namespace at
