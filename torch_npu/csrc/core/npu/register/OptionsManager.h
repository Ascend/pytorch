#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <c10/util/Exception.h>

namespace c10_npu {
namespace option {

class OptionsManager {
public:
  static bool CheckInfNanModeEnable();
  static bool CheckQueueEnable();
  static bool CheckCombinedOptimizerEnable();
  static bool CheckTriCombinedOptimizerEnable();
  static bool CheckAclDumpDateEnable();
  static bool CheckDisableAclopComAndExe();
  static bool CheckSwitchMMOutputEnable();
  static uint32_t GetHCCLExecTimeout();
  static std::string CheckDisableDynamicPath();
  static int32_t GetACLExecTimeout();
private:
  static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
};

} // namespace option
} // namespace c10_npu