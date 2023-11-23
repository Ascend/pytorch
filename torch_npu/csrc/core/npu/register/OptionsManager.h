#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace option {

class OptionsManager {
public:
    static bool IsResumeModeEnable();
    static bool IsMultiStreamMemoryReuse();
    static bool CheckInfNanModeEnable();
    static bool CheckBlockingEnable();
    static bool CheckQueueEnable();
    static bool CheckCombinedOptimizerEnable();
    static bool CheckTriCombinedOptimizerEnable();
    static bool CheckAclDumpDateEnable();
    static uint32_t GetHCCLExecTimeout();
    static std::string CheckDisableDynamicPath();
    static int32_t GetACLExecTimeout();
    static const char* GetAclConfigJsonPath();
    static uint32_t CheckUseHcclAsyncErrorHandleEnable();
    static uint32_t CheckUseDesyncDebugEnable();
    C10_NPU_API static bool isACLGlobalLogOn(aclLogLevel level);

private:
    static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
};

} // namespace option
} // namespace c10_npu