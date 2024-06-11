#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace option {

enum ReuseMode {
    CLOSE = 0,
    ERASE_RECORD_STREAM = 1,
    AVOID_RECORD_STREAM = 2,
};

class OptionsManager {
public:
    static bool IsResumeModeEnable();
    static ReuseMode GetMultiStreamMemoryReuse();
    static bool CheckInfNanModeEnable();
    static bool CheckBlockingEnable();
    static bool CheckQueueEnable();
    static bool CheckCombinedOptimizerEnable();
    static bool CheckTriCombinedOptimizerEnable();
    static bool CheckAclDumpDateEnable();
    static uint32_t GetHCCLExecTimeout();
    static uint32_t GetHCCLEventTimeout();
    static std::string CheckDisableDynamicPath();
    static int32_t GetACLExecTimeout();
    static uint32_t CheckUseHcclAsyncErrorHandleEnable();
    static uint32_t CheckUseDesyncDebugEnable();
    C10_NPU_API static bool isACLGlobalLogOn(aclLogLevel level);
    static int64_t GetRankId();
    static char *GetNslbPath();
    static uint32_t GetNslbCntVal();
    static bool CheckGeInitDisable();
    static bool CheckPerfDumpEnable();
    static std::string GetPerfDumpPath();
    static uint32_t GetP2PBufferSize();
    static uint32_t GetCpuAffinityConf();

private:
    static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
    static std::unordered_map<std::string, std::string> ParsePerfConfig(const std::string& config);
};

} // namespace option
} // namespace c10_npu
