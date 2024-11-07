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

enum SilenceCheckMode {
    CHECK_CLOSE = 0,
    PRINT_WARN_LOG = 1,
    REPORT_ALARM = 2,
    PRINT_ALL_LOG = 3,
};
static std::unordered_map<int32_t, std::string> infNanMode = {{0, "max"}, {1, "inf_nan"}};
static std::unordered_map<int32_t, std::string> combinedEnableMode = {{0, "close"}, {1, "open"}};
static std::unordered_map<int32_t, std::string> launchBlockingMode = {{0, "disable"}, {1, "enable"}};
static std::unordered_map<int32_t, std::string> asyncErrorHandlingMode = {{0, "close"}, {1, "open"}};
static std::unordered_map<int32_t, std::string> desyncDebugMode = {{0, "close"}, {1, "open"}};
static std::unordered_map<int32_t, std::string> logLevelMode = {{0, "debug"}, {1, "info"}, {2, "warning"}, {3, "error"}, {4, "null"}};
static std::unordered_map<int32_t, std::string> memoryCacheMode = {{0, "open"}, {1, "close"}};
static std::unordered_map<int32_t, std::string> taskQueueEnableMode = {{0, "close"}, {1, "level 1"}, {2, "level 2"}};

class OptionsManager {
public:
    static bool IsResumeModeEnable();
    static ReuseMode GetMultiStreamMemoryReuse();
    static bool CheckInfNanModeEnable();
    static bool CheckBlockingEnable();
    static bool CheckCombinedOptimizerEnable();
    static bool CheckTriCombinedOptimizerEnable();
    static bool CheckAclDumpDateEnable();
    static uint32_t GetHCCLConnectTimeout();
    static uint32_t GetHCCLExecTimeout();
    static uint32_t GetHCCLEventTimeout();
    static std::string CheckDisableDynamicPath();
    static int32_t GetACLExecTimeout();
    static int32_t GetACLDeviceSyncTimeout();
    static uint32_t CheckUseHcclAsyncErrorHandleEnable();
    static uint32_t CheckUseDesyncDebugEnable();
    C10_NPU_API static bool isACLGlobalLogOn(aclLogLevel level);
    static int64_t GetRankId();
    static char *GetNslbPath();
    static uint32_t GetNslbCntVal();
    static bool CheckGeInitDisable();
    static bool CheckPerfDumpEnable();
    static std::string GetPerfDumpPath();
    static std::string GetRankTableFilePath();
    static uint32_t GetSilenceCheckFlag();
    static std::pair<double, double> GetSilenceUpperThresh();
    static std::pair<double, double> GetSilenceSigmaThresh();
    static uint32_t GetP2PBufferSize();
    static uint32_t GetTaskQueueEnable();
    static char* GetCpuAffinityConf();
    static bool CheckForceUncached();
    static std::string GetOomSnapshotDumpPath();
    static void IsOomSnapshotEnable();
    static bool ShouldPrintWarning();

private:
    static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
    static std::unordered_map<std::string, std::string> ParsePerfConfig(const std::string& config);
    static std::vector<std::string> Split(const std::string& input, char delimiter);
    static std::pair<double, double> GetSilenceThresh(const std::string& env_str,
        std::pair<double, double> defaultThresh);
};

} // namespace option
} // namespace c10_npu
