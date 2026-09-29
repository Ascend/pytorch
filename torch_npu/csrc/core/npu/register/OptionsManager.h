#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace npu_logging {
class Logger;
}

namespace c10_npu {
std::shared_ptr<npu_logging::Logger>& GetEnvLogger();

#define TORCH_NPU_ENV_LOGI(format, ...)                                        \
    do {                                                                       \
        TORCH_NPU_LOGI(c10_npu::GetEnvLogger(), format, ##__VA_ARGS__);        \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                    \
    } while (0);

namespace option {
enum ReuseMode {
    CLOSE = 0,
    ERASE_RECORD_STREAM = 1,
    AVOID_RECORD_STREAM = 2,
    ERASE_RECORD_STREAM_WITH_OPTIMIZE = 3,
};

enum SilenceCheckMode {
    CHECK_CLOSE = 0,
    PRINT_WARN_LOG = 1,
    REPORT_ALARM = 2,
    PRINT_ALL_LOG = 3,
};

// Sentinel value: user has not called set_task_queue_enable, fallback to env var
constexpr int32_t TASK_QUEUE_ENABLE_ENV = -1;
extern std::atomic<int32_t> g_task_queue_enable_mode;

class OptionsManager {
public:
    static bool IsHcclZeroCopyEnable();
    static bool IsResumeModeEnable();
    static bool IsCpuFallbackEnable();
    static bool IsSubCommRootInfoEnable();
    static bool IsScalableRootInfoEnable();
    static uint32_t GetHcclRanksPerRoot();
    static ReuseMode GetMultiStreamMemoryReuse();
    static bool CheckInfNanModeEnable();
    static bool CheckInfNanModeForceDisable();
    static bool CheckBlockingEnable();
    static bool CheckCombinedOptimizerEnable();
    static bool CheckTriCombinedOptimizerEnable();
    static bool CheckAclDumpDateEnable();
    static uint32_t GetHCCLConnectTimeout();
    static int32_t GetHCCLExecTimeout();
    static int32_t GetHCCLEventTimeout();
    static std::string CheckDisableDynamicPath();
    static int32_t GetACLExecTimeout();
    static int32_t GetACLDeviceSyncTimeout();
    static uint32_t CheckUseHcclAsyncErrorHandleEnable();
    static uint32_t CheckUseDesyncDebugEnable();
    C10_NPU_API static bool isACLGlobalLogOn(aclLogLevel level);
    C10_NPU_API static int64_t GetRankId();
    static char *GetNslbPath();
    static bool CheckStatusSaveEnable();
    static std::string GetStatusSavePath() noexcept;
    static uint32_t GetStatusSaveInterval();
    static uint32_t GetNslbCntVal();
    static bool CheckGeInitDisable();
    static bool CheckPerfDumpEnable();
    static std::string GetPerfDumpPath();
    static std::string GetRankTableFilePath();
    static uint32_t GetSilenceCheckFlag();
    static std::pair<double, double> GetSilenceUpperThresh();
    static std::pair<double, double> GetSilenceSigmaThresh();
    static uint32_t GetHcclBufferSize();
    static uint32_t GetP2PBufferSize();
    static uint32_t GetTaskQueueEnable();
    static void SetTaskQueueEnable(int32_t mode);
    static uint32_t GetPerStreamQueue();
    static uint32_t GetAclOpInitMode();
    static uint32_t GetStreamsPerDevice();
    static char* GetCpuAffinityConf();
    static bool CheckForceUncached();
    static std::string GetOomSnapshotDumpPath();
    static bool IsOomSnapshotEnable();
    static bool ShouldPrintWarning();
    static bool IsCompactErrorOutput();
    static uint64_t GetShmemSymmetricSize();
    static char *GetAclInitPath();

private:
    static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
    static std::unordered_map<std::string, std::string> ParsePerfConfig(const std::string& config);
    static std::vector<std::string> Split(const std::string& input, char delimiter);
    static std::pair<double, double> GetSilenceThresh(const std::string& env_str,
        std::pair<double, double> defaultThresh);
};

void oom_observer(int64_t device = 0, int64_t allocated = 0, int64_t device_total = 0, int64_t device_free = 0);
char* get_and_log_env(const char* env_str);

} // namespace option
} // namespace c10_npu
