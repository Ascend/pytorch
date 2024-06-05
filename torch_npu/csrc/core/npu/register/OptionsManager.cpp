#include <string>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::IsResumeModeEnable()
{
    const static bool isResumeModeEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("RESUME_MODE_ENABLE", 0);
        return enable != 0;
    }();
    return isResumeModeEnable;
}

ReuseMode OptionsManager::GetMultiStreamMemoryReuse()
{
    const static ReuseMode reuseMode = []() -> ReuseMode {
        char *env_val = std::getenv("MULTI_STREAM_MEMORY_REUSE");
        int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 1;
        ReuseMode mode = ERASE_RECORD_STREAM;
        switch (envFlag) {
            case 0:
                mode = CLOSE;
                break;
            case 1:
                mode = ERASE_RECORD_STREAM;
                break;
            case 2:
                mode = AVOID_RECORD_STREAM;
                break;
            default:
                TORCH_CHECK(false, "MULTI_STREAM_MEMORY_REUSE only allow 0, 1, 2", PTA_ERROR(ErrCode::VALUE));
        }
        return mode;
    }();
    return reuseMode;
}

bool OptionsManager::CheckInfNanModeEnable()
{
    const static bool checkInfNanModeEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("INF_NAN_MODE_ENABLE", 1);
        return enable != 0;
    }();
    return checkInfNanModeEnable;
}

bool OptionsManager::CheckBlockingEnable()
{
    const static bool checkBlockingEnable = []() -> bool {
        int32_t blocking_enable = OptionsManager::GetBoolTypeOption("ASCEND_LAUNCH_BLOCKING", 0);
        return blocking_enable != 0;
    }();
    return checkBlockingEnable;
}

bool OptionsManager::CheckQueueEnable()
{
    if (CheckBlockingEnable()) {
        return false;
    }
    const static bool checkQueueEnable = []() -> bool {
        int32_t queue_enable = OptionsManager::GetBoolTypeOption("TASK_QUEUE_ENABLE", 1);
        return queue_enable != 0;
    }();
    return checkQueueEnable;
}

bool OptionsManager::CheckCombinedOptimizerEnable()
{
    const static bool checkCombinedOptimizerEnable = []() -> bool {
        int32_t combined_optimize = OptionsManager::GetBoolTypeOption("COMBINED_ENABLE");
        return combined_optimize != 0;
    }();
    return checkCombinedOptimizerEnable;
}

bool OptionsManager::CheckAclDumpDateEnable()
{
    const static bool checkAclDumpDateEnable = []() -> bool {
        int32_t acl_dump_data = OptionsManager::GetBoolTypeOption("ACL_DUMP_DATA");
        return acl_dump_data != 0;
    }();
    if (checkAclDumpDateEnable) {
        TORCH_NPU_WARN_ONCE(
            "The environment variable ACL_DUMP_DATA has been deprecated, "
            "please use torch_npu.npu.init_dump() instead");
    }
    return checkAclDumpDateEnable;
}

int OptionsManager::GetBoolTypeOption(const char* env_str, int defaultVal)
{
    char* env_val = std::getenv(env_str);
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : defaultVal;
    return (envFlag != 0) ? 1 : 0;
}

uint32_t OptionsManager::GetHCCLExecTimeout()
{
    char* env_val = std::getenv("HCCL_EXEC_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(envFlag);
}

uint32_t OptionsManager::GetHCCLEventTimeout()
{
    char* env_val = std::getenv("HCCL_EVENT_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(envFlag);
}

int32_t OptionsManager::GetACLExecTimeout()
{
    char* env_val = std::getenv("ACL_STREAM_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
    return static_cast<int32_t>(envFlag);
}

uint32_t OptionsManager::CheckUseHcclAsyncErrorHandleEnable()
{
    char* asyncErrorHandling_val = std::getenv("HCCL_ASYNC_ERROR_HANDLING");
    int64_t asyncErrorHandlingFlag =
        (asyncErrorHandling_val != nullptr) ? strtol(asyncErrorHandling_val, nullptr, 10) : 1;
    return static_cast<uint32_t>(asyncErrorHandlingFlag);
}

uint32_t OptionsManager::CheckUseDesyncDebugEnable()
{
    char* desyncDebug_val = std::getenv("HCCL_DESYNC_DEBUG");
    int64_t desyncDebugFlag = (desyncDebug_val != nullptr) ? strtol(desyncDebug_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(desyncDebugFlag);
}

bool OptionsManager::isACLGlobalLogOn(aclLogLevel level)
{
    const static int getACLGlobalLogLevel = []() -> int {
        char* env_val = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
        int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : ACL_ERROR;
        return static_cast<int>(envFlag);
    }();
    return (getACLGlobalLogLevel <= level);
}

int64_t OptionsManager::GetRankId()
{
    char* rankId_val = std::getenv("RANK");
    int64_t rankId = (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
    return rankId;
}

char *OptionsManager::GetNslbPath()
{
    return std::getenv("NSLB_CP");
}

uint32_t OptionsManager::GetNslbCntVal()
{
    const static uint32_t nslb_val = []() -> uint32_t {
        char* nslb_num = std::getenv("NSLB_MAX_RECORD_NUM");
        int64_t nslb_val = (nslb_num != nullptr) ? strtol(nslb_num, nullptr, 10) : 1000;
        return static_cast<uint32_t>(nslb_val);
    }();
    return nslb_val;
}

bool OptionsManager::CheckGeInitDisable()
{
    const static bool Check_Ge_Init_Disable = []() -> bool {
        int32_t ge_init_disable = OptionsManager::GetBoolTypeOption("GE_INIT_DISABLE");
        return ge_init_disable != 0;
    }();
    if (Check_Ge_Init_Disable) {
        TORCH_NPU_WARN_ONCE(
            "The environment variable GE_INIT_DISABLE has been enabled, "
            "this switch is only used for single operator simulation");
    }
    return Check_Ge_Init_Disable;
}

std::unordered_map<std::string, std::string> OptionsManager::ParsePerfConfig(const std::string& config)
{
    std::unordered_map<std::string, std::string> config_map;
    size_t start = 0;
    size_t end = config.find(',');

    while (end != std::string::npos) {
        std::string item = config.substr(start, end - start);
        size_t delimiter_pos = item.find(':');
        if (delimiter_pos != std::string::npos) {
            std::string key = item.substr(0, delimiter_pos);
            std::string value = item.substr(delimiter_pos + 1);
            config_map[key] = value;
        }
        start = end + 1;
        end = config.find(',', start);
    }

    // Handle the last item
    std::string last_item = config.substr(start);
    size_t delimiter_pos = last_item.find(':');
    if (delimiter_pos != std::string::npos) {
        std::string key = last_item.substr(0, delimiter_pos);
        std::string value = last_item.substr(delimiter_pos + 1);
        config_map[key] = value;
    }

    return config_map;
}

bool OptionsManager::CheckPerfDumpEnable()
{
    char* perf_dump_config = std::getenv("PERF_DUMP_CONFIG");
    if (perf_dump_config != nullptr) {
        std::unordered_map<std::string, std::string> config_dict = ParsePerfConfig(perf_dump_config);
        auto it = config_dict.find("enable");
        if (it != config_dict.end()) {
            return it->second == "true";
        }
    }
    return false;
}

std::string OptionsManager::GetPerfDumpPath()
{
    char* perf_dump_path = std::getenv("PERF_DUMP_PATH");
    if (perf_dump_path != nullptr) {
        return std::string(perf_dump_path);
    } else {
        return "";
    }
}

uint32_t OptionsManager::GetP2PBufferSize()
{
    const static uint32_t buf_size = []() -> uint32_t {
        char* buf_val = std::getenv("P2P_HCCL_BUFFSIZE");
        // Default 0M
        int64_t buf_size = (buf_val != nullptr) ? strtol(buf_val, nullptr, 10) : 0;
        return static_cast<uint32_t>(buf_size);
    }();
    return buf_size;
}

uint32_t OptionsManager::GetBindCpuConf()
{
    const static uint32_t bind_cpu_conf = []() -> uint32_t {
        char* bind_core_str = std::getenv("BIND_CPU_CONF");
        int64_t bind_cpu_conf = (bind_core_str != nullptr) ? strtol(bind_core_str, nullptr, 10) : 0;
        return static_cast<uint32_t>(bind_cpu_conf);
    }();
    return bind_cpu_conf;
}

} // namespace option
} // namespace c10_npu
