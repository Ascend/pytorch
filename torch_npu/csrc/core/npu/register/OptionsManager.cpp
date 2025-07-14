#include <string>
#include <unistd.h>
#include <iomanip>

#ifndef BUILD_LIBTORCH
#include <Python.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>
#endif

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::IsHcclZeroCopyEnable()
{
    const static bool isHcclZeroCopyEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("TORCH_HCCL_ZERO_COPY", 0);
        std::unordered_map<int32_t, std::string> hcclZeroCopyMode = getHcclZeroCopyMode();
        if (hcclZeroCopyMode.find(enable) == hcclZeroCopyMode.end()) {
            TORCH_CHECK(false, "TORCH_HCCL_ZERO_COPY should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
        return enable != 0;
    }();
    return isHcclZeroCopyEnable;
}

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
                TORCH_CHECK(false, "MULTI_STREAM_MEMORY_REUSE should be 0, 1 or 2", PTA_ERROR(ErrCode::VALUE));
        }
        return mode;
    }();
    return reuseMode;
}

bool OptionsManager::CheckInfNanModeEnable()
{
    const static bool checkInfNanModeEnable = []() -> bool {
        int32_t enable = OptionsManager::GetBoolTypeOption("INF_NAN_MODE_ENABLE", 1);
        std::unordered_map<int32_t, std::string> infNanMode = getInfNanMode();
        if (infNanMode.find(enable) == infNanMode.end()) {
            TORCH_CHECK(false, "INF_NAN_MODE_ENABLE should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
        return enable != 0;
    }();
    return checkInfNanModeEnable;
}

bool OptionsManager::CheckInfNanModeForceDisable()
{
    const static bool checkInfNanModeForceDisable = []() -> bool {
        int32_t disable = OptionsManager::GetBoolTypeOption("INF_NAN_MODE_FORCE_DISABLE", 0);
        std::unordered_map<int32_t, std::string> disableInfNanMode = getDisableInfNanMode();
        if (disableInfNanMode.find(disable) == disableInfNanMode.end()) {
            TORCH_CHECK(false, "INF_NAN_MODE_FORCE_DISABLE should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
        return disable != 0;
    }();
    return checkInfNanModeForceDisable;
}

bool OptionsManager::CheckBlockingEnable()
{
    const static bool checkBlockingEnable = []() -> bool {
        int32_t blocking_enable = OptionsManager::GetBoolTypeOption("ASCEND_LAUNCH_BLOCKING", 0);
        std::unordered_map<int32_t, std::string> launchBlockingMode = getLaunchBlockingMode();
        if (launchBlockingMode.find(blocking_enable) == launchBlockingMode.end()) {
            TORCH_CHECK(false, "ASCEND_LAUNCH_BLOCKING should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
        return blocking_enable != 0;
    }();
    return checkBlockingEnable;
}

bool OptionsManager::CheckCombinedOptimizerEnable()
{
    const static bool checkCombinedOptimizerEnable = []() -> bool {
        int32_t combined_optimize = OptionsManager::GetBoolTypeOption("COMBINED_ENABLE");
        std::unordered_map<int32_t, std::string> combinedEnableMode = getCombinedEnableMode();
        if (combinedEnableMode.find(combined_optimize) == combinedEnableMode.end()) {
            TORCH_CHECK(false, "COMBINED_ENABLE should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
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

uint32_t OptionsManager::GetHCCLConnectTimeout()
{
    char* env_val = std::getenv("HCCL_CONNECT_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
    return static_cast<uint32_t>(envFlag);
}

int32_t OptionsManager::GetHCCLExecTimeout()
{
    char* env_val = std::getenv("HCCL_EXEC_TIMEOUT");
    int64_t envFlag;
    if (env_val != nullptr) {
        envFlag = strtol(env_val, nullptr, 10);
        if (envFlag < 0) {
            envFlag = -1;
            TORCH_NPU_WARN_ONCE("Get env HCCL_EXEC_TIMEOUT less than 0, so reset it to the default value.");
        }
    } else {
        envFlag = -1;
    }
    return static_cast<int32_t>(envFlag);
}

int32_t OptionsManager::GetHCCLEventTimeout()
{
    char* env_val = std::getenv("HCCL_EVENT_TIMEOUT");
    int64_t envFlag;
    if (env_val != nullptr) {
        envFlag = strtol(env_val, nullptr, 10);
        if (envFlag < 0) {
            envFlag = -1;
            TORCH_NPU_WARN_ONCE("Get env HCCL_EVENT_TIMEOUT less than 0, so reset it to the default value.");
        }
    } else {
        envFlag = -1;
    }
    return static_cast<int32_t>(envFlag);
}

int32_t OptionsManager::GetACLExecTimeout()
{
    char* env_val = std::getenv("ACL_STREAM_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
    return static_cast<int32_t>(envFlag);
}

int32_t OptionsManager::GetACLDeviceSyncTimeout()
{
    char* env_val = std::getenv("ACL_DEVICE_SYNC_TIMEOUT");
    int64_t timeout = -1;
    if (env_val != nullptr) {
        int64_t envFlag = strtol(env_val, nullptr, 10);
        TORCH_CHECK(envFlag > 0, "ACL_DEVICE_SYNC_TIMEOUT must be positive.", PTA_ERROR(ErrCode::VALUE));
        // convert s to ms
        timeout = envFlag * 1000;
    }
    return static_cast<int32_t>(timeout);
}

uint32_t OptionsManager::CheckUseHcclAsyncErrorHandleEnable()
{
    char* asyncErrorHandling_val = std::getenv("HCCL_ASYNC_ERROR_HANDLING");
    int64_t asyncErrorHandlingFlag =
        (asyncErrorHandling_val != nullptr) ? strtol(asyncErrorHandling_val, nullptr, 10) : 1;
    std::unordered_map<int32_t, std::string> asyncErrorHandlingMode = getAsyncErrorHandlingMode();
    if (asyncErrorHandlingMode.find(asyncErrorHandlingFlag) == asyncErrorHandlingMode.end()) {
        TORCH_CHECK(false, "HCCL_ASYNC_ERROR_HANDLING should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
    }
    return static_cast<uint32_t>(asyncErrorHandlingFlag);
}

uint32_t OptionsManager::CheckUseDesyncDebugEnable()
{
    char* desyncDebug_val = std::getenv("HCCL_DESYNC_DEBUG");
    int64_t desyncDebugFlag = (desyncDebug_val != nullptr) ? strtol(desyncDebug_val, nullptr, 10) : 0;
    std::unordered_map<int32_t, std::string> desyncDebugMode = getDesyncDebugMode();
    if (desyncDebugMode.find(desyncDebugFlag) == desyncDebugMode.end()) {
        TORCH_CHECK(false, "HCCL_DESYNC_DEBUG should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
    }
    return static_cast<uint32_t>(desyncDebugFlag);
}

bool OptionsManager::isACLGlobalLogOn(aclLogLevel level)
{
    const static int getACLGlobalLogLevel = []() -> int {
        char* env_val = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
        int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : ACL_ERROR;
        std::unordered_map<int32_t, std::string> logLevelMode = getLogLevelMode();
        if (logLevelMode.find(envFlag) == logLevelMode.end()) {
            TORCH_CHECK(false, "ASCEND_GLOBAL_LOG_LEVEL should be 0, 1, 2, 3 or 4.", PTA_ERROR(ErrCode::VALUE));
        }
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

bool OptionsManager::CheckStatusSaveEnable()
{
    const static bool CheckStatusSaveEnable = []() -> bool {
        int32_t status_save_enable = OptionsManager::GetBoolTypeOption("TORCH_HCCL_STATUS_SAVE_ENABLE");
        return status_save_enable != 0;
    }();
    return CheckStatusSaveEnable;
}

std::string OptionsManager::GetStatusSavePath() noexcept
{
    char* status_save_val = std::getenv("TORCH_HCCL_STATUS_SAVE_PATH");
    std::string status_save_path = (status_save_val != nullptr) ? std::string(status_save_val) : "/tmp";
    return status_save_path;
}

uint32_t OptionsManager::GetStatusSaveInterval()
{
    const static uint32_t status_save_interval = []() -> uint32_t {
        char* env_val = std::getenv("TORCH_HCCL_STATUS_SAVE_INTERVAL");
        int64_t envFlag = 2;
        if (env_val != nullptr) {
            envFlag = strtol(env_val, nullptr, 10);
            if (envFlag <= 0) {
                envFlag = 2;
                TORCH_NPU_WARN_ONCE("Get env TORCH_HCCL_STATUS_SAVE_INTERVAL less than or equal to 0, so reset it to the default value.");
            }
        }
        return static_cast<uint32_t>(envFlag);
    }();
    return status_save_interval;
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

std::string OptionsManager::GetRankTableFilePath()
{
    char* rank_table_file = std::getenv("RANK_TABLE_FILE");
    if (rank_table_file != nullptr) {
        return std::string(rank_table_file);
    } else {
        return "";
    }
}

uint32_t OptionsManager::GetSilenceCheckFlag()
{
    const static uint32_t silence_check_flag = []() -> uint32_t {
        char* silence_check_flag_str = std::getenv("NPU_ASD_ENABLE");
        int64_t silence_check_flag = (silence_check_flag_str != nullptr) ? strtol(silence_check_flag_str, nullptr, 10) : 0;
        SilenceCheckMode mode = CHECK_CLOSE;
        switch (silence_check_flag) {
            case 0:
                mode = CHECK_CLOSE;
                break;
            case 1:
                mode = PRINT_WARN_LOG;
                break;
            case 2:
                mode = REPORT_ALARM;
                break;
            case 3:
                mode = PRINT_ALL_LOG;
                break;
            default:
                TORCH_CHECK(false, "NPU_ASD_ENABLE should be 0, 1, 2 or 3", PTA_ERROR(ErrCode::VALUE));
        }
        return static_cast<uint32_t>(silence_check_flag);
    }();
    return silence_check_flag;
}

std::vector<std::string> OptionsManager::Split(const std::string& input, char delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = input.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(input.substr(start, end - start));
        start = end + 1;
        end = input.find(delimiter, start);
    }

    if (start < input.length()) {
        result.push_back(input.substr(start));
    }

    return result;
}

std::pair<double, double> OptionsManager::GetSilenceThresh(const std::string& env_str,
    std::pair<double, double> defaultThresh)
{
    char* upper_thresh_ptr = std::getenv(env_str.c_str());
    std::string upper_thresh_str = (upper_thresh_ptr != nullptr) ? std::string(upper_thresh_ptr) : "";
    std::vector<std::string> split_result = Split(upper_thresh_str, ',');
    if (split_result.size() != 2) {
        return defaultThresh;
    }
    try {
        double value1 = std::stod(split_result[0]);
        value1 = value1 >= 3 ? value1 : 3;
        double value2 = std::stod(split_result[1]);
        value2 = value2 >= 3 ? value2 : 3;
        if (value1 <= value2) {
            return defaultThresh;
        }
        return std::make_pair(value1, value2);
    } catch (std::exception& e) {
        TORCH_NPU_WARN("Invalid value for environment variable: ", env_str, ". Use default values");
    }
    return defaultThresh;
}

std::pair<double, double> OptionsManager::GetSilenceUpperThresh()
{
    const static std::pair<double, double> upper_thresh = []() -> std::pair<double, double> {
        return GetSilenceThresh("NPU_ASD_UPPER_THRESH", std::make_pair(1000000.0, 10000.0));
    }();
    return upper_thresh;
}

std::pair<double, double> OptionsManager::GetSilenceSigmaThresh()
{
    const static std::pair<double, double> sigma_thresh = []() -> std::pair<double, double> {
        return GetSilenceThresh("NPU_ASD_SIGMA_THRESH", std::make_pair(100000.0, 5000.0));
    }();
    return sigma_thresh;
}

uint32_t OptionsManager::GetHcclBufferSize()
{
    const static uint32_t hccl_buf_size = []() -> uint32_t {
        char* buf_val = std::getenv("HCCL_BUFFSIZE");
        // Default 200M
        int64_t buf_size = (buf_val != nullptr) ? strtol(buf_val, nullptr, 10) : 200;
        TORCH_CHECK(buf_size > 0, "HCCL_BUFFSIZE should be positive.", PTA_ERROR(ErrCode::VALUE));
        return static_cast<uint32_t>(buf_size);
    }();
    return hccl_buf_size;
}

uint32_t OptionsManager::GetP2PBufferSize()
{
    const static uint32_t buf_size = []() -> uint32_t {
        char* buf_val = std::getenv("P2P_HCCL_BUFFSIZE");
        // Default 0M
        int64_t buf_size = (buf_val != nullptr) ? strtol(buf_val, nullptr, 10) : 20;
        TORCH_CHECK(buf_size >= 0, "P2P_HCCL_BUFFSIZE cannot be negative.", PTA_ERROR(ErrCode::VALUE));
        return static_cast<uint32_t>(buf_size);
    }();
    return buf_size;
}

uint32_t OptionsManager::GetAclOpInitMode()
{
    const static uint32_t acl_op_init_mode = []() -> uint32_t {
        char* buf_val = std::getenv("ACL_OP_INIT_MODE");
        // Default 0
        int64_t acl_op_init_mode = (buf_val != nullptr) ? strtol(buf_val, nullptr, 10) : 0;
        std::unordered_map<int32_t, std::string> aclOpInitMode = getAclOpInitMode();
        if (aclOpInitMode.find(acl_op_init_mode) == aclOpInitMode.end()) {
            TORCH_NPU_WARN_ONCE("Get env ACL_OP_INIT_MODE not in [0, 1, 2], so reset it to the default value 0.");
        }
        return static_cast<uint32_t>(acl_op_init_mode);
    }();
    return acl_op_init_mode;
}

uint32_t OptionsManager::GetStreamsPerDevice()
{
    const static uint32_t streams_per_device = []() -> uint32_t {
        char* buf_val = std::getenv("STREAMS_PER_DEVICE");
        // Default 8
        int64_t streams_per_device = (buf_val != nullptr) ? strtol(buf_val, nullptr, 10) : 8;
        if (streams_per_device != 8 && streams_per_device != 32) {
            streams_per_device = 8;
            TORCH_NPU_WARN_ONCE("STREAMS_PER_DEVICE only support 8 or 32, but get other value, so reset it to the default value 8");
        }
        return static_cast<uint32_t>(streams_per_device);
    }();
    return streams_per_device;
}

char* OptionsManager::GetCpuAffinityConf()
{
    return std::getenv("CPU_AFFINITY_CONF");
}

uint32_t OptionsManager::GetTaskQueueEnable()
{
    if (CheckBlockingEnable()) {
        return 0;
    }
    const static uint32_t task_queue_enable = []() -> uint32_t {
        char* env_val = std::getenv("TASK_QUEUE_ENABLE");
        int64_t task_queue_enable = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 1;
        std::unordered_map<int32_t, std::string> taskQueueEnableMode = getTaskQueueEnableMode();
        if (taskQueueEnableMode.find(task_queue_enable) == taskQueueEnableMode.end()) {
            TORCH_CHECK(false, "TASK_QUEUE_ENABLE should be 0, 1 or 2", PTA_ERROR(ErrCode::VALUE));
        }
        return static_cast<uint32_t>(task_queue_enable);
    }();
    return task_queue_enable;
}

bool OptionsManager::CheckForceUncached()
{
    const static bool force_uncached = []() -> bool {
        bool force_uncached = OptionsManager::GetBoolTypeOption("PYTORCH_NO_NPU_MEMORY_CACHING");
        std::unordered_map<int32_t, std::string> memoryCacheMode = getMemoryCacheMode();
        if (memoryCacheMode.find(force_uncached) == memoryCacheMode.end()) {
            TORCH_CHECK(false, "PYTORCH_NO_NPU_MEMORY_CACHING should be 0 or 1.", PTA_ERROR(ErrCode::VALUE));
        }
        return force_uncached;
    }();
    return force_uncached;
}

std::string OptionsManager::GetOomSnapshotDumpPath()
{
    char* sanpshot_dump_path = std::getenv("OOM_SNAPSHOT_PATH");
    std::string dump_path = "./";
    if (sanpshot_dump_path != nullptr) {
        dump_path = std::string(sanpshot_dump_path);
    }
    char dump_abs_path[PATH_MAX] = {'\0'};
    if (realpath(dump_path.c_str(), dump_abs_path) == nullptr) {
        TORCH_CHECK(false, "`OOM_SNAPSHOT_PATH` is invalid.", PTA_ERROR(ErrCode::NOT_FOUND));
    }
    return dump_abs_path;
}

bool OptionsManager::ShouldPrintWarning()
{
    static bool should_print = []() {
        char* disabled_warning = std::getenv("TORCH_NPU_DISABLED_WARNING");
        if (disabled_warning != nullptr && strtol(disabled_warning, nullptr, 10) == 1) {
            return false;
        }
        const auto rank_id = c10_npu::option::OptionsManager::GetRankId();
        return rank_id == 0 || rank_id == -1;
    }();
    return should_print;
}

void oom_observer(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free)
{
#ifndef BUILD_LIBTORCH
    auto dumppath = c10_npu::option::OptionsManager::GetOomSnapshotDumpPath();
    std::stringstream filename;
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);
    filename << "oom_snapshot_" << getpid() << "_" << std::put_time(timeInfo, "%Y%m%d%H%M%S") << ".pickle";
    auto savefilepath = c10::str(dumppath, "/", filename.str());

    pybind11::gil_scoped_acquire g;
    auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu.memory"));
    if (!module) {
        throw python_error();
    }
    PyObject* p_func = PyObject_GetAttrString(module, "_dump_snapshot");
    if (!module) {
        throw python_error();
    }
    PyObject* p_args = PyTuple_New(1);
    PyTuple_SetItem(p_args, 0, PyUnicode_FromString(savefilepath.c_str()));
    PyObject* p_res = PyObject_CallObject(p_func, p_args);
#endif
}


bool OptionsManager::IsOomSnapshotEnable()
{
    static bool isFirstCall = true;
    const static char *env_val = std::getenv("OOM_SNAPSHOT_ENABLE");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
#ifndef BUILD_LIBTORCH
    if (isFirstCall) {
        switch (envFlag) {
            case 0:
                break;
            case 2:
                c10_npu::NPUCachingAllocator::attachOutOfMemoryObserver(std::move(oom_observer));
                torch_npu::_record_memory_history("state", "all", "python", UINT64_MAX);
                isFirstCall = false;
                break;
            default:
                c10_npu::NPUCachingAllocator::attachOutOfMemoryObserver(std::move(oom_observer));
                torch_npu::_record_memory_history("all", "all", "python", UINT64_MAX);
                isFirstCall = false;
                break;
        }
    }
#endif
    return (envFlag != 0);
}

bool OptionsManager::IsCompactErrorOutput()
{
    static bool should_print = []() -> bool {
        int32_t disabled_error = OptionsManager::GetBoolTypeOption("TORCH_NPU_COMPACT_ERROR_OUTPUT");
        return disabled_error != 0;
    }();
    return should_print;
}

} // namespace option
} // namespace c10_npu
