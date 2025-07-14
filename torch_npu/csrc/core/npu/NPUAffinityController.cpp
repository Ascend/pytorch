#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/GetAffinityCPUInfo.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <string>
#include <unordered_map>
#include <mutex>

namespace c10_npu {

static thread_local ThreadType local_thread = ThreadType::MAIN_THREAD;

static pthread_t main_thread;
static bool start_main_thread_bind = false;
static std::mutex core_map_mutex;

using ThreadCoreMap = std::unordered_map<ThreadType, CoreIdRange>;

static uint32_t cpu_affinity_mode;
static std::vector<CoreIdRange> device_ranges;
static std::unordered_map<c10::DeviceIndex, ThreadCoreMap> device_thread_core_maps;

const std::initializer_list<ThreadType> threadTypeList = {
    MAIN_THREAD, ACL_THREAD, RELEASE_THREAD, WATCHDOG_THREAD, OTHER_THREAD};

const std::unordered_map<ThreadType, std::string> threadTypeToNameMap = {
    {MAIN_THREAD,       "main_thread"},
    {ACL_THREAD,        "acl_thread"},
    {RELEASE_THREAD,    "release_thread"},
    {WATCHDOG_THREAD,   "hccl_watchdog_t"},
    {OTHER_THREAD,      "other_thread"}};

CoreIdRange getCPUDefaultRange(c10::DeviceIndex device_id)
{
    static int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
    int device_nums = device_count_ensure_non_zero();
    int block_size = (core_nums > 0 && device_nums > 0) ? core_nums / device_nums : 0;
    return CoreIdRange{static_cast<CoreId>(device_id * block_size),
                       static_cast<CoreId>((device_id + 1) * block_size - 1)};
}

inline bool isAllDigits(const std::string &str)
{
    if (str.empty()) {
        return false;
    }
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

void parseCPUAffinityConf(uint32_t &mode, std::vector<CoreIdRange> &ranges)
{
    // init
    int device_nums = device_count_ensure_non_zero();
    ranges.clear();
    ranges.resize(device_nums);
    for (int i = 0; i < device_nums; ++i) {
        ranges[i] = getCPUDefaultRange(i);
    }
    mode = 0;

    const char *input = c10_npu::option::OptionsManager::GetCpuAffinityConf();
    if (input == nullptr || strlen(input) == 0) {
        return;
    }

    std::string inputStr(input);
    std::istringstream stream(inputStr);
    std::string option;

    std::regex pattern("npu_affine:(\\d)");
    std::smatch match;
    if (std::regex_search(inputStr, match, pattern)) {
        int isAffinity = std::stoi(match[1].str());
        if (isAffinity != 0) {
            for (int i = 0; i < device_nums; i++) {
                CoreIdRange getRange = GetAssignAffinityCPU(i);
                if (getRange.start == 0 && getRange.end == 0) {
                    break;
                }
                ranges[i] = getRange;
            }
        }
    }

    // Handle cases where only `mode` is provided, or `mode:` without value
    if (isAllDigits(inputStr)) {
        mode = static_cast<uint32_t>(std::stoi(inputStr));
        return; // Return directly, `mode` has already been processed
    }

    // Parse each option
    while (std::getline(stream, option, ',')) {
        // Split `option` based on colon
        size_t colonPos = option.find(':');
        if (colonPos != std::string::npos) {
            std::string key = option.substr(0, colonPos);
            std::string value = option.substr(colonPos + 1);

            // Process `mode`
            if (key == "mode") {
                if (isAllDigits(value)) {
                    mode = static_cast<uint32_t>(std::stoi(value));
                } else {
                    ASCEND_LOGW("mode is %s, should be all digits", value.c_str());
                }
            } else if (key.rfind("npu", 0) == 0) {
                // Handle NPU core binding range
                // The key is like 'npu:0', so skip first 3 chars.
                if (isAllDigits(key.substr(3))) {
                    int device_id = std::stoi(key.substr(3)); // Parse NPU device ID
                    if (device_id < device_nums) {
                        size_t dashPos = value.find('-');
                        if (dashPos != std::string::npos) {
                            std::string startStr = value.substr(0, dashPos);
                            std::string endStr = value.substr(dashPos + 1);
                            if (isAllDigits(startStr) && isAllDigits(endStr)) {
                                CoreId start = static_cast<CoreId>(std::stoi(startStr));
                                CoreId end = static_cast<CoreId>(std::stoi(endStr));
                                ranges[device_id] = {start, end};
                            } else {
                                ASCEND_LOGW("core range is %s-%s, should be all digits", startStr.c_str(), endStr.c_str());
                            }
                        } else {
                            if (isAllDigits(value)) {
                                CoreId singleCore = static_cast<CoreId>(std::stoi(value));
                                ranges[device_id] = {singleCore, singleCore};
                            } else {
                                ASCEND_LOGW("core range is string : %s, should be all digits", value.c_str());
                            }
                        }
                    }
                }
            }
        } else if (isAllDigits(option)) {
            // If no colon and the value is a number, use it directly as `mode`
            mode = static_cast<uint32_t>(std::stoi(option));
        }
    }
}

void printCoreRanges(const uint32_t mode, const std::vector<CoreIdRange> &ranges)
{
    std::ostringstream oss;
    oss << "Mode: " << mode << ". Core range for each device ID: ";

    for (size_t i = 0; i < ranges.size(); ++i) {
        oss << "Device " << i << ": [" << ranges[i].start << ", " << ranges[i].end << "]";
        if (i != ranges.size() - 1) {
            oss << "; ";
        } else {
            oss << ".";
        }
    }

    ASCEND_LOGD("Read CPU affinity config: %s", oss.str().c_str());
}

bool getThreadAffinityInfo()
{
    parseCPUAffinityConf(cpu_affinity_mode, device_ranges);
    printCoreRanges(cpu_affinity_mode, device_ranges);

    if (cpu_affinity_mode == 0) {
        return false;
    }

    cpu_set_t mask;
    pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
    for (auto &range : device_ranges) {
        for (unsigned int i = range.start; i < range.end; i++) {
            if (!CPU_ISSET(i, &mask)) {
                ASCEND_LOGW("Thread affinity is already set.");
                return false;
            }
        }
    }
    return true;
}

inline bool needToSetThreadAffinity()
{
    static bool need_to_set_affinity = getThreadAffinityInfo();
    return need_to_set_affinity;
}

void SetThreadType(ThreadType type)
{
    // Called at the start of the thread's execution to avoid frequent triggering of this function.
    local_thread = type;
    if (type == ThreadType::OTHER_THREAD || type == ThreadType::MAIN_THREAD) {
        return;
    }
    if (prctl(PR_SET_NAME, threadTypeToNameMap.at(type).c_str()) != 0) {
        ASCEND_LOGW("Set thread name to %s failed!", threadTypeToNameMap.at(type).c_str());
    }
}

std::string getAffinityMapAsString(c10::DeviceIndex device_id, const ThreadCoreMap &threadCoreMap)
{
    std::ostringstream oss;
    for (auto thread_type : threadTypeList) {
        oss << threadTypeToNameMap.at(thread_type) << ": ["
            << threadCoreMap.at(thread_type).start << ", "
            << threadCoreMap.at(thread_type).end << "]";
        if (thread_type != OTHER_THREAD) {
            oss << "; ";
        } else {
            oss << ".";
        }
    }
    return oss.str();
}

ThreadCoreMap getCpuAffinityMap(c10::DeviceIndex device_id, const std::vector<CoreIdRange> &device_ranges)
{
    ThreadCoreMap threadCoreMap;
    CoreIdRange range = device_ranges[device_id];
    unsigned int core_nums = range.end - range.start + 1;
    if (core_nums < threadTypeList.size()) {
        ASCEND_LOGW("Device %d available core numbers (%d) are insufficient for all %zu thread types and will bind available cores to all threads.",
                    device_id, core_nums, threadTypeList.size());
        for (auto thread_type : threadTypeList) {
            threadCoreMap[thread_type] = range;
        }
        return threadCoreMap;
    }

    CoreId now = range.start;
    for (auto thread_type : threadTypeList) {
        if (thread_type != ThreadType::OTHER_THREAD) {
            threadCoreMap[thread_type] = CoreIdRange{now, now};
        } else {
            threadCoreMap[ThreadType::OTHER_THREAD] = CoreIdRange{now, range.end};
        }
        now++;
    }

    ASCEND_LOGD("Device %d thread affinity map: %s", device_id, getAffinityMapAsString(device_id, threadCoreMap).c_str());
    return threadCoreMap;
}

bool setThreadAffinityImpl(pthread_t thread, CoreIdRange core_range)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto i = core_range.start; i <= core_range.end; i++) {
        CPU_SET(i, &mask);
    }
    if (!pthread_setaffinity_np(thread, sizeof(mask), &mask)) {
        return true;
    } else {
        return false;
    }
}

CoreIdRange getCoreRange(c10::DeviceIndex device_id, ThreadType type)
{
    CoreIdRange core_range;
    if (cpu_affinity_mode == 0 || cpu_affinity_mode == 1) {
        core_range = device_ranges[device_id];
    } else {
        std::lock_guard<std::mutex> lock(core_map_mutex);
        if (device_thread_core_maps.find(device_id) == device_thread_core_maps.end()) {
            device_thread_core_maps.emplace(device_id, getCpuAffinityMap(device_id, device_ranges));
        }
        core_range = device_thread_core_maps.at(device_id).at(type);
    }
    return core_range;
}

void SetThreadAffinity(c10::DeviceIndex device_id)
{
    if (!needToSetThreadAffinity() || local_thread == ThreadType::USER_THREAD) {
        return;
    }

    CoreIdRange core_range = getCoreRange(device_id, local_thread);
    if (setThreadAffinityImpl(pthread_self(), core_range)) {
        ASCEND_LOGD("Device %d set %s affinity to %d-%d success.",
                    device_id, threadTypeToNameMap.at(local_thread).c_str(), core_range.start, core_range.end);
    } else {
        ASCEND_LOGE("Device %d set %s affinity to %d-%d failed.",
                    device_id, threadTypeToNameMap.at(local_thread).c_str(), core_range.start, core_range.end);
    }
}

void SetThreadAffinity(ThreadType type)
{
    if (!needToSetThreadAffinity()) {
        return;
    }
    int device_index;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDevice(&device_index));
    c10::DeviceIndex device = static_cast<c10::DeviceIndex>(device_index);
    local_thread = type;
    if (local_thread == ThreadType::MAIN_THREAD) {
        start_main_thread_bind = true;
    }
    SetThreadAffinity(device);
}

void SetThreadAffinity(int core_start, int core_end)
{
    if (!needToSetThreadAffinity()) {
        return;
    }

    static int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
    CoreIdRange core_range;
    core_range.start = static_cast<CoreId>(std::min(core_start, core_nums));
    core_range.end = static_cast<CoreId>(std::min(core_end, core_nums));
    local_thread = ThreadType::USER_THREAD;

    if (setThreadAffinityImpl(pthread_self(), core_range)) {
        ASCEND_LOGD("Set thread affinity to user-defined range %d-%d success.", core_range.start, core_range.end);
    } else {
        ASCEND_LOGE("Set thread affinity to user-defined range %d-%d failed.", core_range.start, core_range.end);
    }
}

void SetMainThread()
{
    main_thread = pthread_self();
}

bool NeedMainThreadBind()
{
    return start_main_thread_bind && (local_thread == ThreadType::MAIN_THREAD);
}

void StartMainThreadBind(c10::DeviceIndex device_id)
{
    if (!needToSetThreadAffinity() || local_thread == ThreadType::USER_THREAD) {
        return;
    }

    static thread_local bool seted = false;
    if (!seted) {
        seted = true;
        if (syscall(SYS_gettid) != getpid()) {
            start_main_thread_bind = true;

            SetThreadAffinity(device_id);

            CoreIdRange core_range = getCoreRange(device_id, ThreadType::MAIN_THREAD);
            if (setThreadAffinityImpl(main_thread, core_range)) {
                ASCEND_LOGD("Device %d set %s affinity to %d-%d success.",
                            device_id, threadTypeToNameMap.at(ThreadType::MAIN_THREAD).c_str(),
                            core_range.start, core_range.end);
            } else {
                ASCEND_LOGE("Device %d set %s affinity to %d-%d failed.",
                            device_id, threadTypeToNameMap.at(ThreadType::MAIN_THREAD).c_str(),
                            core_range.start, core_range.end);
            }
        }
    }
}

} // namespace c10_npu