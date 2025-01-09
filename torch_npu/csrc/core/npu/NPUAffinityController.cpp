
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <cstdio>
#include <sys/prctl.h>
#include <string>
#include <unordered_map>
#include <cctype>
#include <algorithm>

namespace c10_npu {

    static pthread_t mainthread_tid;
    static bool has_set_affinity = false;

    const std::unordered_map<ThreadType, std::string> threadTypeToNameMap = {
        {releaseThread, "release_thread"},
        {aclThread, "acl_thread"},
        {mainThread, "main_thread"},
        {hcclCommWatchdogThread, "hcclComm_watchd"}, // thread name no more than 15 chars
        {backwardThread, "backward_thread"}};

    const std::unordered_map<std::string, ThreadType> threadNameToTypeMap = {
        {"release_thread", releaseThread},
        {"acl_thread", aclThread},
        {"main_thread", mainThread},
        {"hcclComm_watchd", hcclCommWatchdogThread},
        {"backward_thread", backwardThread}};

    inline bool has_set_pthread_affinity()
    {
        unsigned int core_nums = static_cast<unsigned int>(sysconf(_SC_NPROCESSORS_ONLN));

        cpu_set_t mask;
        pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
        for (unsigned int i = 0; i < core_nums; i++) {
            if (!CPU_ISSET(i, &mask)) {
                return true;
            }
        }
        return false;
    }

    void GetAffinityInfo()
    {
        mainthread_tid = pthread_self();
        has_set_affinity = has_set_pthread_affinity();
    }

    ThreadType getCurrentThreadType()
    {
        char thread_name[16];

        if (prctl(PR_GET_NAME, thread_name, 0, 0, 0) == 0) {
            std::string name(thread_name);

            auto it = threadNameToTypeMap.find(name);
            if (it != threadNameToTypeMap.end()) {
                return it->second;
            }
        }
        return ThreadType::unknownThread;
    }

    aclError SetThreadAffinity(coreIdRange core_range, pthread_t thread)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);

        for (auto i = core_range.start; i <= core_range.end; i++) {
            CPU_SET(i, &mask);
        }
        if (!pthread_setaffinity_np(thread, sizeof(mask), &mask)) {
            ASCEND_LOGD("Set Thread Affinity to %d-%d", core_range.start, core_range.end);
            return ACL_ERROR_NONE;
        }
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    coreIdRange GetCPUDefaultRange(c10::DeviceIndex device_id)
    {
        int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
        int device_nums = device_count_ensure_non_zero();
        int block_size = (core_nums > 0 && device_nums > 0) ? (core_nums + device_nums - 1) / device_nums : 0;
        return coreIdRange{static_cast<unsigned int>(device_id * block_size),
                           static_cast<coreId>(std::min((device_id + 1) * block_size, core_nums) - 1)};
    }


    std::string GetAffinityMapAsString(const std::unordered_map<ThreadType, coreIdRange> &threadToCoreidMap, c10::DeviceIndex device_id)
    {
        std::ostringstream oss;
        oss << "threadToCoreidMap plan to bind device " << static_cast<unsigned int>(device_id) << " to "
            << " [" << threadToCoreidMap.at(unknownThread).start << "," << threadToCoreidMap.at(unknownThread).end << "]、"
            << " [" << threadToCoreidMap.at(mainThread).start << "," << threadToCoreidMap.at(mainThread).end << "]、"
            << " [" << threadToCoreidMap.at(backwardThread).start << "," << threadToCoreidMap.at(backwardThread).end << "]、"
            << " [" << threadToCoreidMap.at(aclThread).start << "," << threadToCoreidMap.at(aclThread).end << "]、"
            << " [" << threadToCoreidMap.at(releaseThread).start << "," << threadToCoreidMap.at(releaseThread).end << "]、"
            << " [" << threadToCoreidMap.at(hcclCommWatchdogThread).start << "," << threadToCoreidMap.at(hcclCommWatchdogThread).end << "]";

        return oss.str();
    }

    std::unordered_map<ThreadType, coreIdRange> GetCpuAffinityMap(c10::DeviceIndex device_id)
    {
        std::unordered_map<ThreadType, coreIdRange> threadToCoreidMap;
        std::initializer_list<ThreadType> thread_types = {unknownThread, mainThread, backwardThread, aclThread,
                                                          releaseThread, hcclCommWatchdogThread};

        coreIdRange current_core_range = GetCPUDefaultRange(device_id);
        coreId offset = current_core_range.start;

        // calculate env2 default map
        coreId core_nums = current_core_range.end - current_core_range.start;
        if (core_nums < thread_types.size()) {
            ASCEND_LOGW("Available core numbers (%d) are insufficient for all %zu thread types. Binding available cores to all threads.",
                        core_nums, thread_types.size());
            for (auto thread_type : thread_types) {
                threadToCoreidMap[thread_type] = current_core_range;
            }
        } else {
            int remaining_type_count = thread_types.size() - 1;
            int i = 0;
            for (auto thread_type : thread_types) {
                if (thread_type == ThreadType::unknownThread) {
                    threadToCoreidMap[ThreadType::unknownThread] = coreIdRange{current_core_range.start + remaining_type_count, current_core_range.end};
                } else {
                    threadToCoreidMap[thread_type] = coreIdRange{offset + i, offset + (i++)};
                }
            }
        }

        ASCEND_LOGD("Thread affinity map for device %d: %s", device_id, GetAffinityMapAsString(threadToCoreidMap, device_id).c_str());

        return threadToCoreidMap;
    }

    aclError SetThreadAffinity(c10::DeviceIndex device_id)
    {
        return SetThreadAffinity(device_id, getCurrentThreadType());
    }

    void printCoreRanges(const std::vector<coreIdRange> &ranges, uint32_t mode)
    {
        std::ostringstream oss;
        oss << "Mode: " << mode << " ";

        for (size_t i = 0; i < ranges.size(); ++i) {
            oss << "Device " << i << " Core Range: " << ranges[i].start << " - " << ranges[i].end << " ";
        }

        ASCEND_LOGD("Core ranges: %s", oss.str().c_str());
    }

    bool isAllDigits(const std::string &str)
    {
        if (str.empty()) {
            return false;
        }
        return std::all_of(str.begin(), str.end(), [](unsigned char c) {
            return std::isdigit(c);
        });
    }

    void parseCPUAffinityConf(uint32_t &mode, std::vector<coreIdRange> &ranges)
    {
        const char *input = c10_npu::option::OptionsManager::GetCpuAffinityConf();

        if (input == nullptr || strlen(input) == 0) {
            mode = 0;
            return;
        }

        mode = 0;
        int device_nums = device_count_ensure_non_zero();
        ranges.clear();
        ranges.resize(device_nums);

        // init
        for (int i = 0; i < device_nums; ++i) {
            ranges[i] = GetCPUDefaultRange(i);
        }

        std::string inputStr(input);
        std::istringstream stream(inputStr);
        std::string option;

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
                    if (isAllDigits(key.substr(3))) {
                        int device_id = std::stoi(key.substr(3)); // Parse NPU device ID
                        if (device_id < device_nums) {
                            size_t dashPos = value.find('-');
                            if (dashPos != std::string::npos) {
                                std::string startStr = value.substr(0, dashPos);
                                std::string endStr = value.substr(dashPos + 1);
                                if (isAllDigits(startStr) && isAllDigits(endStr)) {
                                    coreId start = static_cast<coreId>(std::stoi(startStr));
                                    coreId end = static_cast<coreId>(std::stoi(endStr));
                                    ranges[device_id] = {start, end};
                                } else {
                                    ASCEND_LOGW("core range is %s-%s, should be all digits", startStr.c_str(), endStr.c_str());
                                }
                            } else {
                                if (isAllDigits(value)) {
                                    coreId singleCore = static_cast<coreId>(std::stoi(value));
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

    aclError SetThreadAffinity(c10::DeviceIndex device_id, ThreadType current_thread_type)
    {
        if (has_set_affinity) {
            ASCEND_LOGW("Thread affinity is already set.");
            return ACL_ERROR_NONE;
        }
        uint32_t bind_conf;
        std::vector<coreIdRange> ranges;
        parseCPUAffinityConf(bind_conf, ranges);
        printCoreRanges(ranges, bind_conf);

        // bind_conf=1, bind cores averagely based on device_id
        if (bind_conf == 1) {
            return SetThreadAffinity(ranges[device_id], pthread_self());
        } else if (bind_conf == 2) {
            auto thread_core_map = GetCpuAffinityMap(device_id);
            // Bind the main thread only when the dispatch phase begins (i.e., when ThreadType::backwardThread is set)
            if (current_thread_type == ThreadType::backwardThread)
                SetThreadAffinity(thread_core_map.at(ThreadType::mainThread), mainthread_tid);
            return SetThreadAffinity(thread_core_map.at(current_thread_type), pthread_self());
        } else {
            ASCEND_LOGD("Thread affinity setting is disabled.");
        }
        return ACL_ERROR_NONE;
    }

    void SetBackwardThreadName(c10::DeviceIndex device_id)
    {
        static thread_local bool seted = false;
        if (!seted) {
            seted = true;
            if (syscall(SYS_gettid) != getpid()) {
                SetThreadName(ThreadType::backwardThread);
                SetThreadAffinity(device_id);
            }
        }
    }

    void SetThreadName(ThreadType type)
    {
        // Ensure this is called at the start of the thread's execution to avoid frequent triggering of this function.
        if (prctl(PR_SET_NAME, threadTypeToNameMap.at(type).c_str()) != 0) {
            ASCEND_LOGW("set thread name failed!");
        }
    }

}