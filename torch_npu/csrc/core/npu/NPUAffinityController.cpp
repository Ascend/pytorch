#include <torch_npu/csrc/core/npu/NPUAffinityController.h>

#include <pthread.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include <torch_npu/csrc/core/npu/GetAffinityCPUInfo.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NpuVariables.h>

namespace c10_npu {

namespace {

thread_local ThreadType local_thread = ThreadType::MAIN_THREAD;

pthread_t main_thread;
bool start_main_thread_bind = false;
std::mutex core_map_mutex;
bool lazy_bind = true;
bool force_bind = false;

using ThreadCoreMap = std::unordered_map<ThreadType, CoreIdList>;

uint32_t cpu_affinity_mode;
std::vector<CoreIdList> devices_aff_cores;
std::unordered_map<c10::DeviceIndex, ThreadCoreMap> device_thread_core_maps;

const std::initializer_list<ThreadType> threadTypeList =
    {MAIN_THREAD, ACL_THREAD, RELEASE_THREAD, WATCHDOG_THREAD, OTHER_THREAD};

const std::unordered_map<ThreadType, std::string> threadTypeToNameMap = {
    {MAIN_THREAD, "main_thread"},
    {ACL_THREAD, "acl_thread"},
    {RELEASE_THREAD, "release_thread"},
    {WATCHDOG_THREAD, "hccl_watchdog_t"},
    {OTHER_THREAD, "other_thread"}};

std::string formatCoreRange(const CoreIdList& cores) {
  if (cores.empty()) {
    return "";
  }

  std::ostringstream oss;
  auto it = cores.begin();
  while (it != cores.end()) {
    CoreId start = *it;
    CoreId end = start;
    auto next_it = std::next(it);
    while (next_it != cores.end() && *next_it == end + 1) {
      ++end;
      ++next_it;
    }

    if (start == end) {
      oss << start;
    } else {
      oss << start << "-" << end;
    }

    it = next_it;
    if (it != cores.end()) {
      oss << ",";
    }
  }
  return oss.str();
}

CoreIdList getNPUDefaultCores(c10::DeviceIndex device_id) {
  static int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
  int device_nums = device_count_ensure_non_zero();
  int block_size =
      (core_nums > 0 && device_nums > 0) ? core_nums / device_nums : 0;
  CoreIdList cores;
  for (int i = 0; i < block_size; ++i) {
    cores.insert(static_cast<CoreId>(device_id * block_size + i));
  }
  return cores;
}

// Parse npu_affine setting from CPU_AFFINITY_CONF
void parseNpuAffineMode(
    const std::string& inputStr,
    int device_nums,
    std::vector<CoreIdList>& devices_aff_cores) {
  static const std::regex pattern("npu_affine:(\\d)");
  std::smatch match;
  if (std::regex_search(inputStr, match, pattern) &&
      std::stoi(match[1].str()) != 0) {
    ASCEND_LOGD(
        "Get npu_affine mode: %s, device_nums: %d, set affinity cores for each device.",
        match[1].str().c_str(),
        device_nums);
    for (int i = 0; i < device_nums; i++) {
      CoreIdList cores = GetAffinityCores(i);
      if (cores.size() == 0) {
        ASCEND_LOGW("Device-%d has no affinity cores.", i);
        continue;
      }
      devices_aff_cores[i] = cores;
      ASCEND_LOGD(
          "Device-%d affinity cores [%s].",
          i,
          formatCoreRange(devices_aff_cores[i]).c_str());
    }
  }
}

// Parse lazy_bind setting from CPU_AFFINITY_CONF
void parseLazyBindMode(const std::string& inputStr) {
  std::regex pattern_for_lazy_bind("lazy_bind:(\\d)");
  std::smatch match_for_lazy_bind;
  if (std::regex_search(inputStr, match_for_lazy_bind, pattern_for_lazy_bind)) {
    lazy_bind = std::stoi(match_for_lazy_bind[1].str()) == 0 ? false : true;
  }
}

// Parse force setting from CPU_AFFINITY_CONF
void parseForceMode(const std::string& inputStr) {
  std::regex pattern_for_force("force:(\\d)");
  std::smatch match_for_force;
  if (std::regex_search(inputStr, match_for_force, pattern_for_force)) {
    int force_val = std::stoi(match_for_force[1].str());
    if (force_val != 0 && force_val != 1) {
      ASCEND_LOGE("force value must be 0 or 1, got: %d", force_val);
    } else {
      force_bind = (force_val != 0);
    }
  } else {
    std::regex pattern_for_force_check("force:([^,]+)");
    std::smatch match_for_force_check;
    if (std::regex_search(
            inputStr, match_for_force_check, pattern_for_force_check)) {
      ASCEND_LOGE(
          "force value must be 0 or 1, got: %s",
          match_for_force_check[1].str().c_str());
    }
  }
}

// Parse mode from CPU_AFFINITY_CONF when only digits or mode:xxx is provided
bool parseModeOnly(const std::string& inputStr, uint32_t& mode) {
  // Handle cases where only `mode` is provided, or `mode:` without value
  if (isAllDigits(inputStr)) {
    mode = static_cast<uint32_t>(std::stoi(inputStr));
    return true;
  }

  std::istringstream stream(inputStr);
  std::string option;
  while (std::getline(stream, option, ',')) {
    size_t colonPos = option.find(':');
    if (colonPos == std::string::npos && isAllDigits(option)) {
      mode = static_cast<uint32_t>(std::stoi(option));
      return false;
    }
    std::string key = option.substr(0, colonPos);
    if (key == "mode") {
      std::string value = option.substr(colonPos + 1);
      if (isAllDigits(value)) {
        mode = static_cast<uint32_t>(std::stoi(value));
      } else {
        ASCEND_LOGW("mode is %s, should be all digits", value.c_str());
      }
      return false;
    }
  }
  return false;
}

// Parse device-specific core range from CPU_AFFINITY_CONF (e.g., npu0:0-1)
void parseDeviceCoreRange(
    const std::string& inputStr,
    int device_nums,
    std::vector<CoreIdList>& devices_aff_cores) {
  std::istringstream stream(inputStr);
  std::string option;
  std::set<int> user_def_devices;
  while (std::getline(stream, option, ',')) {
    size_t colonPos = option.find(':');
    if (colonPos == std::string::npos) {
      continue;
    }
    std::string key = option.substr(0, colonPos);
    std::string value = option.substr(colonPos + 1);

    std::regex npuPattern("^npu[0-9]{1,2}$");
    if (!std::regex_match(key, npuPattern)) {
      ASCEND_LOGW("Invalid device name: %s", key.c_str());
      continue;
    }
    int device_id = std::stoi(key.substr(3)); // Skip first 3 chars ("npu").
    if (device_id >= device_nums || device_id < 0) {
      ASCEND_LOGW(
          "device_id in CPU_AFFINITY_CONF is %d, should be in range [0, %d)",
          device_id,
          device_nums);
      continue;
    }
    if (user_def_devices.count(device_id) == 0) {
      user_def_devices.insert(device_id);
      devices_aff_cores[device_id].clear();
    }
    if (isAllDigits(value)) {
      CoreId singleCore = static_cast<CoreId>(std::stoi(value));
      devices_aff_cores[device_id].insert(singleCore);
      continue;
    }
    size_t dashPos = value.find('-');
    if (dashPos != std::string::npos) {
      std::string startStr = value.substr(0, dashPos);
      std::string endStr = value.substr(dashPos + 1);
      if (isAllDigits(startStr) && isAllDigits(endStr)) {
        CoreId start = static_cast<CoreId>(std::stoi(startStr));
        CoreId end = static_cast<CoreId>(std::stoi(endStr));
        for (CoreId core = start; core <= end; ++core) {
          devices_aff_cores[device_id].insert(core);
        }
      } else {
        ASCEND_LOGW(
            "core range is %s-%s, should be all digits",
            startStr.c_str(),
            endStr.c_str());
      }
    }
  }
}

void parseCPUAffinityConf(
    uint32_t& mode,
    std::vector<CoreIdList>& devices_aff_cores) {
  int device_nums = device_count_ensure_non_zero();
  devices_aff_cores.clear();
  devices_aff_cores.resize(device_nums);
  ASCEND_LOGD("Get device nums: %d by aclrtGetDeviceCount.", device_nums);
  for (int i = 0; i < device_nums; ++i) {
    devices_aff_cores[i] = getNPUDefaultCores(i);
  }
  mode = 0;

  const char* input = c10_npu::option::OptionsManager::GetCpuAffinityConf();
  if (input == nullptr || strlen(input) == 0) {
    return; // CPU_AFFINITY_CONF is not set, use default cores
  }
  ASCEND_LOGD("Get env var CPU_AFFINITY_CONF: %s", input);

  const std::string inputStr(input);

  // Parse mode if only digits or mode:xxx is provided
  if (parseModeOnly(inputStr, mode)) {
    ASCEND_LOGD("Only mode is provided, mode: %d", mode);
    return;
  }

  parseNpuAffineMode(inputStr, device_nums, devices_aff_cores);
  parseLazyBindMode(inputStr);
  parseForceMode(inputStr);
  // Parse device-specific core ranges defined by user
  parseDeviceCoreRange(inputStr, device_nums, devices_aff_cores);
}

void printCoreRanges(
    const uint32_t mode,
    const std::vector<CoreIdList>& devices_aff_cores) {
  std::ostringstream oss;
  oss << "Mode: " << mode << ". Core range for each device ID: ";

  for (size_t i = 0; i < devices_aff_cores.size(); ++i) {
    oss << "Device " << i << ": [" << formatCoreRange(devices_aff_cores[i])
        << "]";
    std::string end_str = (i == devices_aff_cores.size() - 1) ? "." : "; ";
    oss << end_str;
  }
  ASCEND_LOGD("Read CPU affinity config: %s", oss.str().c_str());
}

std::string formatCPUSetMask(const cpu_set_t& mask) {
  CoreIdList cores;
  int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
  for (int i = 0; i < cpu_count; ++i) {
    if (CPU_ISSET(i, &mask)) {
      cores.insert(i);
    }
  }
  return formatCoreRange(cores);
}

bool checkThreadAffinityConflict(
    const std::vector<CoreIdList>& devices_aff_cores) {
  cpu_set_t mask;
  pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
  std::string affinity_mask_str = formatCPUSetMask(mask);
  ASCEND_LOGI(
      "Current thread CPU affinity mask: %s", affinity_mask_str.c_str());

  for (auto& cores : devices_aff_cores) {
    for (auto& core : cores) {
      if (!CPU_ISSET(core, &mask)) {
        ASCEND_LOGW(
            "Thread affinity conflict detected! Expected core %u (in config range [%s]) is NOT in current thread affinity mask. %s",
            core,
            formatCoreRange(cores).c_str(),
            affinity_mask_str.c_str());
        ASCEND_LOGW(
            "Thread affinity is already set. Use force:1 to skip this check and force bind.");
        return false;
      }
    }
  }
  return true;
}

bool getThreadAffinityInfo() {
  parseCPUAffinityConf(cpu_affinity_mode, devices_aff_cores);
  printCoreRanges(cpu_affinity_mode, devices_aff_cores);

  for (int i = 0; i < devices_aff_cores.size(); ++i) {
    if (devices_aff_cores[i].size() > 0) {
      ASCEND_LOGD(
          "Device %d get cores %s.",
          i,
          formatCoreRange(devices_aff_cores[i]).c_str());
    }
  }

  if (cpu_affinity_mode == 0) {
    return false;
  }

  if (force_bind) {
    ASCEND_LOGI(
        "CPU affinity force mode enabled, skipping affinity conflict detection, applying CPU_AFFINITY_CONF binding.");
    return true;
  }

  return checkThreadAffinityConflict(devices_aff_cores);
}

std::string getAffinityMapAsString(
    c10::DeviceIndex device_id,
    const ThreadCoreMap& threadCoreMap) {
  std::ostringstream oss;
  for (auto thread_type : threadTypeList) {
    oss << threadTypeToNameMap.at(thread_type) << ": ["
        << formatCoreRange(threadCoreMap.at(thread_type)) << "]";
    std::string end_str =
        (thread_type == ThreadType::OTHER_THREAD) ? "." : "; ";
    oss << end_str;
  }
  return oss.str();
}

ThreadCoreMap getCpuAffinityMap(
    c10::DeviceIndex device_id,
    const std::vector<CoreIdList>& devices_aff_cores) {
  ThreadCoreMap threadCoreMap;
  CoreIdList cores = devices_aff_cores[device_id];
  if (cores.size() < threadTypeList.size()) {
    ASCEND_LOGW(
        "Device %d available core numbers (%zu) are insufficient for all %zu thread types and will bind available cores to all threads.",
        device_id,
        cores.size(),
        threadTypeList.size());
    for (auto thread_type : threadTypeList) {
      threadCoreMap[thread_type] = cores;
    }
    return threadCoreMap;
  }
  for (auto thread_type : threadTypeList) {
    if (thread_type != ThreadType::OTHER_THREAD) {
      CoreId first_core = *cores.begin();
      threadCoreMap[thread_type].insert(first_core);
      cores.erase(first_core);
    } else {
      threadCoreMap[ThreadType::OTHER_THREAD] = cores;
    }
  }

  ASCEND_LOGD(
      "Device %d thread affinity map: %s",
      device_id,
      getAffinityMapAsString(device_id, threadCoreMap).c_str());
  return threadCoreMap;
}

CoreIdList getCoreList(c10::DeviceIndex device_id, ThreadType type) {
  CoreIdList core_list;
  if (cpu_affinity_mode == 0 || cpu_affinity_mode == 1) {
    core_list = devices_aff_cores[device_id];
  } else {
    std::lock_guard<std::mutex> lock(core_map_mutex);
    if (device_thread_core_maps.find(device_id) ==
        device_thread_core_maps.end()) {
      device_thread_core_maps.emplace(
          device_id, getCpuAffinityMap(device_id, devices_aff_cores));
    }
    core_list = device_thread_core_maps.at(device_id).at(type);
  }
  return core_list;
}

bool setThreadAffinityImpl(pthread_t thread, CoreIdList core_list) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto core : core_list) {
    CPU_SET(core, &mask);
  }
  return pthread_setaffinity_np(thread, sizeof(mask), &mask) == 0;
}

inline bool needToSetThreadAffinity() {
  static bool need_to_set_affinity = getThreadAffinityInfo();
  return need_to_set_affinity;
}

} // namespace

void SetThreadType(ThreadType type) {
  // Called at the start of the thread's execution to avoid frequent triggering
  // of this function.
  local_thread = type;
  if (type == ThreadType::OTHER_THREAD || type == ThreadType::MAIN_THREAD) {
    return;
  }
  if (prctl(PR_SET_NAME, threadTypeToNameMap.at(type).c_str()) != 0) {
    ASCEND_LOGW(
        "Set thread name to %s failed!", threadTypeToNameMap.at(type).c_str());
  }
  ASCEND_LOGD(
      "Set thread name to %s success.", threadTypeToNameMap.at(type).c_str());
}

void SetThreadAffinity(c10::DeviceIndex device_id) {
  if (!needToSetThreadAffinity() || local_thread == ThreadType::USER_THREAD) {
    return;
  }

  CoreIdList core_list = getCoreList(device_id, local_thread);
  std::string range_str = formatCoreRange(core_list);
  if (setThreadAffinityImpl(pthread_self(), core_list)) {
    ASCEND_LOGD(
        "Device %d set %s affinity to %s success.",
        device_id,
        threadTypeToNameMap.at(local_thread).c_str(),
        range_str.c_str());
  } else {
    ASCEND_LOGE(
        "Device %d set %s affinity to %s failed.",
        device_id,
        threadTypeToNameMap.at(local_thread).c_str(),
        range_str.c_str());
  }
}

void SetThreadAffinity(ThreadType type) {
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

void SetThreadAffinity(const CoreIdList core_ids) {
  if (!needToSetThreadAffinity()) {
    return;
  }
  CoreIdList processed_core_ids = core_ids;
  static int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
  for (auto it = processed_core_ids.begin(); it != processed_core_ids.end();) {
    if (static_cast<int>(*it) >= core_nums) {
      ASCEND_LOGW(
          "core id %d >= core_nums %d, it will be ignored when setting thread affinity.",
          *it,
          core_nums);
      it = processed_core_ids.erase(it);
    } else {
      ++it;
    }
  }
  local_thread = ThreadType::USER_THREAD;
  if (setThreadAffinityImpl(pthread_self(), processed_core_ids)) {
    ASCEND_LOGD(
        "Set thread affinity to user-defined range %s success.",
        formatCoreRange(processed_core_ids).c_str());
  } else {
    ASCEND_LOGE(
        "Set thread affinity to user-defined range %s failed.",
        formatCoreRange(processed_core_ids).c_str());
  }
}

void SetThreadAffinity(int core_start, int core_end) {
  CoreIdList core_list;
  for (int i = core_start; i <= core_end; ++i) {
    core_list.insert(static_cast<CoreId>(i));
  }
  SetThreadAffinity(core_list);
}

void SetMainThread() {
  main_thread = pthread_self();
}

bool NeedMainThreadBind() {
  return start_main_thread_bind && (local_thread == ThreadType::MAIN_THREAD);
}

bool SetThreadAffinityInInitialize() {
  if (needToSetThreadAffinity() && !lazy_bind) {
    return true;
  }
  return false;
}

void StartMainThreadBind(c10::DeviceIndex device_id) {
  if (!needToSetThreadAffinity() || local_thread == ThreadType::USER_THREAD) {
    return;
  }

  static thread_local bool seted = false;
  if (seted) {
    return;
  }
  seted = true;
  if (syscall(SYS_gettid) != getpid()) {
    start_main_thread_bind = true;
    SetThreadAffinity(device_id);
    CoreIdList core_list = getCoreList(device_id, ThreadType::MAIN_THREAD);
    if (setThreadAffinityImpl(main_thread, core_list)) {
      ASCEND_LOGD(
          "Device %d set %s affinity to %s success.",
          device_id,
          threadTypeToNameMap.at(ThreadType::MAIN_THREAD).c_str(),
          formatCoreRange(core_list).c_str());
    } else {
      ASCEND_LOGE(
          "Device %d set %s affinity to %s failed.",
          device_id,
          threadTypeToNameMap.at(ThreadType::MAIN_THREAD).c_str(),
          formatCoreRange(core_list).c_str());
    }
  }
}

} // namespace c10_npu
