#pragma once

#include <algorithm>
#include <c10/core/Device.h>
#include <torch_npu/csrc/core/npu/GetAffinityCPUInfo.h>
#include <set>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {

enum ThreadType {
  MAIN_THREAD =
      0, // 1st performance hotspot, responsible for operator dispatching.
  ACL_THREAD = 1, // 2nd performance hotspot in PTA, responsible for handling
                  // the task queue.
  RELEASE_THREAD = 2, // Thread responsible for resource release.
  WATCHDOG_THREAD = 3, // Thread responsible for HCCL communication monitoring.
  OTHER_THREAD = 4, // Mostly refers to threads in PyTorch's motorized sleep
                    // thread pool, which are not considered in PTA.
  USER_THREAD = 5, // Thread responsible for user.
};

void SetThreadType(ThreadType type);
C10_NPU_API void SetThreadAffinity(c10::DeviceIndex device);
C10_NPU_API void SetThreadAffinity(ThreadType type);
C10_NPU_API void SetThreadAffinity(const CoreIdList core_ids);
C10_NPU_API void SetThreadAffinity(int core_start, int core_end);

bool NeedMainThreadBind();
bool SetThreadAffinityInInitialize();
void StartMainThreadBind(c10::DeviceIndex device_id);

inline bool isAllDigits(const std::string& str) {
  if (str.empty()) {
    return false;
  }
  return std::all_of(
      str.begin(), str.end(), [](unsigned char c) { return std::isdigit(c); });
}

} // namespace c10_npu