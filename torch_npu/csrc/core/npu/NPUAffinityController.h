#pragma once
#include <c10/core/Device.h>

namespace c10_npu {

using CoreId = unsigned int;
struct CoreIdRange {
    CoreId start;
    CoreId end;
};

enum ThreadType {
    MAIN_THREAD = 0,        // 1st performance hotspot, responsible for operator dispatching.
    ACL_THREAD = 1,         // 2rd performance hotspot in PTA, responsible for handling the task queue.
    RELEASE_THREAD = 2,     // Thread responsible for resource release.
    WATCHDOG_THREAD = 3,    // Thread responsible for HCCL communication monitoring.
    OTHER_THREAD = 4,       // Mostly refers to threads in PyTorch's motorized sleep thread pool, which
                            // are not considered in PTA.
    USER_THREAD = 5,        // Thread responsible for user.
};

void SetThreadType(ThreadType type);
void SetThreadAffinity(c10::DeviceIndex device);
void SetThreadAffinity(ThreadType type);
void SetThreadAffinity(int core_start, int core_end);

void SetMainThread();
bool NeedMainThreadBind();
void StartMainThreadBind(c10::DeviceIndex device_id);

} // namespace c10_npu