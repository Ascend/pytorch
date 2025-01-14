#pragma once
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace c10_npu {

    typedef unsigned int coreId;

    struct coreIdRange {
        coreId start;
        coreId end;
    };

    enum ThreadType {
        unknownThread = 0, // Mostly refers to threads in PyTorch's motorized sleep thread pool, which are not considered in PTA.
        mainThread = 1,    // 1st performance hotspot, responsible for operator dispatching during the forward phase.
        backwardThread = 2,  // 2nd performance hotspot, responsible for operator dispatching during the backward phase.
        aclThread = 3,     // 3rd performance hotspot in PTA, responsible for handling the task queue.
        releaseThread = 4, // Thread responsible for resource release.
        hcclCommWatchdogThread = 5 // Thread responsible for HCCL communication monitoring.
    };

    aclError SetThreadAffinity(c10::DeviceIndex device);
    aclError SetThreadAffinity(c10::DeviceIndex device, ThreadType current_thread_type);
    void SetThreadName(ThreadType type);

    // The main thread of PTA, which is also the main thread of PyTorch, handles multiple phases of tasks
    // (e.g., first parallel checkpoint data loading, then transitioning to forward training).
    // Each phase may require different thread affinity settings. Therefore, we record the thread's TID
    // to adjust its affinity later as needed.
    void GetAffinityInfo();

    // Set backwardThread Name Once
    void SetBackwardThreadName(c10::DeviceIndex device_id);

}