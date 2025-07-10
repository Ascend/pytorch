#include "torch_npu/csrc/core/npu/NPUIPCPidManager.h"
namespace torch_npu {
namespace ipc {

int32_t* pids = nullptr;
size_t pid_num = 0;
size_t capacity = 0;

void addPid(int pid)
{
    const size_t requiredCapacity = pid_num + 1;

    if (requiredCapacity > capacity) {
        size_t newCapacity = capacity + 10;

        int32_t* newArray = new int32_t[newCapacity];
        for (int i = 0; i < pid_num; ++i) {
            newArray[i] = pids[i];
        }

        delete[] pids;
        pids = newArray;
        capacity = newCapacity;
    }

    pids[pid_num++] = static_cast<int32_t>(pid);
}

size_t getPids(int32_t** ret_pids)
{
    *ret_pids = pids;
    return pid_num;
}

} // namespace ipc
} // namespace torch_npu