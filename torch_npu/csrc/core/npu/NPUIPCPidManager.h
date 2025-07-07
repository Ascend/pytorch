#pragma once
#include <cstdint>

namespace torch_npu {
namespace ipc {

void addPid(int pid);
int getPids(int32_t** pids);

} // namespace ipc
} // namespace torch_npu