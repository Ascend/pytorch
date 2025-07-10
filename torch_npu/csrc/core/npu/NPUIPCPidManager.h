#pragma once
#include <cstdint>
#include <cstddef>

namespace torch_npu {
namespace ipc {

void addPid(int pid);
size_t getPids(int32_t** pids);

} // namespace ipc
} // namespace torch_npu