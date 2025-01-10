#pragma once
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace torch_npu {
namespace profiler {

constexpr int MaxParentPids = 5;
int32_t GetProcessId();
std::string GenerateUuidV4();
std::vector<int32_t> GetPids();
std::pair<int32_t, std::string> GetParentPidAndCommand(int32_t pid);
std::vector<std::pair<int32_t, std::string>> GetPidCommandPairsofAncestors();

} // namespace profiler
} // namespace torch_npu
