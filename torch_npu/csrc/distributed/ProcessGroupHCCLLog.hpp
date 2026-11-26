#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/logging/Logger.h"

namespace c10d_npu {
class ProcessGroupHCCL;
}

namespace c10d_npu::hccl_log {

extern const std::shared_ptr<npu_logging::Logger> loggerHccl;
const char* reduceOpNameForLog(c10d::ReduceOp::RedOpType op);
const char* reduceOpNameForLog(HcclReduceOp op);
const char* hcclDataTypeNameForLog(HcclDataType type);

// Bound list previews so a large group fits in one API log line.
constexpr size_t kHcclLogListLimit = 8;
template <typename T>
std::string hcclLogList(const std::vector<T>& values) {
  const auto total = values.size();
  const auto shown = c10::ArrayRef<T>(values).slice(0, std::min(values.size(), kHcclLogListLimit));
  const auto joined = c10::Join(", ", shown);
  if (shown.size() < total) {
    return c10::str("[", joined, ", ... (", total, " total)]");
  }
  return c10::str("[", joined, "]");
}

long long hcclGlobalRankForLog(const std::vector<uint32_t>& ranks, int64_t pgRank);
std::string hcclRankPrefixForLog(const std::string& groupName, ProcessGroupHCCL& pg);
std::string hcclCommNameForLog(HcclComm comm);

#define TORCH_NPU_HCCL_API_BEFORE_LOGD(api, comm, commName, stream, format, ...)                 \
  TORCH_NPU_LOGD(                                                                                \
      ::c10d_npu::hccl_log::loggerHccl, "[%s] Calling %s, device %d, stream %lld, " format ".", \
      ((commName) = ::c10d_npu::hccl_log::hcclCommNameForLog(comm)).c_str(),                     \
      #api, static_cast<int>((stream).device_index()), static_cast<long long>((stream).id()),    \
      __VA_ARGS__)

#define TORCH_NPU_HCCL_API_AFTER_LOGD(api, commName, stream, result)                                  \
  TORCH_NPU_LOGD(                                                                                     \
      ::c10d_npu::hccl_log::loggerHccl, "[%s] %s returned, device %d, stream %lld, result %d.", \
      (commName).c_str(), #api, static_cast<int>((stream).device_index()),                            \
      static_cast<long long>((stream).id()), static_cast<int>(result))

#define TORCH_NPU_HCCL_ENTRY_LOGD(fields, ...)                                      \
  TORCH_NPU_LOGD(                                                                   \
      ::c10d_npu::hccl_log::loggerHccl, "%sEntering %s, group size %d" fields ".", \
      ::c10d_npu::hccl_log::hcclRankPrefixForLog(pg_name_, *this).c_str(), __func__, getSize(), ##__VA_ARGS__)

} // namespace c10d_npu::hccl_log
