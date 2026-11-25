#pragma once


#include <atomic>
#include <set>
#include <unordered_set>

#include <ATen/record_function.h>

#include "torch_npu/csrc/profiler/profiler_types.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {

std::atomic<bool>& profDataReportEnable();

C10_NPU_API void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities);

C10_NPU_API void warmupNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> &activities);

C10_NPU_API void enableProfilerInChildThread(const NpuProfilerConfig &config);

C10_NPU_API void startNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> &activities, const std::unordered_set<at::RecordScope> &scops = {});

C10_NPU_API void disableProfilerInChildThread();

C10_NPU_API void stopNpuProfiler();

C10_NPU_API void finalizeNpuProfiler();

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id);

void reportMemoryDataToNpuProfiler(const MemoryUsage& data);

} // namespace profiler
} // namespace torch_npu
