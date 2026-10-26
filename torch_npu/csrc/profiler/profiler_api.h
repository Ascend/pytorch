#pragma once

#include <atomic>
#include <set>
#include <unordered_set>

#include <ATen/record_function.h>

#include "torch_npu/csrc/profiler/profiler_types.h"

namespace torch_npu {
namespace profiler {

std::atomic<bool>& profDataReportEnable();

void initNpuProfiler(const std::string& path, const std::set<NpuActivityType>& activities);

void warmupNpuProfiler(const NpuProfilerConfig& config, const std::set<NpuActivityType>& activities);

void enableProfilerInChildThread(const NpuProfilerConfig& config);

void startNpuProfiler(
    const NpuProfilerConfig& config,
    const std::set<NpuActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scops = {});

void disableProfilerInChildThread();

void stopNpuProfiler();

void finalizeNpuProfiler();

void reportMarkDataToNpuProfiler(uint32_t category, const std::string& msg, uint64_t correlation_id);

void reportMemoryDataToNpuProfiler(const MemoryUsage& data);

} // namespace profiler
} // namespace torch_npu
