#ifndef TORCH_NPU_TOOLKIT_PROFILER_NPU_PROFILER_INC
#define TORCH_NPU_TOOLKIT_PROFILER_NPU_PROFILER_INC
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include <ATen/record_function.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

namespace torch_npu {
namespace profiler {
enum class NpuActivityType {
  NONE = 0,
  CPU,
  NPU,
};

struct NpuProfilerConfig {
  explicit NpuProfilerConfig(
    std::string path,
    bool record_shapes = false,
    bool profile_memory = false,
    bool with_stack = false,
    bool with_flops = false,
    bool with_modules = false)
    : path(path),
      record_shapes(record_shapes),
      profile_memory(profile_memory),
      with_stack(with_stack),
      with_flops(with_flops),
      with_modules(with_modules) {}

    ~NpuProfilerConfig() = default;
    std::string path;
    bool record_shapes;
    bool profile_memory;
    bool with_stack;
    bool with_flops;
    bool with_modules;
};

bool profDataReportEnable();

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities);

void startNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> & activities, const std::unordered_set<at::RecordScope> &scops = {});

void stopNpuProfiler();

void finalizeNpuProfiler();

void reportData(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data);

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id);
} // profiler
} // torch_npu
#endif
