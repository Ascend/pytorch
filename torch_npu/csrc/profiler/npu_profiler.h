#pragma once

#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include <ATen/record_function.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"

namespace torch_npu {
namespace profiler {
enum class NpuActivityType {
  NONE = 0,
  CPU,
  NPU,
};

enum class MemoryDataType {
    MEMORY_MALLOC = 0,
    MEMORY_FREE,
    MEMORY_BLOCK_FREE,
    MEMORY_INVALID
};

struct MemoryUsage {
    int8_t device_type{0};
    uint8_t device_index{0};
    uint8_t data_type{static_cast<uint8_t>(MemoryDataType::MEMORY_INVALID)};
    int64_t ptr{0};
    int64_t alloc_size{0};
    int64_t total_allocated{0};
    int64_t total_reserved{0};
    int64_t total_active{0};
    int64_t stream_ptr{0};
};

struct ExperimentalConfig {
  ExperimentalConfig(std::string level = "Level0", std::string metrics = "ACL_AICORE_NONE", bool l2_cache = false, bool record_op_args = false)
      : trace_level(level),
        metrics(metrics),
        l2_cache(l2_cache),
        record_op_args(record_op_args) {}
  ~ExperimentalConfig() = default;

  std::string trace_level;
  std::string metrics;
  bool l2_cache;
  bool record_op_args;
};

struct NpuProfilerConfig {
  explicit NpuProfilerConfig(
    std::string path,
    bool record_shapes = false,
    bool profile_memory = false,
    bool with_stack = false,
    bool with_flops = false,
    bool with_modules = false,
    ExperimentalConfig experimental_config = ExperimentalConfig())
    : path(path),
      record_shapes(record_shapes),
      profile_memory(profile_memory),
      with_stack(with_stack),
      with_flops(with_flops),
      with_modules(with_modules),
      experimental_config(experimental_config) {}

    ~NpuProfilerConfig() = default;
    std::string path;
    bool record_shapes;
    bool profile_memory;
    bool with_stack;
    bool with_flops;
    bool with_modules;
    ExperimentalConfig experimental_config;
};

std::atomic<bool>& profDataReportEnable();

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities);

void startNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> &activities, const std::unordered_set<at::RecordScope> &scops = {});

void stopNpuProfiler();

void finalizeNpuProfiler();

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id);

void reportMemoryDataToNpuProfiler(const MemoryUsage& data);
} // profiler
} // torch_npu
