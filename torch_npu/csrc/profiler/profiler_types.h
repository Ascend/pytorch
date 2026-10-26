#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <acl/acl_rt.h>

namespace torch_npu {
namespace profiler {

enum class NpuActivityType {
  NONE = 0,
  CPU,
  NPU,
};

enum class MemoryComponentType {
  CACHING_ALLOCATOR = 0,
  WORKSPACE_ALLOCATOR,
};

enum class MemoryDataType { MEMORY_MALLOC = 0, MEMORY_FREE, MEMORY_BLOCK_FREE, MEMORY_INVALID };

enum class MemoryAllocatorType {
  ALLOCATOR_INNER = 0,
  ALLOCATOR_EXTERNAL,
  ALLOCATOR_INVALID,
};

struct MemoryUsage {
  int8_t device_type{0};
  int8_t device_index{0};
  uint8_t component_type{static_cast<uint8_t>(MemoryComponentType::CACHING_ALLOCATOR)};
  uint8_t data_type{static_cast<uint8_t>(MemoryDataType::MEMORY_INVALID)};
  uint8_t allocator_type{static_cast<uint8_t>(MemoryAllocatorType::ALLOCATOR_INVALID)};
  int64_t ptr{0};
  int64_t alloc_size{0};
  int64_t total_allocated{0};
  int64_t total_reserved{0};
  int64_t total_active{0};
  aclrtStream stream{nullptr};
};

struct ExperimentalConfig {
  ExperimentalConfig(
      std::string level = "Level0",
      std::string metrics = "ACL_AICORE_NONE",
      bool l2_cache = false,
      bool record_op_args = false,
      bool msprof_tx = false,
      bool op_attr = false,
      std::vector<std::string> host_sys = {},
      std::vector<std::string> mstx_domain_include = {},
      std::vector<std::string> mstx_domain_exclude = {},
      bool sys_io = false,
      bool sys_interconnection = false)
      : trace_level(level),
        metrics(metrics),
        l2_cache(l2_cache),
        record_op_args(record_op_args),
        msprof_tx(msprof_tx),
        op_attr(op_attr),
        host_sys(host_sys),
        mstx_domain_include(mstx_domain_include),
        mstx_domain_exclude(mstx_domain_exclude),
        sys_io(sys_io),
        sys_interconnection(sys_interconnection) {}
  ~ExperimentalConfig() = default;

  std::string trace_level;
  std::string metrics;
  bool l2_cache;
  bool record_op_args;
  bool msprof_tx;
  bool op_attr;
  std::vector<std::string> host_sys;
  std::vector<std::string> mstx_domain_include;
  std::vector<std::string> mstx_domain_exclude;
  bool sys_io;
  bool sys_interconnection;
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

} // namespace profiler
} // namespace torch_npu
