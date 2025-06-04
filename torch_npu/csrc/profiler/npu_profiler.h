#pragma once

#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include <ATen/record_function.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/profiler/mstx_mgr.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"

namespace torch_npu {
namespace profiler {
namespace python_tracer {
enum class Command { kStartOne = 0, kStartAll, kStop, kClear };
using CallFn = void (*)(Command);
void registerFunctions(CallFn call);
} // python_tracer

enum class NpuActivityType {
    NONE = 0,
    CPU,
    NPU,
};

enum class MemoryComponentType {
    CACHING_ALLOCATOR = 0,
    WORKSPACE_ALLOCATOR,
};

enum class MemoryDataType {
    MEMORY_MALLOC = 0,
    MEMORY_FREE,
    MEMORY_BLOCK_FREE,
    MEMORY_INVALID
};

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
    int64_t stream_ptr{0};
};

struct ExperimentalConfig {
    ExperimentalConfig(std::string level = "Level0", std::string metrics = "ACL_AICORE_NONE",
                       bool l2_cache = false, bool record_op_args = false, bool msprof_tx = false,
                       bool op_attr = false, std::vector<std::string> host_sys = {}, std::vector<std::string> mstx_domain_include = {},
                       std::vector<std::string> mstx_domain_exclude = {}, bool sys_io = false,
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

std::atomic<bool>& profDataReportEnable();

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities);

void warmupNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> &activities);

void startNpuProfiler(const NpuProfilerConfig &config, const std::set<NpuActivityType> &activities, const std::unordered_set<at::RecordScope> &scops = {});

void stopNpuProfiler();

void finalizeNpuProfiler();

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id);

void reportMemoryDataToNpuProfiler(const MemoryUsage& data);

inline void mstxMark(const char* message, const aclrtStream stream, const char* domain)
{
    if (at_npu::native::IsSupportMstxFunc()) {
        MstxMgr::GetInstance()->mark(message, stream, domain);
    } else {
        (void)at_npu::native::AclProfilingMarkEx(message, strlen(message), stream);
    }
}

inline int mstxRangeStart(const char* message, const aclrtStream stream, const char* domain)
{
    return MstxMgr::GetInstance()->rangeStart(message, stream, domain);
}

inline void mstxRangeEnd(int id, const char* domain)
{
    MstxMgr::GetInstance()->rangeEnd(id, domain);
}

inline bool mstxEnable()
{
    return MstxMgr::GetInstance()->isMstxEnable();
}

struct MstxRange {
    int rangeId{0};
    mstxDomainHandle_t domainHandle{nullptr};
    MstxRange(const std::string &message, aclrtStream stream, const std::string &domainName = "default")
    {
        if (!mstxEnable()) {
            return;
        }
        rangeId = MstxMgr::GetInstance()->getRangeId();
        if (at_npu::native::IsSupportMstxDomainFunc()) {
            if (MstxMgr::GetInstance()->isMstxTxDomainEnable(domainName)) {
                domainHandle = MstxMgr::GetInstance()->createProfDomain(domainName);
                at_npu::native::MstxDomainRangeStartA(domainHandle, message.c_str(), stream, rangeId);
            }
        } else {
            at_npu::native::MstxRangeStartA(message.c_str(), stream, rangeId);
        }
    }

    ~MstxRange()
    {
        if (rangeId == 0 || !mstxEnable()) {
            return;
        }
        if (at_npu::native::IsSupportMstxDomainFunc()) {
            if (domainHandle != nullptr) {
                at_npu::native::MstxDomainRangeEnd(domainHandle, rangeId);
            }
        } else {
            at_npu::native::MstxRangeEnd(rangeId);
        }
    }
};
} // profiler
} // torch_npu
