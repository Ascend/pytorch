#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
namespace torch_npu {
namespace profiler {
std::map<std::string, aclprofAicoreMetrics> ProfilerMgr::npu_metrics_map_ = {
  {"ACL_AICORE_PIPE_UTILIZATION", ACL_AICORE_PIPE_UTILIZATION},
  {"ACL_AICORE_ARITHMETIC_UTILIZATION", ACL_AICORE_ARITHMETIC_UTILIZATION},
  {"ACL_AICORE_MEMORY_BANDWIDTH", ACL_AICORE_MEMORY_BANDWIDTH},
  {"ACL_AICORE_L0B_AND_WIDTH", ACL_AICORE_L0B_AND_WIDTH},
  {"ACL_AICORE_RESOURCE_CONFLICT_RATIO", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
  {"ACL_AICORE_MEMORY_UB", ACL_AICORE_MEMORY_UB},
  {"ACL_AICORE_L2_CACHE", ACL_AICORE_L2_CACHE},
  {"ACL_AICORE_NONE", ACL_AICORE_NONE},
};

std::map<std::string, uint64_t> ProfilerMgr::trace_level_map_ = {
  {"Level0", Level0},
  {"Level1", Level1},
  {"Level2", Level2},
};

ProfilerMgr::ProfilerMgr()
  : report_enable_(false),
    npu_trace_(false),
    profConfig_(nullptr) {}

void ProfilerMgr::Init(const std::string &path, bool npu_trace) {
  if (npu_trace == true) {
    at_npu::native::AclProfilingInit(path.c_str(), path.size());
    npu_trace_.store(true);
  }
  path_ = path;
}

void ProfilerMgr::EnableMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig) {
  profConfig_ = at_npu::native::AclProfilingCreateConfig(deviceIdList, deviceNum, aicMetrics, nullptr, dataTypeConfig);
  if (profConfig_ == nullptr) {
    NPU_LOGE("Create Prof Config failed.");
    return;
  }
  auto ret = at_npu::native::AclProfilingStart(profConfig_);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("Profiling start failed.");
    return;
  }
}

void ProfilerMgr::Start(const NpuTraceConfig &npu_config, bool cpu_trace) {
  c10_npu::npuSynchronizeDevice();
  if (npu_trace_.load() == true) {
    aclprofAicoreMetrics aic_metrics = ACL_AICORE_NONE;
    auto level_iter = trace_level_map_.find(npu_config.trace_level);
    uint64_t datatype_config = (level_iter == trace_level_map_.end()) ?
      Level0 : trace_level_map_[npu_config.trace_level];
    auto metrics_iter = npu_metrics_map_.find(npu_config.metrics);
    if (metrics_iter != npu_metrics_map_.end() && npu_config.metrics.compare("ACL_AICORE_NONE") != 0) {
      datatype_config |= ACL_PROF_AICORE_METRICS;
      aic_metrics = npu_metrics_map_[npu_config.metrics];
    }
    if (npu_config.l2_cache) {
      datatype_config |= ACL_PROF_L2CACHE;
    }
    if (npu_config.npu_memory) {
      datatype_config |= ACL_PROF_TASK_MEMORY;
      const std::string freq = "50";
      auto prof_ret = at_npu::native::AclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, freq.c_str(), freq.size());
      if (prof_ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        NPU_LOGW("not support to set config for sys-hardware-mem.");
      }
    }
    int32_t deviceId = 0;
    auto ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("Get Device ID failed.");
      return;
    }
    const uint32_t deviceNum = 1;
    uint32_t deviceIdList[deviceNum] = {deviceId};
    EnableMsProfiler(deviceIdList, deviceNum, aic_metrics, datatype_config);
  }

  if (cpu_trace == true) {
    std::string fwk_path = path_ + "/FRAMEWORK";
    constexpr uint32_t capacity = 262144;
    dataReceiver_.Init(fwk_path, capacity);
    dataReceiver_.Start();
    report_enable_.store(true);
  }
}

void ProfilerMgr::Stop() {
  c10_npu::npuSynchronizeDevice();
  if (report_enable_.load() == true) {
    dataReceiver_.Flush();
    dataReceiver_.Stop();
  }
  report_enable_.store(false);
  if (npu_trace_.load() == true) {
    at_npu::native::AclProfilingStop(profConfig_);
  }
}

void ProfilerMgr::Finalize() {
  if (npu_trace_.load() == true) {
    at_npu::native::AclProfilingFinalize();
  }
  npu_trace_.store(false);
}

void ProfilerMgr::Upload(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data) {
  dataReceiver_.Report(std::move(data));
}
} // profiler
} // torch_npu
