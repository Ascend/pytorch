#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/profiler/feature_mgr.h"

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
    {"Level_none", Level_none},
};

constexpr uint32_t capacity_ = 262144;          // 2^18, Experience value for default ringbuffer size for single data
constexpr uint32_t trace_capacity_ = 128;       // 2^7, Experience value for python trace data ringbuffer size for batch data

ProfilerMgr::ProfilerMgr()
    : report_enable_(false),
      npu_trace_(false),
      record_op_args_(false),
      profile_memory_(false),
      msprof_tx_(false),
      profConfig_(nullptr) {}

void ProfilerMgr::Init(const std::string &path, bool npu_trace) {
  if (npu_trace == true) {
    at_npu::native::AclProfilingInit(path.c_str(), path.size());
    npu_trace_.store(true);
    FeatureMgr::GetInstance()->Init();
  }
  path_ = path;
}

void ProfilerMgr::EnableMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig) {
  profConfig_ = at_npu::native::AclProfilingCreateConfig(deviceIdList, deviceNum, aicMetrics, nullptr, dataTypeConfig);
  if (profConfig_ == nullptr) {
    ASCEND_LOGE("Create Prof Config failed.");
    return;
  }
  auto ret = at_npu::native::AclProfilingStart(profConfig_);
  if (ret != ACL_ERROR_NONE) {
    ASCEND_LOGE("Profiling start failed.");
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
        if (npu_config.msprof_tx) {
            datatype_config |= ACL_PROF_MSPROFTX;
        }
    if (npu_config.npu_memory) {
      datatype_config |= ACL_PROF_TASK_MEMORY;
      const std::string freq = "50";
      auto prof_ret = at_npu::native::AclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, freq.c_str(), freq.size());
      if (prof_ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        ASCEND_LOGW("not support to set config for sys-hardware-mem.");
      }
    }
    if (npu_config.op_attr) {
        datatype_config |= ACL_PROF_OP_ATTR;
    }
    datatype_config = CheckFeatureConfig(datatype_config);
    int32_t deviceId = 0;
    auto ret = c10_npu::GetDevice(&deviceId);
    if (ret != ACL_ERROR_NONE) {
      ASCEND_LOGE("Get Device ID failed.");
      return;
    }
    const uint32_t deviceNum = 1;
    uint32_t deviceIdList[deviceNum] = {deviceId};
    EnableMsProfiler(deviceIdList, deviceNum, aic_metrics, datatype_config);
  }

  if (cpu_trace == true) {
    StartDataReceiver();
    report_enable_.store(true);
    profile_memory_.store(npu_config.npu_memory);
  }
    msprof_tx_.store(npu_config.msprof_tx);
  if (npu_config.record_op_args) {
    record_op_args_.store(true);
    const std::string op_dump_path = std::string(path_.begin(), path_.begin() + path_.find_last_not_of("/") + 1) +
      "_op_args";
    at_npu::native::AclopStartDumpArgs(ACL_OP_DUMP_OP_AICORE_ARGS, op_dump_path.c_str());
  }
}

void ProfilerMgr::Stop() {
  c10_npu::npuSynchronizeDevice();
  if (report_enable_.load() == true) {
    StopDataReceiver();
    profile_memory_.store(false);
  }
  report_enable_.store(false);
  if (npu_trace_.load() == true) {
    at_npu::native::AclProfilingStop(profConfig_);
    auto ret = at_npu::native::AclProfilingDestroyConfig(profConfig_);
    if (ret != ACL_SUCCESS) {
        ASCEND_LOGE("AclProfDestoryConfig fail, error code: %d", ret);
    }
    profConfig_ = nullptr;
  }
    msprof_tx_.store(false);
  if (record_op_args_.load() == true) {
    at_npu::native::AclopStopDumpArgs(ACL_OP_DUMP_OP_AICORE_ARGS);
    record_op_args_.store(false);
  }
}

void ProfilerMgr::Finalize() {
  if (npu_trace_.load() == true) {
    at_npu::native::AclProfilingFinalize();
  }
  npu_trace_.store(false);
}

void ProfilerMgr::StartDataReceiver()
{
    std::string fwk_path = path_ + "/FRAMEWORK";
    dataReceiver_.Init(fwk_path, capacity_);
    dataReceiver_.Start();
    traceDataReceiver_.Init(fwk_path, trace_capacity_);
    traceDataReceiver_.Start();
    dataReceiverWithLock_.Init(fwk_path, capacity_);
    dataReceiverWithLock_.Start();
}

void ProfilerMgr::StopDataReceiver()
{
    dataReceiver_.Stop();
    dataReceiver_.UnInit();
    traceDataReceiver_.Stop();
    traceDataReceiver_.UnInit();
    dataReceiverWithLock_.Stop();
    dataReceiverWithLock_.UnInit();
}

void ProfilerMgr::Upload(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data)
{
    dataReceiver_.Report(std::move(data));
}

void ProfilerMgr::UploadWithLock(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data)
{
    std::lock_guard<std::mutex> lock(reportDataMutex_);
    dataReceiverWithLock_.Report(std::move(data));
}

void ProfilerMgr::UploadTraceEventData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerFuncData> data)
{
    traceDataReceiver_.Report(std::move(data));
}

void ProfilerMgr::UploadTraceHashData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerHashData> data)
{
    traceDataReceiver_.ReportHash(std::move(data));
}

uint64_t ProfilerMgr::CheckFeatureConfig(uint64_t datatype_config)
{
    if (!FeatureMgr::GetInstance()->IsSupportFeature(FeatureType::FEATURE_ATTR)) {
        ASCEND_LOGW("Not support to set config for ATTR.");
        return datatype_config & ~(ACL_PROF_OP_ATTR);
    }
    return datatype_config;
}
} // profiler
} // torch_npu
