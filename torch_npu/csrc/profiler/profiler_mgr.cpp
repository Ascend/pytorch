#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
namespace torch_npu {
namespace profiler {

uint64_t dataTypeConfigCann = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_HCCL_TRACE | ACL_PROF_TRAINING_TRACE | ACL_PROF_RUNTIME_API;
std::map<int32_t, uint64_t> ProfilerMgr::dataTypeConfigMap_ = {
  {static_cast<int32_t>(ProfLevel::MSPROF_TRACE_NPU), dataTypeConfigCann},
};

ProfilerMgr::ProfilerMgr()
  : report_enable_(false),
    npu_trace_(false),
    profConfig_(nullptr) {}

void ProfilerMgr::Init(const std::string &path, bool npu_trace) {
  c10_npu::npuSynchronizeDevice();
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

void ProfilerMgr::Start(ProfLevel level, bool cpu_trace) {
  if (npu_trace_.load() == true) {
    aclprofAicoreMetrics aicMetrics = ACL_AICORE_PIPE_UTILIZATION;
    int32_t deviceId = 0;
    auto ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("Get Device ID failed.");
      return;
    }
    const uint32_t deviceNum = 1;
    uint32_t deviceIdList[deviceNum] = {deviceId};
    switch (level) {
      case ProfLevel::MSPROF_TRACE_NPU:
        EnableMsProfiler(deviceIdList, deviceNum, aicMetrics, dataTypeConfigMap_[static_cast<int32_t>(level)]);
        break;
      default:
        break;
    }
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
