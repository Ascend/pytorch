#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/profiler/cann_profiling.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace torch_npu {
namespace profiler {

NpuProfiling& NpuProfiling::Instance() {
  static NpuProfiling npuProfiling;
  return npuProfiling;
}

void NpuProfiling::Init(const std::string &path) {
  TORCH_CHECK(status == PROFILING_FINALIZE, "init current profile status is: ", status, " error!");
  c10_npu::npuSynchronizeDevice();
  auto ret = at_npu::native::AclProfilingInit(path.c_str(), path.length());
  if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
    ASCEND_LOGE("npu AclProfInit fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
  status = PROFILING_INIT;
}

void NpuProfiling::Start(uint64_t npu_event, uint64_t aicore_metrics) {
  TORCH_CHECK(status == PROFILING_INIT || status == PROFILING_STOP,
      "start current profile status is: ", status, " error!");
  int deviceIndex = 0;
  aclError ret = c10_npu::GetDevice(&deviceIndex);
  if (ret) {
    ASCEND_LOGE("npu profiling aclrtGetDevice fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    status = PROFILING_FINALIZE;
    return;
  }
  const uint32_t deviceNum = 1;
  uint32_t deviceIdList[deviceNum] = {deviceIndex};
  profCfg = at_npu::native::AclProfilingCreateConfig(
      deviceIdList,
      deviceNum,
      (aclprofAicoreMetrics)aicore_metrics,
      nullptr,
      npu_event);
  if (profCfg == nullptr) {
    ASCEND_LOGE("npu profiling profiling_create_config fail, error  profCfg is null.");
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    status = PROFILING_FINALIZE;
    return;
  }
  c10_npu::npuSynchronizeDevice();
  ret = at_npu::native::AclProfilingStart(profCfg);
  if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
    ASCEND_LOGE("npu profiling AclProfStart fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_START;
}

void NpuProfiling::Stop() {
  TORCH_CHECK(status == PROFILING_START, "stop current profile status is: ", status, " error!");
  c10_npu::npuSynchronizeDevice();
  auto ret = at_npu::native::AclProfilingStop(profCfg);
  if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
    ASCEND_LOGE("npu AclProfStop fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_STOP;
}

void NpuProfiling::Finalize() {
  if (profCfg != nullptr) {
    if (status != PROFILING_STOP) {
      ASCEND_LOGW("finalize current profile status ( %u ) is not stopped, and call stop now.", status);
      auto ret = at_npu::native::AclProfilingStop(profCfg);
      if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
        ASCEND_LOGE("npu AclProfStop fail, error code: %d", ret);
        C10_NPU_SHOW_ERR_MSG();
      }
    }
    auto ret = at_npu::native::AclProfilingDestroyConfig(profCfg);
    if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
      ASCEND_LOGE("npu AclProfDestoryConfig fail, error code: %d", ret);
      C10_NPU_SHOW_ERR_MSG();
    }
    profCfg = nullptr;
  }
  auto ret = at_npu::native::AclProfilingFinalize();
  if (ret && (ret != ACL_ERROR_PROF_ALREADY_RUN)) {
    ASCEND_LOGE("npu AclProfFinalize fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_FINALIZE;
}


NpuProfilingDispatch& NpuProfilingDispatch::Instance() {
  static NpuProfilingDispatch npuProfilingDispatch;
  return npuProfilingDispatch;
}

void NpuProfilingDispatch::init() {
    profStepInfo = at_npu::native::init_stepinfo();
}

void NpuProfilingDispatch::start() {
  this->init();
  auto stream = c10_npu::getCurrentNPUStream();
  auto ret = at_npu::native::start_deliver_op(
      profStepInfo,
      aclprofStepTag::ACL_STEP_START,
      stream);
  if (ret != ACL_ERROR_NONE) {
      ASCEND_LOGE("npu profiling start fail, error code: %d", ret);
      C10_NPU_SHOW_ERR_MSG();
  }
}

void NpuProfilingDispatch::stop() {
  auto stream = c10_npu::getCurrentNPUStream();
  auto ret = at_npu::native::stop_deliver_op(
      profStepInfo,
      aclprofStepTag::ACL_STEP_END,
      stream);
  if (ret != ACL_ERROR_NONE) {
      ASCEND_LOGE("npu profiling stop fail, error code: %d", ret);
      C10_NPU_SHOW_ERR_MSG();
  }
  this->destroy();
}

void NpuProfilingDispatch::destroy() {
  if (profStepInfo != nullptr) {
    at_npu::native::destroy_stepinfo(profStepInfo);
  }
}

}
}
