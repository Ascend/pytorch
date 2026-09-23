#include "hardware/ascend/res_manager/ascend_hal_manager.h"

#include <unistd.h>
#include <fstream>
#include <string>
#include "common/common.h"
#include "acl/acl_rt.h"

#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt {
namespace device {
namespace ascend {
namespace {
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";
} // namespace
static thread_local aclrtContext threadLocalRtContext{nullptr};

AscendHalManager AscendHalManager::instance_{};
AscendHalManager& AscendHalManager::GetInstance() {
  return instance_;
}

void AscendHalManager::InitDevice(uint32_t deviceId) {
  RT_VLOG(VL_HARDWARE) << "Enter SetRtDevice, current initialize device number:" << initializedDeviceSet_.size();
  if (initializedDeviceSet_.find(deviceId) != initializedDeviceSet_.end()) {
    RT_VLOG(VL_HARDWARE) << "Device " << deviceId << " has been set";
    return;
  }

  auto ret = CALL_ASCEND_API(aclrtSetDevice, UintToInt(deviceId));
  if (ret != ACL_SUCCESS) {
    auto deviceCount = GetDeviceCount();
    RT_GLOG(ERROR) << "Call aclrtSetDevice failed, ret[" << static_cast<int>(ret) << "]. Got device count["
                   << deviceCount << "] and device id[" << deviceId << "], please check if device id is valid.";
  }

  aclrtContext rtContext;
  ret = CALL_ASCEND_API(aclrtGetCurrentContext, &rtContext);
  if (ret != ACL_SUCCESS || rtContext == nullptr) {
    RT_GLOG(ERROR) << "Call aclrtGetCurrentContext failed, ret[" << ret << "]";
    return;
  }

  defaultDeviceContextMap_[deviceId] = rtContext;
  (void)initializedDeviceSet_.insert(deviceId);
}

void AscendHalManager::ResetDevice(uint32_t deviceId) {
  if (initializedDeviceSet_.find(deviceId) != initializedDeviceSet_.end()) {
    auto ret = CALL_ASCEND_API(aclrtResetDevice, UintToInt(deviceId));
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "Call aclrtResetDevice, ret[" << ret << "]";
    }
    defaultDeviceContextMap_[deviceId] = nullptr;
    (void)initializedDeviceSet_.erase(deviceId);
  }
}

uint32_t AscendHalManager::GetDeviceCount() {
  uint32_t deviceCount = 0;
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &deviceCount);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }
  return deviceCount;
}

void AscendHalManager::SetDeviceSatMode(const aclrtFloatOverflowMode& overflowMode) {
  auto overflowModeStr =
      (overflowMode == aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION) ? kSaturationMode : kINFNANMode;
  RT_VLOG(VL_HARDWARE) << "The current overflow detection mode is " << overflowModeStr << ".";
  auto ret = CALL_ASCEND_API(aclrtSetDeviceSatMode, overflowMode);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Set " << overflowModeStr << " mode failed.";
  }
}

void AscendHalManager::SetOpWaitTimeout(uint32_t opWaitTimeout) {
  RT_VLOG(VL_HARDWARE) << "Set op wait timeout: " << opWaitTimeout << " s";
  auto aclRet = CALL_ASCEND_API(aclrtSetOpWaitTimeout, opWaitTimeout);
  if (aclRet != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Set op wait timeout failed, error: " << aclRet;
  }
}

void AscendHalManager::SetOpExecuteTimeOut(uint32_t opExecuteTimeout) {
  RT_VLOG(VL_HARDWARE) << "Set op execute timeout: " << opExecuteTimeout << " s";
  auto aclRet = CALL_ASCEND_API(aclrtSetOpExecuteTimeOut, opExecuteTimeout);
  if (aclRet != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Set op execute timeout failed, error: " << aclRet;
  }
}

aclrtContext AscendHalManager::CreateContext(uint32_t deviceId) {
  aclrtContext rtContext;
  auto ret = CALL_ASCEND_API(aclrtCreateContext, &rtContext, deviceId);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Call aclrtCreateContext failed, ret: " << ret;
  }
  rtContexts_.insert(rtContext);
  return rtContext;
}

void AscendHalManager::ResetContext(uint32_t deviceId) {
  aclrtContext rtContext = CreateContext(deviceId);
  defaultDeviceContextMap_[deviceId] = rtContext;
}

void AscendHalManager::DestroyContext(aclrtContext context) {
  auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Failed to destroy context, ret = " << ret << ".";
  }
  rtContexts_.erase(context);
}

void AscendHalManager::DestroyAllContext() {
  for (auto context : rtContexts_) {
    auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
    if (ret != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "Failed to destroy context, ret = " << ret << ".";
    }
  }
  rtContexts_.clear();
}

void AscendHalManager::SetContextForce(uint32_t deviceId) {
  if (defaultDeviceContextMap_[deviceId] == nullptr) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, defaultDeviceContextMap_[deviceId]);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
}

void AscendHalManager::SetContext(uint32_t deviceId) {
  if (defaultDeviceContextMap_[deviceId] == nullptr) {
    return;
  }
  if (threadLocalRtContext == defaultDeviceContextMap_[deviceId]) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, defaultDeviceContextMap_[deviceId]);
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
  threadLocalRtContext = defaultDeviceContextMap_[deviceId];
}

void AscendHalManager::InitializeAcl() {}

bool AscendHalManager::EnableLccl() {
  return false;
}
} // namespace ascend
} // namespace device
} // namespace fxrt
