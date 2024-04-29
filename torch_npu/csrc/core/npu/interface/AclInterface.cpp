#include "AclInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

namespace acl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)           \
  GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclrtCreateEventWithFlag)
LOAD_FUNCTION(aclrtCreateEventExWithFlag)
LOAD_FUNCTION(aclrtQueryEventWaitStatus)
LOAD_FUNCTION(aclrtQueryEventStatus)
LOAD_FUNCTION(aclprofCreateStepInfo)
LOAD_FUNCTION(aclprofGetStepTimestamp)
LOAD_FUNCTION(aclprofDestroyStepInfo)
LOAD_FUNCTION(aclprofInit)
LOAD_FUNCTION(aclprofStart)
LOAD_FUNCTION(aclprofStop)
LOAD_FUNCTION(aclprofFinalize)
LOAD_FUNCTION(aclprofCreateConfig)
LOAD_FUNCTION(aclprofDestroyConfig)
LOAD_FUNCTION(aclrtGetSocName)
LOAD_FUNCTION(aclrtCreateStream)
LOAD_FUNCTION(aclrtSetStreamFailureMode)
LOAD_FUNCTION(aclrtSetOpWaitTimeout)
LOAD_FUNCTION(aclrtCreateStreamWithConfig)
LOAD_FUNCTION(aclrtSetDeviceSatMode)
LOAD_FUNCTION(aclrtSetOpExecuteTimeOut)
LOAD_FUNCTION(aclrtSetStreamOverflowSwitch)
LOAD_FUNCTION(aclrtGetStreamOverflowSwitch)
LOAD_FUNCTION(aclrtSynchronizeStreamWithTimeout)
LOAD_FUNCTION(aclrtDestroyStreamForce)
LOAD_FUNCTION(aclrtGetDeviceUtilizationRate)
LOAD_FUNCTION(aclrtMallocAlign32)
LOAD_FUNCTION(aclrtDeviceCanAccessPeer)
LOAD_FUNCTION(aclrtSynchronizeStream)
LOAD_FUNCTION(aclrtStreamQuery)
LOAD_FUNCTION(aclrtReserveMemAddress)
LOAD_FUNCTION(aclrtReleaseMemAddress)
LOAD_FUNCTION(aclrtMallocPhysical)
LOAD_FUNCTION(aclrtFreePhysical)
LOAD_FUNCTION(aclrtMapMem)
LOAD_FUNCTION(aclrtUnmapMem)
LOAD_FUNCTION(aclGetCannAttributeList)
LOAD_FUNCTION(aclGetCannAttribute)
LOAD_FUNCTION(aclGetDeviceCapability)

aclprofStepInfoPtr init_stepinfo() {
  typedef aclprofStepInfoPtr(*npdInitFunc)();
  static npdInitFunc func = nullptr;
  if (func == nullptr) {
      func = (npdInitFunc)GET_FUNC(aclprofCreateStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStepInfo", PROF_ERROR(ErrCode::NOT_FOUND));
  auto ret = func();
  return ret;
}

NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo) {
  typedef NpdStatus(*npdDestroyFunc)(aclprofStepInfoPtr);
  static npdDestroyFunc func = nullptr;
  if (func == nullptr) {
      func = (npdDestroyFunc)GET_FUNC(aclprofDestroyStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStepInfo", PROF_ERROR(ErrCode::NOT_FOUND));
  auto ret = func(stepInfo);
  return ret;
}

NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream) {
  typedef NpdStatus(*npdStartProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStartProfiling func = nullptr;
  if (func == nullptr) {
      func = (npdStartProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp", PROF_ERROR(ErrCode::NOT_FOUND));
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream) {
  typedef NpdStatus(*npdStopProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStopProfiling func = nullptr;
  if (func == nullptr) {
      func = (npdStopProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp", PROF_ERROR(ErrCode::NOT_FOUND));
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

const char *AclGetErrMsg()
{
  typedef const char *(*aclGetErrMsg)();
  static aclGetErrMsg func = nullptr;
  if (func == nullptr) {
    func = (aclGetErrMsg)GET_FUNC(aclGetRecentErrMsg);
  }
  if (func != nullptr) {
    auto res = func();
    return res != nullptr ? res : "";
  }
  return "";
}

aclError AclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  typedef aclError(*aclrtCreateStreamWithConfigFunc)(aclrtStream*, uint32_t, uint32_t);
  static aclrtCreateStreamWithConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (aclrtCreateStreamWithConfigFunc)GET_FUNC(aclrtCreateStreamWithConfig);
  }

  aclError ret;
  if (func != nullptr) {
    ret = func(stream, priority, flag);
  } else {
    ret = aclrtCreateStream(stream);
  }
  if (ret == ACL_SUCCESS && stream != nullptr) {
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuStreamCreation(reinterpret_cast<uintptr_t>(*stream));
    }
#endif
    if (!c10_npu::IsSupportInfNan()) {
      TORCH_CHECK(AclrtSetStreamOverflowSwitch(*stream, 1) == ACL_SUCCESS, "SET StreamOverflowSwitch Failed.", PROF_ERROR(ErrCode::ACL));
    }
    return AclrtSetStreamFailureMode(*stream, ACL_STOP_ON_FAILURE);
  } else {
    return ret;
  }
}

aclError AclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) {
  if (stream == nullptr) { // default stream
    return ACL_ERROR_INVALID_PARAM;
  }
  typedef aclError(*aclrtSetStreamFailureModeFunc)(aclrtStream, uint64_t);
  static aclrtSetStreamFailureModeFunc func = (aclrtSetStreamFailureModeFunc)GET_FUNC(aclrtSetStreamFailureMode);
  if (func == nullptr) {
    return ACL_SUCCESS;
  }
  return func(stream, mode);
}

aclError AclrtSetOpWaitTimeout(uint32_t timeout) {
  typedef aclError(*aclrtSetOpWaitTimeoutFunc)(uint32_t);
  static aclrtSetOpWaitTimeoutFunc func = nullptr;
  if (func == nullptr) {
    func = (aclrtSetOpWaitTimeoutFunc)GET_FUNC(aclrtSetOpWaitTimeout);
  }
  TORCH_CHECK(func, "Failed to find function aclrtSetOpWaitTimeout", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(timeout);
}

bool IsExistCreateEventExWithFlag()
{
    typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
    static AclrtCreateEventWithFlagFunc func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventExWithFlag);
    return func != nullptr;
}

aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
    // Recommend aclrtCreateEventExWithFlag.
    // Differences from aclrtCreateEventWithFlag:
    //   1. Event can be reused naturally, aclrtResetEvent is not supported.
    //   2. There is no limit on the number of events.
    //   3. Only support query event record status, aclrtQueryEvent and aclrtQueryEventWaitStatus are not supported.
    //   4. aclrtDestroyEvent change to asynchronous destroy event.
    static AclrtCreateEventWithFlagFunc func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventExWithFlag);
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtCreateEventExWithFlag");
        func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(event, flag);
}

aclError AclQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *waitStatus)
{
  typedef aclError (*aclQueryEventWaitStatus)(aclrtEvent event, aclrtEventWaitStatus *waitStatus);
  static aclQueryEventWaitStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventWaitStatus)GET_FUNC(aclrtQueryEventWaitStatus);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtQueryEventWaitStatus", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(event, waitStatus);
  }

aclError AclQueryEventRecordedStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
  typedef aclError (*aclQueryEventStatus)(aclrtEvent event, aclrtEventRecordedStatus *status);
  static aclQueryEventStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventStatus)GET_FUNC(aclrtQueryEventStatus);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtQueryEventStatus", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(event, status);
}

bool IsExistQueryEventRecordedStatus()
{
  typedef aclError (*aclQueryEventStatus)(aclrtEvent event, aclrtEventRecordedStatus *status);
  static aclQueryEventStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventStatus)GET_FUNC(aclrtQueryEventStatus);
  }
  if (func != nullptr) {
    return true;
  } else {
    return false;
  }
}

aclError AclProfilingInit(const char *profilerResultPath, size_t length) {
  typedef aclError (*AclProfInitFunc) (const char *, size_t);
  static AclProfInitFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfInitFunc)GET_FUNC(aclprofInit);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofInit", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(profilerResultPath, length);
}

aclError AclProfilingStart(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStartFunc) (const aclprofConfig *);
  static AclProfStartFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStartFunc)GET_FUNC(aclprofStart);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStart", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(profilerConfig);
}

aclError AclProfilingStop(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStopFunc) (const aclprofConfig*);
  static AclProfStopFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStopFunc)GET_FUNC(aclprofStop);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStop", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(profilerConfig);
}

aclError AclProfilingFinalize() {
  typedef aclError (*AclProfFinalizeFunc) ();
  static AclProfFinalizeFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfFinalizeFunc)GET_FUNC(aclprofFinalize);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofFinalize", PROF_ERROR(ErrCode::NOT_FOUND));
  return func();
}

aclprofConfig *AclProfilingCreateConfig(
    uint32_t *deviceIdList,
    uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics,
    aclprofAicoreEvents *aicoreEvents,
    uint64_t dataTypeConfig) {
  typedef aclprofConfig *(*AclProfCreateConfigFunc) \
    (uint32_t *, uint32_t, aclprofAicoreMetrics, aclprofAicoreEvents *, uint64_t);
  static AclProfCreateConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfCreateConfigFunc)GET_FUNC(aclprofCreateConfig);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateConfig", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(deviceIdList, deviceNums, aicoreMetrics, aicoreEvents, dataTypeConfig);
}

aclError AclProfilingDestroyConfig(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfDestroyConfigFunc) (const aclprofConfig *);
  static AclProfDestroyConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfDestroyConfigFunc)GET_FUNC(aclprofDestroyConfig);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyConfig", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(profilerConfig);
}

const char *AclrtGetSocName() {
  typedef const char *(*aclrtGetSocNameFunc)();
  static aclrtGetSocNameFunc func = nullptr;
  if (func == nullptr) {
    func = (aclrtGetSocNameFunc)GET_FUNC(aclrtGetSocName);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtGetSocName", PROF_ERROR(ErrCode::NOT_FOUND));
  return func();
}

const char *AclGetSocName() {
  typedef const char *(*AclGetSoc) ();
  static AclGetSoc func = nullptr;
  if (func == nullptr) {
    func = (AclGetSoc)GET_FUNC(aclrtGetSocName);
  }
  if (func == nullptr) {
    return nullptr;
  }
  return func();
}

aclError AclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
  typedef aclError (*AclrtSetDeviceSatMode)(aclrtFloatOverflowMode mode);
  static AclrtSetDeviceSatMode func = nullptr;
  if (func == nullptr) {
    func = (AclrtSetDeviceSatMode)GET_FUNC(aclrtSetDeviceSatMode);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtSetDeviceSatMode", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(mode);
}

aclError AclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag) {
  typedef aclError (*AclrtSetStreamOverflowSwitch)(aclrtStream, uint32_t);
  static AclrtSetStreamOverflowSwitch func = nullptr;
  if (func == nullptr) {
    func = (AclrtSetStreamOverflowSwitch)GET_FUNC(aclrtSetStreamOverflowSwitch);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtSetStreamOverflowSwitch", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(stream, flag);
}

aclError AclrtSetOpExecuteTimeOut(uint32_t timeout) {
  typedef aclError (*AclrtSetOpExecuteTimeOut)(uint32_t);
  static AclrtSetOpExecuteTimeOut func = nullptr;
  if (func == nullptr) {
    func = (AclrtSetOpExecuteTimeOut)GET_FUNC(aclrtSetOpExecuteTimeOut);
  }
  if (func == nullptr) {
    return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
  }
  return func(timeout);
}

aclError AclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag) {
  typedef aclError (*AclrtGetStreamOverflowSwitch)(aclrtStream, uint32_t*);
  static AclrtGetStreamOverflowSwitch func = nullptr;
  if (func == nullptr) {
    func = (AclrtGetStreamOverflowSwitch)GET_FUNC(aclrtGetStreamOverflowSwitch);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtGetStreamOverflowSwitch", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(stream, flag);
}

aclError AclrtSynchronizeStreamWithTimeout(aclrtStream stream) {
  if (C10_UNLIKELY(
      c10_npu::warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    c10_npu::warn_or_error_on_sync();
  }
#ifndef BUILD_LIBTORCH
  const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
  if (C10_UNLIKELY(trigger)) {
      trigger->traceNpuStreamSynchronization(reinterpret_cast<uintptr_t>(stream));
  }
#endif
  typedef aclError (*AclrtSynchronizeStreamWithTimeout)(aclrtStream, int32_t);
  static AclrtSynchronizeStreamWithTimeout func = (AclrtSynchronizeStreamWithTimeout)GET_FUNC(aclrtSynchronizeStreamWithTimeout);
  int32_t timeout = c10_npu::option::OptionsManager::GetACLExecTimeout();
  if (func != nullptr) {
    return func(stream, timeout);
  } else {
    TORCH_NPU_WARN_ONCE(func, "Failed to find function", "aclrtSynchronizeStreamWithTimeout");
    typedef aclError (*AclrtSynchronizeStream)(aclrtStream);
    static AclrtSynchronizeStream func_backup = nullptr;
    if (func_backup == nullptr) {
      func_backup = (AclrtSynchronizeStream)GET_FUNC(aclrtSynchronizeStream);
    }
    TORCH_CHECK(func_backup, "Failed to find function", "aclrtSynchronizeStreamWithTimeout and aclrtSynchronizeStream", PROF_ERROR(ErrCode::NOT_FOUND));
    return func_backup(stream);
  }
}

aclError AclrtDestroyStreamForce(aclrtStream stream) {
  typedef aclError (*AclrtDestroyStreamForce)(aclrtStream);
  static AclrtDestroyStreamForce func = (AclrtDestroyStreamForce)GET_FUNC(aclrtDestroyStreamForce);
  if (func != nullptr) {
    return func(stream);
  }
  TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtDestroyStreamForce");
  return aclrtDestroyStream(stream);
}

aclError AclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo) {
  typedef aclError (*AclrtGetDeviceUtilizationRate)(int32_t, aclrtUtilizationInfo*);
  static AclrtGetDeviceUtilizationRate func = nullptr;
  if (func == nullptr) {
    func = (AclrtGetDeviceUtilizationRate)GET_FUNC(aclrtGetDeviceUtilizationRate);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtGetDeviceUtilizationRate", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(deviceId, utilizationInfo);
}

aclError AclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  typedef aclError (*AclrtMallocAlign32)(void**, size_t, aclrtMemMallocPolicy);
  static AclrtMallocAlign32 func = (AclrtMallocAlign32)GET_FUNC(aclrtMallocAlign32);
  if (func != nullptr) {
    return func(devPtr, size, policy);
  }
  TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtMallocAlign32");
  return aclrtMalloc(devPtr, size, policy);
}

aclError AclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) {
  typedef aclError (*AclrtStreamQuery)(aclrtStream, aclrtStreamStatus*);
  static AclrtStreamQuery func = nullptr;
  if (func == nullptr) {
    func = (AclrtStreamQuery)GET_FUNC(aclrtStreamQuery);
  }
  TORCH_CHECK(func, "Failed to find function aclrtStreamQuery, Please upgrade CANN version.", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(stream, status);
}

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id) {
  int32_t can_access_peer = 0;
  c10::DeviceIndex num_npus = c10_npu::device_count();
  TORCH_CHECK(device_id >= 0 && device_id < num_npus, PROF_ERROR(ErrCode::VALUE));
  TORCH_CHECK(peer_device_id >= 0 && peer_device_id < num_npus, PROF_ERROR(ErrCode::VALUE));
  // To maintain consistency with cuda, returns false when deviceid and peerdeviceid are equal.
  if (device_id == peer_device_id) {
    return false;
  }
  typedef aclError (*AclrtDeviceCanAccessPeer)(int32_t*, int32_t, int32_t);
  static AclrtDeviceCanAccessPeer func = nullptr;
  if (func == nullptr) {
    func = (AclrtDeviceCanAccessPeer)GET_FUNC(aclrtDeviceCanAccessPeer);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtDeviceCanAccessPeer", PROF_ERROR(ErrCode::NOT_FOUND));
  NPU_CHECK_ERROR(func(&can_access_peer, device_id, peer_device_id), "aclrtDeviceCanAccessPeer");
  return can_access_peer != 0;
}

aclError AclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags) {
  typedef aclError (*AclrtReserveMemAddress)(void**, size_t, size_t, void*, uint64_t);
  static AclrtReserveMemAddress func = nullptr;
  if (func == nullptr) {
    func = (AclrtReserveMemAddress)GET_FUNC(aclrtReserveMemAddress);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtReserveMemAddress", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(virPtr, size, alignment, expectPtr, flags);
}

aclError AclrtReleaseMemAddress(void *virPtr) {
  typedef aclError (*AclrtReleaseMemAddress)(void*);
  static AclrtReleaseMemAddress func = nullptr;
  if (func == nullptr) {
    func = (AclrtReleaseMemAddress)GET_FUNC(aclrtReleaseMemAddress);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtReleaseMemAddress", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(virPtr);
}

aclError AclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop,
    uint64_t flags) {
  typedef aclError (*AclrtMallocPhysical)(aclrtDrvMemHandle*, size_t, const aclrtPhysicalMemProp*, uint64_t);
  static AclrtMallocPhysical func = nullptr;
  if (func == nullptr) {
    func = (AclrtMallocPhysical)GET_FUNC(aclrtMallocPhysical);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtMallocPhysical", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(handle, size, prop, flags);
}

aclError AclrtFreePhysical(aclrtDrvMemHandle handle) {
  typedef aclError (*AclrtFreePhysical)(aclrtDrvMemHandle);
  static AclrtFreePhysical func = nullptr;
  if (func == nullptr) {
    func = (AclrtFreePhysical)GET_FUNC(aclrtFreePhysical);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtFreePhysical", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(handle);
}

aclError AclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
  typedef aclError (*AclrtMapMem)(void*, size_t, size_t, aclrtDrvMemHandle, uint64_t);
  static AclrtMapMem func = nullptr;
  if (func == nullptr) {
    func = (AclrtMapMem)GET_FUNC(aclrtMapMem);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtMapMem", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(virPtr, size, offset, handle, flags);
}

aclError AclrtUnmapMem(void *virPtr) {
  typedef aclError (*AclrtUnmapMem)(void*);
  static AclrtUnmapMem func = nullptr;
  if (func == nullptr) {
    func = (AclrtUnmapMem)GET_FUNC(aclrtUnmapMem);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtUnmapMem", PROF_ERROR(ErrCode::NOT_FOUND));
  return func(virPtr);
}

bool IsExistGetCannAttribute()
{
    typedef aclError (*AclGetCannAttribute)(aclCannAttr, int32_t *);
    static AclGetCannAttribute func = (AclGetCannAttribute) GET_FUNC(aclGetCannAttribute);
    return func != nullptr;
}

aclError AclGetCannAttributeList(const aclCannAttr **cannAttrList, size_t *num)
{
    typedef aclError (*AclGetCannAttributeList)(const aclCannAttr **, size_t *);
    static AclGetCannAttributeList func = nullptr;
    if (func == nullptr) {
        func = (AclGetCannAttributeList) GET_FUNC(aclGetCannAttributeList);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclGetCannAttributeList", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(cannAttrList, num);
}

aclError AclGetCannAttribute(aclCannAttr cannAttr, int32_t *value)
{
    typedef aclError (*AclGetCannAttribute)(aclCannAttr, int32_t *);
    static AclGetCannAttribute func = nullptr;
    if (func == nullptr) {
        func = (AclGetCannAttribute) GET_FUNC(aclGetCannAttribute);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclGetCannAttribute", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(cannAttr, value);
}

aclError AclGetDeviceCapability(uint32_t deviceId, aclDeviceInfo deviceInfo, int64_t *value)
{
    typedef aclError (*AclGetDeviceCapability)(uint32_t, aclDeviceInfo, int64_t *);
    static AclGetDeviceCapability func = nullptr;
    if (func == nullptr) {
        func = (AclGetDeviceCapability) GET_FUNC(aclGetDeviceCapability);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclGetDeviceCapability", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, deviceInfo, value);
}

} // namespace acl
} // namespace c10
