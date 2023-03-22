#include "AclInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "c10/util/Exception.h"

namespace c10_npu {

namespace acl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclrtCreateEventWithFlag)
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
LOAD_FUNCTION(aclrtSetStreamOverflowSwitch)
LOAD_FUNCTION(aclrtGetStreamOverflowSwitch)

aclprofStepInfoPtr init_stepinfo(){
  typedef aclprofStepInfoPtr(*npdInitFunc)();
  static npdInitFunc func = nullptr;
  if(func == nullptr){
      func = (npdInitFunc)GET_FUNC(aclprofCreateStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStepInfo");
  auto ret = func();
  return ret;
}

NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo){
  typedef NpdStatus(*npdDestroyFunc)(aclprofStepInfoPtr);
  static npdDestroyFunc func = nullptr;
  if(func == nullptr){
      func = (npdDestroyFunc)GET_FUNC(aclprofDestroyStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStepInfo");
  auto ret = func(stepInfo);
  return ret;
}

NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream){
  typedef NpdStatus(*npdStartProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStartProfiling func = nullptr;
  if(func == nullptr){
      func = (npdStartProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream){
  typedef NpdStatus(*npdStopProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStopProfiling func = nullptr;
  if(func == nullptr){
      func = (npdStopProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
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
  TORCH_CHECK(func, "Failed to find function aclrtSetOpWaitTimeout");
  return func(timeout);
}

aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
  typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
  static AclrtCreateEventWithFlagFunc func = nullptr;
  if (func == nullptr) {
    func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag");
  return func(event, flag);
}

aclError AclQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *waitStatus)
{
  typedef aclError (*aclQueryEventWaitStatus)(aclrtEvent event, aclrtEventWaitStatus *waitStatus);
  static aclQueryEventWaitStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventWaitStatus)GET_FUNC(aclrtQueryEventWaitStatus);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtQueryEventWaitStatus");
  return func(event, waitStatus);
  }

aclError AclQueryEventRecordedStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
  typedef aclError (*aclQueryEventStatus)(aclrtEvent event, aclrtEventRecordedStatus *status);
  static aclQueryEventStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventStatus)GET_FUNC(aclrtQueryEventStatus);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtQueryEventStatus");
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
  TORCH_CHECK(func, "Failed to find function ", "aclprofInit");
  return func(profilerResultPath, length);
}

aclError AclProfilingStart(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStartFunc) (const aclprofConfig *);
  static AclProfStartFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStartFunc)GET_FUNC(aclprofStart);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStart");
  return func(profilerConfig);
}

aclError AclProfilingStop(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStopFunc) (const aclprofConfig*);
  static AclProfStopFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStopFunc)GET_FUNC(aclprofStop);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStop");
  return func(profilerConfig);
}

aclError AclProfilingFinalize() {
  typedef aclError (*AclProfFinalizeFunc) ();
  static AclProfFinalizeFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfFinalizeFunc)GET_FUNC(aclprofFinalize);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofFinalize");
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
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateConfig");
  return func(deviceIdList, deviceNums, aicoreMetrics, aicoreEvents, dataTypeConfig);
}

aclError AclProfilingDestroyConfig(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfDestroyConfigFunc) (const aclprofConfig *);
  static AclProfDestroyConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfDestroyConfigFunc)GET_FUNC(aclprofDestroyConfig);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyConfig");
  return func(profilerConfig);
}

const char *AclrtGetSocName() {
  typedef const char *(*aclrtGetSocNameFunc)();
  static aclrtGetSocNameFunc func = nullptr;
  if (func == nullptr) {
    func = (aclrtGetSocNameFunc)GET_FUNC(aclrtGetSocName);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtGetSocName");
  return func();
}

const char *AclGetSocName() {
  typedef const char * (*AclGetSoc) ();
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
  TORCH_CHECK(func, "Failed to find function ", "aclrtSetDeviceSatMode");
  return func(mode);
}

aclError AclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag) {
  typedef aclError (*AclrtSetStreamOverflowSwitch)(aclrtStream, uint32_t);
  static AclrtSetStreamOverflowSwitch func = nullptr;
  if (func == nullptr) {
    func = (AclrtSetStreamOverflowSwitch)GET_FUNC(aclrtSetStreamOverflowSwitch);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtSetStreamOverflowSwitch");
  return func(stream, flag);
}

aclError AclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag) {
  typedef aclError (*AclrtGetStreamOverflowSwitch)(aclrtStream, uint32_t*);
  static AclrtGetStreamOverflowSwitch func = nullptr;
  if (func == nullptr) {
    func = (AclrtGetStreamOverflowSwitch)GET_FUNC(aclrtGetStreamOverflowSwitch);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtGetStreamOverflowSwitch");
  return func(stream, flag);
}

} // namespace acl
} // namespace c10
