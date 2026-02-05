#include "AclInterface.h"
#include <unistd.h>
#include <dlfcn.h>
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
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
LOAD_FUNCTION(aclrtIpcGetEventHandle)
LOAD_FUNCTION(aclrtIpcOpenEventHandle)
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
LOAD_FUNCTION(aclrtGetMemUceInfo)
LOAD_FUNCTION(aclrtGetMemUsageInfo)
LOAD_FUNCTION(aclrtDeviceTaskAbort)
LOAD_FUNCTION(aclrtMemUceRepair)
LOAD_FUNCTION(aclrtCmoAsync)
LOAD_FUNCTION(aclrtGetLastError)
LOAD_FUNCTION(aclrtPeekAtLastError)
LOAD_FUNCTION(aclrtSynchronizeDevice)
LOAD_FUNCTION(aclrtSynchronizeDeviceWithTimeout)
LOAD_FUNCTION(aclrtEventGetTimestamp)
LOAD_FUNCTION(aclmdlRICaptureBegin)
LOAD_FUNCTION(aclmdlRICaptureGetInfo)
LOAD_FUNCTION(aclmdlRICaptureEnd)
LOAD_FUNCTION(aclmdlRIDebugPrint)
LOAD_FUNCTION(aclmdlRIExecuteAsync)
LOAD_FUNCTION(aclmdlRIDestroy)
LOAD_FUNCTION(aclsysGetCANNVersion)
LOAD_FUNCTION(aclsysGetVersionStr)
LOAD_FUNCTION(aclmdlRICaptureTaskGrpBegin)
LOAD_FUNCTION(aclmdlRICaptureTaskGrpEnd)
LOAD_FUNCTION(aclmdlRICaptureTaskUpdateBegin)
LOAD_FUNCTION(aclmdlRICaptureTaskUpdateEnd)
LOAD_FUNCTION(aclmdlRIDebugJsonPrint)
LOAD_FUNCTION(aclrtHostRegister)
LOAD_FUNCTION(aclrtHostUnregister)
LOAD_FUNCTION(aclrtIpcMemGetExportKey)
LOAD_FUNCTION(aclrtIpcMemSetImportPid)
LOAD_FUNCTION(aclrtIpcMemImportByKey)
LOAD_FUNCTION(aclrtIpcMemClose)
LOAD_FUNCTION(aclrtMemExportToShareableHandle)
LOAD_FUNCTION(aclrtMemSetPidToShareableHandle)
LOAD_FUNCTION(aclrtMemImportFromShareableHandle)
LOAD_FUNCTION(aclrtDeviceGetBareTgid)
LOAD_FUNCTION(aclrtGetDeviceResLimit)
LOAD_FUNCTION(aclrtSetDeviceResLimit)
LOAD_FUNCTION(aclrtResetDeviceResLimit)
LOAD_FUNCTION(aclrtStreamGetId)
LOAD_FUNCTION(aclrtLaunchCallback)
LOAD_FUNCTION(aclrtSubscribeReport)
LOAD_FUNCTION(aclrtUnSubscribeReport)
LOAD_FUNCTION(aclrtMemcpyBatch)
LOAD_FUNCTION(aclrtMemcpyBatchAsync)
LOAD_FUNCTION(aclrtMemcpyAsyncWithCondition)
LOAD_FUNCTION(aclrtSetStreamResLimit)
LOAD_FUNCTION(aclrtResetStreamResLimit)
LOAD_FUNCTION(aclrtGetStreamResLimit)
LOAD_FUNCTION(aclrtUseStreamResInCurrentThread)
LOAD_FUNCTION(aclrtUnuseStreamResInCurrentThread)
LOAD_FUNCTION(aclrtGetResInCurrentThread)
LOAD_FUNCTION(aclrtSetOpExecuteTimeOutV2)
LOAD_FUNCTION(aclrtPointerGetAttributes)
LOAD_FUNCTION(aclrtSetStreamAttribute)
LOAD_FUNCTION(aclrtDeviceGetUuid)
LOAD_FUNCTION(aclrtMallocHostWithCfg)

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
    static AclrtCreateEventWithFlagFunc func_ex = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventExWithFlag);
    if (func_ex == nullptr) {
        TORCH_NPU_WARN_ONCE(func_ex, "Failed to find function ", "aclrtCreateEventExWithFlag");
    }
    static AclrtCreateEventWithFlagFunc func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
    TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag", PROF_ERROR(ErrCode::NOT_FOUND));
    if ((flag == ACL_EVENT_EXTERNAL) || (func_ex == nullptr)) {
        return func(event, flag);
    }
    return func_ex(event, flag);
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

aclError AclIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle)
{
    typedef aclError (*aclIpcGetEventHandle)(aclrtEvent event, aclrtIpcEventHandle *handle);
    static aclIpcGetEventHandle func = nullptr;
    if (func == nullptr) {
        func = (aclIpcGetEventHandle)GET_FUNC(aclrtIpcGetEventHandle);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtIpcGetEventHandle", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(event, handle);
}

aclError AclIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event)
{
    typedef aclError (*aclIpcOpenEventHandle)(aclrtIpcEventHandle handle, aclrtEvent *event);
    static aclIpcOpenEventHandle func = nullptr;
    if (func == nullptr) {
        func = (aclIpcOpenEventHandle)GET_FUNC(aclrtIpcOpenEventHandle);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtIpcOpenEventHandle", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(handle, event);
}

bool IsSupportIpcEvent()
{
    const static bool is_support = []() -> bool {
        constexpr long supported_page_size = 4096;
        long size = sysconf(_SC_PAGE_SIZE);
        if (size != supported_page_size) {
            ASCEND_LOGD("IsSupportIpcEvent return false because page size %ld is not supported.", size);
            return false;
        }

        if (!IsExistCreateEventExWithFlag()) {
            ASCEND_LOGD("IsSupportIpcEvent return false because aclrtCreateEventExWithFlag does not exist.");
            return false;
        }

        auto func = GET_FUNC(aclrtIpcGetEventHandle);
        if (func == nullptr) {
            ASCEND_LOGD("IsSupportIpcEvent return false because aclrtIpcGetEventHandle does not exist.");
            return false;
        }

        c10::DeviceIndex device_index = c10_npu::current_device();
        c10_npu::LazySetDevice(device_index);

        aclrtEvent npu_event = nullptr;
        auto ret = c10_npu::acl::AclrtCreateEventWithFlag(&npu_event, ACL_EVENT_IPC);
        if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            ASCEND_LOGD("IsSupportIpcEvent return false because create event with flag ACL_EVENT_IPC failed.");
            return false;
        }
        NPU_CHECK_ERROR(ret);
        NPU_CHECK_ERROR(aclrtDestroyEvent(npu_event));

        ASCEND_LOGD("IsSupportIpcEvent return true.");
        return true;
    }();

    return is_support;
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
        (uint32_t *, uint32_t, aclprofAicoreMetrics, const aclprofAicoreEvents *, uint64_t);
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
    typedef aclError (*AclrtSetOpExecuteTimeOutV2)(uint64_t, uint64_t *);
    static AclrtSetOpExecuteTimeOutV2 funcV2 = nullptr;
    if (funcV2 == nullptr) {
        funcV2 = (AclrtSetOpExecuteTimeOutV2)GET_FUNC(aclrtSetOpExecuteTimeOutV2);
    }

    if (funcV2) {
        uint64_t value = static_cast<uint64_t>(timeout) * 1000 * 1000;
        uint64_t actualTimeout = 0;
        auto ret = funcV2(value, &actualTimeout);
        ASCEND_LOGI("AclrtSetOpExecuteTimeOutV2 set actual timeout: %zuus", static_cast<size_t>(actualTimeout));
        return ret;
    }

    typedef aclError (*AclrtSetOpExecuteTimeOut)(uint32_t);
    static AclrtSetOpExecuteTimeOut func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetOpExecuteTimeOut)GET_FUNC(aclrtSetOpExecuteTimeOut);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtSetOpExecuteTimeOut", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(timeout);
}

aclError AclrtSetOpExecuteTimeOutV2(uint64_t timeout)
{
    typedef aclError (*AclrtSetOpExecuteTimeOutV2)(uint64_t, uint64_t *);
    static AclrtSetOpExecuteTimeOutV2 func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetOpExecuteTimeOutV2)GET_FUNC(aclrtSetOpExecuteTimeOutV2);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtSetOpExecuteTimeOutV2", PTA_ERROR(ErrCode::NOT_FOUND));
    uint64_t actualTimeout = 0;
    auto ret = func(timeout, &actualTimeout);
    ASCEND_LOGI("AclrtSetOpExecuteTimeOutV2 set actual timeout: %zuus", static_cast<size_t>(actualTimeout));
    return ret;
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
    aclError ret;
    if (func != nullptr) {
        ret = func(devPtr, size, policy);
    } else {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtMallocAlign32");
        ret = aclrtMalloc(devPtr, size, policy);
    }

    if (ret != ACL_RT_SUCCESS && (policy == aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE1G_ONLY)) {
        TORCH_NPU_WARN_ONCE("The malloc 1G large-page physical memory failed, so malloc 2M page memory."
                            "Using the 2M memory page may result in performance degradation. "
                            "This warning occurs because the PYTORCH_NPU_ALLOC_CONF = page_size:1g configuration is "
                            "enabled, but the pre-allocated number of 1G large pages is insufficient or 1G large-page "
                            "memory pre-allocation is not enabled.");
        policy = aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST;
        if (func != nullptr) {
            ret = func(devPtr, size, policy);
        } else {
            TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtMallocAlign32");
            ret = aclrtMalloc(devPtr, size, policy);
        }
    }
    return ret;
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

aclError AclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags,
                                HcclComm hcclComm)
{
    typedef aclError (*AclrtReserveMemAddress)(void**, size_t, size_t, void*, uint64_t);
    static AclrtReserveMemAddress func = nullptr;
    if (func == nullptr) {
        func = (AclrtReserveMemAddress)GET_FUNC(aclrtReserveMemAddress);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtReserveMemAddress", PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(virPtr, size, alignment, expectPtr, flags);
    if (hcclComm) {
        HCCL_CHECK_ERROR(at_npu::hccl::HcclCommSetMemoryRangeFace(hcclComm, &virPtr, size, alignment, flags));
    }
    return ret;
}

aclError AclrtReleaseMemAddress(void *virPtr, HcclComm hcclComm)
{
    typedef aclError (*AclrtReleaseMemAddress)(void*);
    static AclrtReleaseMemAddress func = nullptr;
    if (func == nullptr) {
        func = (AclrtReleaseMemAddress)GET_FUNC(aclrtReleaseMemAddress);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtReleaseMemAddress", PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(virPtr);
    if (hcclComm) {
        HCCL_CHECK_ERROR(at_npu::hccl::HcclCommUnsetMemoryRangeFace(hcclComm, virPtr));
    }
    return ret;
}

aclError AclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop,
    uint64_t flags) {
    typedef aclError (*AclrtMallocPhysical)(aclrtDrvMemHandle*, size_t, const aclrtPhysicalMemProp*, uint64_t);
    static AclrtMallocPhysical func = nullptr;
    if (func == nullptr) {
        func = (AclrtMallocPhysical)GET_FUNC(aclrtMallocPhysical);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtMallocPhysical", PROF_ERROR(ErrCode::NOT_FOUND));
    aclError ret = func(handle, size, prop, flags);
    if (ret != ACL_RT_SUCCESS && (prop->memAttr == ACL_HBM_MEM_HUGE1G)) {
        TORCH_NPU_WARN_ONCE("The malloc 1G large-page physical memory failed, so malloc 2M page memory."
                            "Using the 2M memory page may result in performance degradation. "
                            "This warning occurs because the PYTORCH_NPU_ALLOC_CONF = page_size:1g configuration "
                            "is enabled, but the pre-allocated number of 1G large pages is insufficient "
                            "or 1G large-page memory pre-allocation is not enabled.");
        aclrtPhysicalMemProp prop_update = {prop->handleType,
                                            prop->allocationType,
                                            ACL_HBM_MEM_HUGE,
                                            {prop->location.id,
                                             prop->location.type},
                                            prop->reserve};
        ret = func(handle, size, &prop_update, flags);
    }
    return ret;
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

aclError AclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags,
                     HcclComm hcclComm)
{
    typedef aclError (*AclrtMapMem)(void*, size_t, size_t, aclrtDrvMemHandle, uint64_t);
    static AclrtMapMem func = nullptr;
    if (func == nullptr) {
        func = (AclrtMapMem)GET_FUNC(aclrtMapMem);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtMapMem", PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(virPtr, size, offset, handle, flags);
    if (hcclComm) {
        HCCL_CHECK_ERROR(at_npu::hccl::HcclCommActivateCommMemoryFace(hcclComm, virPtr, size, offset, handle, flags));
    }
    return ret;
}

aclError AclrtUnmapMem(void *virPtr, HcclComm hcclComm)
{
    typedef aclError (*AclrtUnmapMem)(void*);
    static AclrtUnmapMem func = nullptr;
    if (func == nullptr) {
        func = (AclrtUnmapMem)GET_FUNC(aclrtUnmapMem);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtUnmapMem", PTA_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(virPtr);
    if (hcclComm) {
        HCCL_CHECK_ERROR(at_npu::hccl::HcclCommDeactivateCommMemoryFace(hcclComm, virPtr));
    }
    return ret;
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

aclError AclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize, size_t *retSize)
{
    typedef aclError (*AclrtGetMemUceInfo)(int32_t, aclrtMemUceInfo*, size_t, size_t *);
    static AclrtGetMemUceInfo func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetMemUceInfo) GET_FUNC(aclrtGetMemUceInfo);
    }
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtGetMemUceInfo");
        return ACL_ERROR_NONE;
    }
    return func(deviceId, memUceInfoArray, arraySize, retSize);
}

aclError AclrtDeviceTaskAbort(int32_t deviceId)
{
    typedef aclError (*AclrtDeviceTaskAbort)(int32_t, uint32_t);
    static AclrtDeviceTaskAbort func = nullptr;
    if (func == nullptr) {
        func = (AclrtDeviceTaskAbort) GET_FUNC(aclrtDeviceTaskAbort);
    }
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtDeviceTaskAbort");
        return ACL_ERROR_NONE;
    }
    uint32_t timeout = 0;
    return func(deviceId, timeout);
}

aclError AclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize)
{
    typedef aclError (*AclrtMemUceRepair)(int32_t, aclrtMemUceInfo*, size_t);
    static AclrtMemUceRepair func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemUceRepair) GET_FUNC(aclrtMemUceRepair);
    }
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtMemUceRepair");
        return ACL_ERROR_NONE;
    }
    return func(deviceId, memUceInfoArray, arraySize);
}

aclError AclrtGetMemUsageInfo(uint32_t deviceId, aclrtMemUsageInfo *memUsageInfo, size_t inputNum, size_t *outputNum)
{
    typedef aclError (*AclrtGetMemUsageInfo)(uint32_t, aclrtMemUsageInfo*, size_t, size_t *);
    static AclrtGetMemUsageInfo func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetMemUsageInfo) GET_FUNC(aclrtGetMemUsageInfo);
    }
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtGetMemUsageInfo");
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    return func(deviceId, memUsageInfo, inputNum, outputNum);
}

aclError AclrtCmoAsync(void* src, size_t size, aclrtCmoType cmoType, aclrtStream stream)
{
    typedef aclError (*AclrtCmoAsync)(void*, size_t, aclrtCmoType, aclrtStream);
    static AclrtCmoAsync func = nullptr;
    if (func == nullptr) {
        func = (AclrtCmoAsync) GET_FUNC(aclrtCmoAsync);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtCmoAsync", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(src, size, cmoType, stream);
}

aclError AclrtGetLastError(aclrtLastErrLevel flag)
{
    typedef aclError (*AclrtGetLastError)(aclrtLastErrLevel flag);
    static AclrtGetLastError func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetLastError) GET_FUNC(aclrtGetLastError);
    }
    if (func == nullptr) {
        return ACL_ERROR_NONE;
    }
    return func(flag);
}

aclError AclrtPeekAtLastError(aclrtLastErrLevel flag)
{
    typedef aclError (*AclrtPeekAtLastError)(aclrtLastErrLevel flag);
    static AclrtPeekAtLastError func = nullptr;
    if (func == nullptr) {
        func = (AclrtPeekAtLastError) GET_FUNC(aclrtPeekAtLastError);
    }
    if (func == nullptr) {
        return ACL_ERROR_NONE;
    }
    return func(flag);
}

aclError AclStressDetect(int32_t deviceId, void *workspace, size_t workspaceSize)
{
    typedef aclError (*AclStressDetect)(int32_t, void*, size_t);
    static AclStressDetect func = nullptr;
    if (func == nullptr) {
        func = (AclStressDetect) GetOpApiFuncAddr("StressDetect");
    }
    TORCH_CHECK(func, "Failed to find function ", "StressDetect", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, workspace, workspaceSize);
}

aclError AclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion *version)
{
    using aclsysGetCANNVersionFunc = aclError(*)(aclCANNPackageName, aclCANNPackageVersion *);
    static aclsysGetCANNVersionFunc func = nullptr;
    if (func == nullptr) {
        func = (aclsysGetCANNVersionFunc)GET_FUNC(aclsysGetCANNVersion);
        if (func == nullptr) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
    }

    return func(name, version);
}

aclError AclsysGetVersionStr(char *pkgName, char *versionStr)
{
    using aclsysGetVersionStrFunc = aclError(*)(char *, char *);
    static aclsysGetVersionStrFunc func = nullptr;
    func = (aclsysGetVersionStrFunc)GET_FUNC(aclsysGetVersionStr);
    if (func == nullptr) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    return func(pkgName, versionStr);
}

aclError AclrtSynchronizeDeviceWithTimeout(void)
{
    typedef aclError (*AclrtSynchronizeDeviceWithTimeout)(int32_t);
    static AclrtSynchronizeDeviceWithTimeout func = (AclrtSynchronizeDeviceWithTimeout)GET_FUNC(aclrtSynchronizeDeviceWithTimeout);
    int32_t timeout = c10_npu::option::OptionsManager::GetACLDeviceSyncTimeout();
    if (func != nullptr) {
        return func(timeout);
    } else {
        if (timeout > 0) {
            TORCH_NPU_WARN_ONCE("The ACL_DEVICE_SYNC_TIMEOUT does not take effect. If you want to enable this env, please upgrade CANN to the matching version.");
        }
        typedef aclError (*AclrtSynchronizeDevice)(void);
        static AclrtSynchronizeDevice func_backup = nullptr;
        if (func_backup == nullptr) {
            func_backup = (AclrtSynchronizeDevice)GET_FUNC(aclrtSynchronizeDevice);
        }
        TORCH_CHECK(func_backup, "Failed to find function ", "aclrtSynchronizeDeviceWithTimeout and aclrtSynchronizeDevice", PTA_ERROR(ErrCode::NOT_FOUND));
        return func_backup();
    }
}

aclError AclrtEventGetTimestamp(aclrtEvent event, uint64_t *timestamp)
{
    typedef aclError (*AclrtEventGetTimestamp)(aclrtEvent, uint64_t*);
    static AclrtEventGetTimestamp func = nullptr;
    if (func == nullptr) {
        func = (AclrtEventGetTimestamp)GET_FUNC(aclrtEventGetTimestamp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtEventGetTimestamp", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(event, timestamp);
}

aclError AclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode)
{
    typedef aclError (*AclmdlRICaptureBegin)(aclrtStream, aclmdlRICaptureMode);
    static AclmdlRICaptureBegin func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureBegin) GET_FUNC(aclmdlRICaptureBegin);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureBegin", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, mode);
}

aclError AclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI)
{
    typedef aclError (*AclmdlRICaptureGetInfo)(aclrtStream, aclmdlRICaptureStatus *, aclmdlRI *);
    static AclmdlRICaptureGetInfo func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureGetInfo) GET_FUNC(aclmdlRICaptureGetInfo);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureGetInfo", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, status, modelRI);
}

aclError AclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI)
{
    typedef aclError (*AclmdlRICaptureEnd)(aclrtStream, aclmdlRI *);
    static AclmdlRICaptureEnd func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureEnd) GET_FUNC(aclmdlRICaptureEnd);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureEnd", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, modelRI);
}

aclError AclmdlRIDebugPrint(aclmdlRI modelRI)
{
    typedef aclError (*AclmdlRIDebugPrint)(aclmdlRI);
    static AclmdlRIDebugPrint func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRIDebugPrint) GET_FUNC(aclmdlRIDebugPrint);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRIDebugPrint", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(modelRI);
}

aclError AclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream)
{
    typedef aclError (*AclmdlRIExecuteAsync)(aclmdlRI, aclrtStream);
    static AclmdlRIExecuteAsync func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRIExecuteAsync) GET_FUNC(aclmdlRIExecuteAsync);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRIExecuteAsync", PTA_ERROR(ErrCode::NOT_FOUND));

    return func(modelRI, stream);
}

aclError AclmdlRIDestroy(aclmdlRI modelRI)
{
    typedef aclError (*AclmdlRIDestroy)(aclmdlRI);
    static AclmdlRIDestroy func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRIDestroy) GET_FUNC(aclmdlRIDestroy);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRIDestroy", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(modelRI);
}

bool IsCaptureSupported()
{
    static bool is_support = false;
    static bool have_load_func = false;
    static bool default_support_capture = ((GetSocVersion() >= SocVersion::Ascend910B1) &&
        (GetSocVersion() < SocVersion::Ascend310B1)) ||
        ((GetSocVersion() >= SocVersion::Ascend910_9391));
    if (default_support_capture && !have_load_func) {
        have_load_func = true;
        typedef aclError (*AclmdlRICaptureGetInfo)(aclrtStream, aclmdlRICaptureStatus *, aclmdlRI *);
        static AclmdlRICaptureGetInfo func = (AclmdlRICaptureGetInfo) GET_FUNC(aclmdlRICaptureGetInfo);
        is_support = (func != nullptr);
    }

    return is_support;
}

aclError AclmdlRICaptureTaskGrpBegin(aclrtStream stream)
{
    typedef aclError (*AclmdlRICaptureTaskGrpBegin)(aclrtStream);
    static AclmdlRICaptureTaskGrpBegin func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureTaskGrpBegin) GET_FUNC(aclmdlRICaptureTaskGrpBegin);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureTaskGrpBegin", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream);
}

aclError AclmdlRICaptureTaskGrpEnd(aclrtStream stream, aclrtTaskGrp *handle)
{
    typedef aclError (*AclmdlRICaptureTaskGrpEnd)(aclrtStream, aclrtTaskGrp*);
    static AclmdlRICaptureTaskGrpEnd func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureTaskGrpEnd) GET_FUNC(aclmdlRICaptureTaskGrpEnd);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureTaskGrpEnd", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, handle);
}

aclError AclmdlRICaptureTaskUpdateBegin(aclrtStream stream, aclrtTaskGrp handle)
{
    typedef aclError (*AclmdlRICaptureTaskUpdateBegin)(aclrtStream, aclrtTaskGrp);
    static AclmdlRICaptureTaskUpdateBegin func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureTaskUpdateBegin) GET_FUNC(aclmdlRICaptureTaskUpdateBegin);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureTaskUpdateBegin", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, handle);
}

aclError AclmdlRICaptureTaskUpdateEnd(aclrtStream stream)
{
    typedef aclError (*AclmdlRICaptureTaskUpdateEnd)(aclmdlRI);
    static AclmdlRICaptureTaskUpdateEnd func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRICaptureTaskUpdateEnd) GET_FUNC(aclmdlRICaptureTaskUpdateEnd);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRICaptureTaskUpdateEnd", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream);
}

aclError AclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char* path)
{
    typedef aclError (*AclmdlRIDebugJsonPrint)(aclmdlRI, const char*);
    static AclmdlRIDebugJsonPrint func = nullptr;
    if (func == nullptr) {
        func = (AclmdlRIDebugJsonPrint) GET_FUNC(aclmdlRIDebugJsonPrint);
    }

    TORCH_CHECK(func, "Failed to find function aclmdlRIDebugJsonPrint", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(modelRI, path);
}

aclError AclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr)
{
    typedef aclError (*AclrtHostRegister)(void *, uint64_t, aclrtHostRegisterType, void **);
    static AclrtHostRegister func = nullptr;
    if (func == nullptr) {
        func = (AclrtHostRegister) GET_FUNC(aclrtHostRegister);
    }

    TORCH_CHECK(func, "Failed to find function aclrtHostRegister", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(ptr, size, type, devPtr);
}

aclError AclrtHostUnregister(void *ptr)
{
    typedef aclError (*AclrtHostUnregister)(void *);
    static AclrtHostUnregister func = nullptr;
    if (func == nullptr) {
        func = (AclrtHostUnregister) GET_FUNC(aclrtHostUnregister);
    }

    TORCH_CHECK(func, "Failed to find function aclrtHostUnregister", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(ptr);
}

aclError AclrtMallocHostWithCfg(void **ptr, uint64_t size, aclrtMallocConfig *cfg)
{
    typedef aclError (*AclrtMallocHostWithCfg)(void **, uint64_t, aclrtMallocConfig *);
    static AclrtMallocHostWithCfg func = nullptr;
    if (func == nullptr) {
        func = (AclrtMallocHostWithCfg) GET_FUNC(aclrtMallocHostWithCfg);
    }

    TORCH_CHECK(func, "Failed to find function aclrtMallocHostWithCfg", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(ptr, size, cfg);
}

bool AclrtMallocHostWithCfgExist()
{
    const static bool isAclrtMallocHostWithCfgExist = []() -> bool {
        std::string currentCANNVersion = GetCANNVersion("CANN");
        if (currentCANNVersion == "") {
            return false;
        }
        // determine the runtime version
        const std::string kMinRuntimeVersion = "8.5.0";
        if (!IsGteCANNVersion(kMinRuntimeVersion, "RUNTIME")) {
            return false;
        }
        auto func = GET_FUNC(aclrtMallocHostWithCfg);
        if (func != nullptr) {
            ASCEND_LOGI("Successfully to find function aclrtMallocHostWithCfg");
            return c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
        }
        return false;
    }();
    return isAclrtMallocHostWithCfgExist;
}

aclError AclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flag)
{
    typedef aclError (*AclrtIpcMemGetExportKey)(void *, size_t, char *, size_t, uint64_t);
    static AclrtIpcMemGetExportKey func = nullptr;
    if (func == nullptr) {
        func = (AclrtIpcMemGetExportKey) GET_FUNC(aclrtIpcMemGetExportKey);
    }

    TORCH_CHECK(func, "Failed to find function aclrtIpcMemGetExportKey", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(devPtr, size, key, len, flag);
}

aclError AclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num)
{
    typedef aclError (*AclrtIpcMemSetImportPid)(const char *, int32_t *, size_t);
    static AclrtIpcMemSetImportPid func = nullptr;
    if (func == nullptr) {
        func = (AclrtIpcMemSetImportPid) GET_FUNC(aclrtIpcMemSetImportPid);
    }

    TORCH_CHECK(func, "Failed to find function aclrtIpcMemSetImportPid", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(key, pid, num);
}

aclError AclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flag)
{
    typedef aclError (*AclrtIpcMemImportByKey)(void **, const char *, uint64_t);
    static AclrtIpcMemImportByKey func = nullptr;
    if (func == nullptr) {
        func = (AclrtIpcMemImportByKey) GET_FUNC(aclrtIpcMemImportByKey);
    }

    TORCH_CHECK(func, "Failed to find function aclrtIpcMemImportByKey", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(devPtr, key, flag);
}

aclError AclrtIpcMemClose(const char *key)
{
    typedef aclError (*AclrtIpcMemClose)(const char *);
    static AclrtIpcMemClose func = nullptr;
    if (func == nullptr) {
        func = (AclrtIpcMemClose) GET_FUNC(aclrtIpcMemClose);
    }

    TORCH_CHECK(func, "Failed to find function aclrtIpcMemClose", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(key);
}

aclError AclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType,
                                         uint64_t flags, uint64_t *shareableHandle)
{
    typedef aclError (*AclrtMemExportToShareableHandle)(aclrtDrvMemHandle, aclrtMemHandleType, uint64_t, uint64_t *);
    static AclrtMemExportToShareableHandle func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemExportToShareableHandle) GET_FUNC(aclrtMemExportToShareableHandle);
    }

    TORCH_CHECK(func, "Failed to find function aclrtMemExportToShareableHandle", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(handle, handleType, flags, shareableHandle);
}

aclError AclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum)
{
    typedef aclError (*AclrtMemSetPidToShareableHandle)(uint64_t, int32_t *, size_t);
    static AclrtMemSetPidToShareableHandle func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemSetPidToShareableHandle) GET_FUNC(aclrtMemSetPidToShareableHandle);
    }

    TORCH_CHECK(func, "Failed to find function aclrtMemSetPidToShareableHandle", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(shareableHandle, pid, pidNum);
}

aclError AclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle)
{
    typedef aclError (*AclrtMemImportFromShareableHandle)(uint64_t, int32_t, aclrtDrvMemHandle *);
    static AclrtMemImportFromShareableHandle func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemImportFromShareableHandle) GET_FUNC(aclrtMemImportFromShareableHandle);
    }

    TORCH_CHECK(func, "Failed to find function aclrtMemImportFromShareableHandle", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(shareableHandle, deviceId, handle);
}

aclError AclrtDeviceGetBareTgid(int32_t *pid)
{
    typedef aclError (*AclrtDeviceGetBareTgid)(int32_t *);
    static AclrtDeviceGetBareTgid func = nullptr;
    if (func == nullptr) {
        func = (AclrtDeviceGetBareTgid) GET_FUNC(aclrtDeviceGetBareTgid);
    }

    TORCH_CHECK(func, "Failed to find function aclrtDeviceGetBareTgid", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(pid);
}

aclError AclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value)
{
    typedef aclError (*AclrtGetDeviceResLimit)(int32_t, aclrtDevResLimitType, uint32_t*);
    static AclrtGetDeviceResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDeviceResLimit) GET_FUNC(aclrtGetDeviceResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtGetDeviceResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, type, value);
}

aclError AclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value)
{
    typedef aclError (*AclrtSetDeviceResLimit)(int32_t, aclrtDevResLimitType, uint32_t);
    static AclrtSetDeviceResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetDeviceResLimit) GET_FUNC(aclrtSetDeviceResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtSetDeviceResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, type, value);
}

aclError AclrtResetDeviceResLimit(int32_t deviceId)
{
    typedef aclError (*AclrtResetDeviceResLimit)(int32_t);
    static AclrtResetDeviceResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtResetDeviceResLimit) GET_FUNC(aclrtResetDeviceResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtResetDeviceResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId);
}

aclError AclrtStreamGetId(aclrtStream stream, int32_t* stream_id)
{
    typedef aclError(*AclrtStreamGetIdFunc)(aclrtStream, int32_t*);
    static AclrtStreamGetIdFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtStreamGetIdFunc)GET_FUNC(aclrtStreamGetId);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtStreamGetId", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stream, stream_id);
}

aclError AclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    typedef aclError (*AclrtLaunchCallback)(aclrtCallback, void *, aclrtCallbackBlockType, aclrtStream);
    static AclrtLaunchCallback func = nullptr;
    if (func == nullptr) {
        func = (AclrtLaunchCallback) GET_FUNC(aclrtLaunchCallback);
    }

    TORCH_CHECK(func, "Failed to find function aclrtLaunchCallback", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(fn, userData, blockType, stream);
}

aclError AclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    typedef aclError (*AclrtSubscribeReport)(uint64_t, aclrtStream);
    static AclrtSubscribeReport func = nullptr;
    if (func == nullptr) {
        func = (AclrtSubscribeReport) GET_FUNC(aclrtSubscribeReport);
    }

    TORCH_CHECK(func, "Failed to find function aclrtSubscribeReport", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(threadId, stream);
}

aclError AclrtUnSubscribeReport(uint64_t theadId, aclrtStream stream)
{
    typedef aclError (*AclrtUnSubscribeReport)(uint64_t, aclrtStream);
    static AclrtUnSubscribeReport func = nullptr;
    if (func == nullptr) {
        func = (AclrtUnSubscribeReport) GET_FUNC(aclrtUnSubscribeReport);
    }

    TORCH_CHECK(func, "Failed to find function aclrtUnSubscribeReport", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(theadId, stream);
}

bool IsExistMemcpyBatch()
{
    typedef aclError(*AclrtMemcpyBatchFunc)(void **, size_t *, void **, size_t *,
                                            size_t, aclrtMemcpyBatchAttr *, size_t *,
                                            size_t, size_t *);
    static AclrtMemcpyBatchFunc func = (AclrtMemcpyBatchFunc) GET_FUNC(aclrtMemcpyBatch);
    return func != nullptr;
}

bool IsExistRtGetStreamId()
{
    typedef aclError(*AclrtStreamGetIdFunc)(aclrtStream, int32_t*);
    static AclrtStreamGetIdFunc func = (AclrtStreamGetIdFunc) GET_FUNC(aclrtStreamGetId);
    return func != nullptr;
}

bool IsExistMemcpyBatchAsync()
{
    typedef aclError(*AclrtMemcpyBatchAsyncFunc)(void **, size_t *, void **, size_t *,
                                                 size_t, aclrtMemcpyBatchAttr *, size_t *,
                                                 size_t, size_t *);
    static AclrtMemcpyBatchAsyncFunc func = (AclrtMemcpyBatchAsyncFunc) GET_FUNC(aclrtMemcpyBatchAsync);
    return func != nullptr;
}

aclError AclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                          size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                          size_t numAttrs, size_t *failIndex)
{
    typedef aclError(*AclrtMemcpyBatchFunc)(void **, size_t *, void **, size_t *,
                                            size_t, aclrtMemcpyBatchAttr *, size_t *,
                                            size_t, size_t *);
    static AclrtMemcpyBatchFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemcpyBatchFunc) GET_FUNC(aclrtMemcpyBatch);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtMemcpyBatch", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(dsts, destMax, srcs, sizes, numBatches, attrs, attrsIndexes, numAttrs, failIndex);
}

aclError AclrtMemcpyBatchAsync(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                               size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                               size_t numAttrs, size_t *failIndex, aclrtStream stream)
{
    typedef aclError(*AclrtMemcpyBatchAsyncFunc)(void **, size_t *, void **, size_t *,
                                                 size_t, aclrtMemcpyBatchAttr *, size_t *,
                                                 size_t, size_t *, aclrtStream);
    static AclrtMemcpyBatchAsyncFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemcpyBatchAsyncFunc) GET_FUNC(aclrtMemcpyBatchAsync);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtMemcpyBatchAsync", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(dsts, destMax, srcs, sizes, numBatches, attrs, attrsIndexes, numAttrs, failIndex, stream);
}

bool AclrtMemcpyAsyncWithConditionExist()
{
    const static bool isAclrtMemcpyAsyncWithConditionExist = []() -> bool {
        const std::string kMinDriverVersion = "25.0.RC1";
        if (!IsGteDriverVersion(kMinDriverVersion)) {
            return false;
        }
        auto func = GET_FUNC(aclrtMemcpyAsyncWithCondition);
        if (func != nullptr) {
            ASCEND_LOGI("Successfully to find function aclrtMemcpyAsyncWithCondition");
            return c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
        }
        return false;
    }();
    return isAclrtMemcpyAsyncWithConditionExist;
}

aclError AclrtMemcpyAsyncWithCondition(void *dst, size_t destMax, const void *src,
                                       size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    typedef aclError(*AclrtMemcpyAsyncWithConditionFunc)(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream);
    static AclrtMemcpyAsyncWithConditionFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtMemcpyAsyncWithConditionFunc)GET_FUNC(aclrtMemcpyAsyncWithCondition);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtMemcpyAsyncWithCondition", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(dst, destMax, src, count, kind, stream);
}

aclError AclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value)
{
    typedef aclError (*AclrtSetStreamResLimit)(aclrtStream, aclrtDevResLimitType, uint32_t);
    static AclrtSetStreamResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetStreamResLimit) GET_FUNC(aclrtSetStreamResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtSetStreamResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, type, value);
}

aclError AclrtResetStreamResLimit(aclrtStream stream)
{
    typedef aclError (*AclrtResetStreamResLimit)(aclrtStream);
    static AclrtResetStreamResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtResetStreamResLimit) GET_FUNC(aclrtResetStreamResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtResetStreamResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream);
}

aclError AclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value)
{
    typedef aclError (*AclrtGetStreamResLimit)(aclrtStream, aclrtDevResLimitType, uint32_t*);
    static AclrtGetStreamResLimit func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetStreamResLimit) GET_FUNC(aclrtGetStreamResLimit);
    }

    TORCH_CHECK(func, "Failed to find function aclrtGetStreamResLimit", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, type, value);
}

aclError AclrtUseStreamResInCurrentThread(aclrtStream stream)
{
    typedef aclError (*AclrtUseStreamResInCurrentThread)(aclrtStream);
    static AclrtUseStreamResInCurrentThread func = nullptr;
    if (func == nullptr) {
        func = (AclrtUseStreamResInCurrentThread) GET_FUNC(aclrtUseStreamResInCurrentThread);
    }

    TORCH_CHECK(func, "Failed to find function aclrtUseStreamResInCurrentThread", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream);
}

aclError AclrtUnuseStreamResInCurrentThread(aclrtStream stream)
{
    typedef aclError (*AclrtUnuseStreamResInCurrentThread)(aclrtStream);
    static AclrtUnuseStreamResInCurrentThread func = nullptr;
    if (func == nullptr) {
        func = (AclrtUnuseStreamResInCurrentThread) GET_FUNC(aclrtUnuseStreamResInCurrentThread);
    }

    TORCH_CHECK(func, "Failed to find function aclrtUnuseStreamResInCurrentThread", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream);
}

aclError AclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    typedef aclError (*AclrtGetResInCurrentThread)(aclrtDevResLimitType, uint32_t*);
    static AclrtGetResInCurrentThread func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetResInCurrentThread) GET_FUNC(aclrtGetResInCurrentThread);
    }

    TORCH_CHECK(func, "Failed to find function aclrtGetResInCurrentThread", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(type, value);
}

aclError AclrtPointerGetAttributes(const void *ptr, aclrtPtrAttributes *attributes)
{
    using AclrtPointerGetAttributes = aclError (*)(const void*, aclrtPtrAttributes*);
    static AclrtPointerGetAttributes func = nullptr;
    if (func == nullptr) {
        func = (AclrtPointerGetAttributes) GET_FUNC(aclrtPointerGetAttributes);
    }

    TORCH_CHECK(func, "Failed to find function aclrtPointerGetAttributes", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(ptr, attributes);
}

bool AclrtPointerGetAttributesExist()
{
    const static bool isAclrtPointerGetAttributesExist = []() -> bool {
        const std::string kMinRuntimeVersion = "8.5.0";
        if (!IsGteCANNVersion(kMinRuntimeVersion, "RUNTIME")) {
            return false;
        }
        auto func = GET_FUNC(aclrtPointerGetAttributes)
        return func != nullptr;
    }();
    return isAclrtPointerGetAttributesExist;
}

aclError AclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    typedef aclError (*AclrtSetStreamAttribute)(aclrtStream, aclrtStreamAttr, aclrtStreamAttrValue*);
    static AclrtSetStreamAttribute func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetStreamAttribute) GET_FUNC(aclrtSetStreamAttribute);
    }

    TORCH_CHECK(func, "Failed to find function aclrtSetStreamAttribute", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(stream, stmAttrType, value);
}

bool IsExistDeviceGetUuid()
{
    typedef aclError (*AclrtDeviceGetUuid)(int32_t, aclrtUuid*);
    static AclrtDeviceGetUuid func = (AclrtDeviceGetUuid)GET_FUNC(aclrtDeviceGetUuid);
    return func != nullptr;
}

aclError AclrtDeviceGetUuid(int32_t deviceId, aclrtUuid* uuid)
{
    typedef aclError (*AclrtDeviceGetUuid)(int32_t, aclrtUuid*);
    static AclrtDeviceGetUuid func = nullptr;
    if (func == nullptr) {
        func = (AclrtDeviceGetUuid)GET_FUNC(aclrtDeviceGetUuid);
    }

    TORCH_CHECK(func, "Failed to find function aclrtDeviceGetUuid", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, uuid);
}

} // namespace acl
} // namespace c10
