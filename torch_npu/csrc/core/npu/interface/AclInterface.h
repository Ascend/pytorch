#pragma once

#include <c10/core/Device.h>

#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_mdl.h"
#include "third_party/acl/inc/acl/acl_prof.h"
#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "third_party/acl/inc/acl/acl.h"

using aclrtHostFunc = void (*)(void *args);
struct aclrtMemUsageInfo;
struct aclOpExecutor;
struct aclrtUuid;
struct aclrtIpcEventHandle;
namespace c10_npu {
namespace acl {
enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xffff,
};
using aclrtEventWaitStatus = enum aclrtEventWaitStatus;

enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE  = 1,
};
using aclrtEventRecordedStatus = enum aclrtEventRecordedStatus;

enum aclrtStreamStatus {
    ACL_STREAM_STATUS_COMPLETE  = 0,
    ACL_STREAM_STATUS_NOT_READY = 1,
    ACL_STREAM_STATUS_RESERVED  = 0xFFFF,
};
using aclrtStreamStatus = enum aclrtStreamStatus;

enum aclrtDevResLimitType {
    ACL_RT_DEV_RES_CUBE_CORE = 0,
    ACL_RT_DEV_RES_VECTOR_CORE = 1,
};
using aclrtDevResLimitType = enum aclrtDevResLimitType;

/**
  aclprofStepInfo is provide by acl, it used to be store dispatch op info.
 */
using aclprofStepInfoPtr = aclprofStepInfo *;
/**
 NpdStatus is provide by acl, it used to store the return value.
 */
using NpdStatus = int;

/**
  This Api is used to init npd, it need to be called once at process.
 */
aclprofStepInfoPtr init_stepinfo();
/**
  This Api is used to destroy npd, it need to be called once at process.
 */
NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo);
/**
  This Api is used to start dispatch op, this operation should be called after init.
 */
NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream);
/**
  This Api is used to stop dispatch op, this operation should be called after start dispatch op.
 */
NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream);

/**
  This API is used to get error msg
  */
const char *AclGetErrMsg();

/**
 * This API is used to create fast streams through the param flag
 */
aclError AclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);

/**
 * This API is used to set stream mode
 */
aclError AclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);

/**
 * This API is used to set op wait timeout
 */
aclError AclrtSetOpWaitTimeout(uint32_t timeout);

/**
 * This API is used to check whether aclrtCreateEventExWithFlag exist
 * Compatible CANN, delete in future
*/
bool IsExistCreateEventExWithFlag();

/**
 * This API is used to check whether aclrtValueWait and aclrtValueWrite exist
*/
bool IsExistValueWaitAndWrite();

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
  This API is used to query wait status of event task
  */
aclError AclQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *waitStatus);

/**
  This API is used to check whether aclrtQueryEventStatus exist
  */
bool IsExistQueryEventRecordedStatus();

/**
  This API is used to query recorded status of event task
  */
aclError AclQueryEventRecordedStatus(aclrtEvent event, aclrtEventRecordedStatus *status);

aclError AclIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle);

aclError AclIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event);

bool IsSupportIpcEvent();

aclError AclProfilingInit(const char *profilerResultPath, size_t length);
aclError AclProfilingStart(const aclprofConfig *profilerConfig);
aclError AclProfilingStop(const aclprofConfig *profilerConfig);
aclError AclProfilingFinalize();
aclprofConfig *AclProfilingCreateConfig(
    uint32_t *deviceIdList,
    uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics,
    aclprofAicoreEvents *aicoreEvents,
    uint64_t dataTypeConfig);
aclError AclProfilingDestroyConfig(const aclprofConfig *profilerConfig);
const char *AclrtGetSocName();
const char *AclGetSocName();
aclError AclrtSetDeviceSatMode(aclrtFloatOverflowMode mode);

aclError AclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag);

aclError AclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag);

aclError AclrtSetOpExecuteTimeOut(uint32_t timeout);

aclError AclrtSetOpExecuteTimeOutV2(uint64_t timeout);

aclError AclrtSynchronizeStreamWithTimeout(aclrtStream stream);

aclError AclrtDestroyStreamForce(aclrtStream stream);

aclError AclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo);

aclError AclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy);

aclError AclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id);

aclError AclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags,
                                HcclComm hcclComm = nullptr);

aclError AclrtReleaseMemAddress(void *virPtr, HcclComm hcclComm = nullptr);

aclError AclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags);

aclError AclrtFreePhysical(aclrtDrvMemHandle handle);

aclError AclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags,
                     HcclComm hcclComm = nullptr);

aclError AclrtUnmapMem(void *virPtr, HcclComm hcclComm = nullptr);

bool IsExistGetCannAttribute();

aclError AclGetCannAttributeList(const aclCannAttr **cannAttrList, size_t *num);

aclError AclGetCannAttribute(aclCannAttr cannAttr, int32_t *value);

aclError AclGetDeviceCapability(uint32_t deviceId, aclDeviceInfo deviceInfo, int64_t *value);

aclError AclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize, size_t *retSize);

aclError AclrtGetMemUsageInfo(uint32_t deviceId, aclrtMemUsageInfo *memUsageInfo, size_t inputNum, size_t *outputNum);

aclError AclrtDeviceTaskAbort(int32_t deviceId);

aclError AclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize);

aclError AclrtCmoAsync(void* src, size_t size, aclrtCmoType cmoType, aclrtStream stream);

aclError AclrtGetLastError(aclrtLastErrLevel flag);

aclError AclrtPeekAtLastError(aclrtLastErrLevel flag);

aclError AclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion *version);

aclError AclsysGetVersionStr(char *pkgName, char *versionStr);

aclError AclStressDetect(int32_t deviceId, void *workspace, size_t workspaceSize);

aclError AclrtSynchronizeDeviceWithTimeout(void);

aclError AclrtEventGetTimestamp(aclrtEvent event, uint64_t *timestamp);

aclError AclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode);

aclError AclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI);

aclError AclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI);

aclError AclmdlRIDebugPrint(aclmdlRI modelRI);

aclError AclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream);

aclError AclmdlRIDestroy(aclmdlRI modelRI);

bool IsCaptureSupported();

aclError AclmdlRICaptureTaskGrpBegin(aclrtStream stream);

aclError AclmdlRICaptureTaskGrpEnd(aclrtStream stream, aclrtTaskGrp *handle);

aclError AclmdlRICaptureTaskUpdateBegin(aclrtStream stream, aclrtTaskGrp handle);

aclError AclmdlRICaptureTaskUpdateEnd(aclrtStream stream);

aclError AclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char* path);

/**
 * @ingroup AscendCL
 * @brief register host memory
 * @param ptr [IN]      memory pointer
 * @param size [IN]     memory size
 * @param type [IN]     memory register size
 * @param devPtr [OUT]  pointer to allocated memory on device
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr);

/**
 * @ingroup AscendCL
 * @brief register an existing host memory range
 *
 * @param ptr [IN]     host pointer to memory to page-lock
 * @param size [IN]    size in bytes of the address range to page-lock in bytes
 * @param flag [IN]    flag for allocation request
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief unregister host memory
 * @param ptr [IN]     memory pointer
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtHostUnregister(void *ptr);

aclError AclrtMallocHostWithCfg(void **ptr, uint64_t size, aclrtMallocConfig *cfg);

bool AclrtMallocHostWithCfgExist();

aclError AclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flag);

aclError AclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num);

aclError AclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flag);

aclError AclrtIpcMemClose(const char *key);

aclError AclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType,
                                         uint64_t flags, uint64_t *shareableHandle);

aclError AclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum);

aclError AclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle);

aclError AclrtDeviceGetBareTgid(int32_t *pid);

aclError AclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value);

aclError AclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value);

aclError AclrtResetDeviceResLimit(int32_t deviceId);

aclError AclrtStreamGetId(aclrtStream stream, int32_t* stream_id);

aclError AclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream);

aclError AclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc func, void *args);

aclError AclrtSubscribeReport(uint64_t theadId, aclrtStream stream);

aclError AclrtUnSubscribeReport(uint64_t theadId, aclrtStream stream);

aclError AclrtMemcpyBatchAsync(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                               size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                               size_t numAttrs, size_t *failIndex, aclrtStream stream);

aclError AclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                          size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                          size_t numAttrs, size_t *failIdx);

bool IsExistMemcpyBatch();

bool IsExistRtGetStreamId();

bool IsExistMemcpyBatchAsync();

bool AclrtMemcpyAsyncWithConditionExist();

aclError AclrtMemcpyAsyncWithCondition(void *dst, size_t destMax, const void *src,
                                       size_t count, aclrtMemcpyKind kind, aclrtStream stream);

aclError AclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value);

aclError AclrtResetStreamResLimit(aclrtStream stream);

aclError AclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value);

aclError AclrtUseStreamResInCurrentThread(aclrtStream stream);

aclError AclrtUnuseStreamResInCurrentThread(aclrtStream stream);

aclError AclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value);

aclError AclrtPointerGetAttributes(const void *ptr, aclrtPtrAttributes *attributes);

bool AclrtPointerGetAttributesExist();

aclError AclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value);

bool IsExistDeviceGetUuid();

aclError AclrtDeviceGetUuid(int32_t deviceId, aclrtUuid *uuid);

aclError AclrtValueWait(void* event, aclrtStream stream);

aclError AclrtValueWrite(void* event, uint64_t value, aclrtStream stream);

} // namespace acl
} // namespace c10_npu
