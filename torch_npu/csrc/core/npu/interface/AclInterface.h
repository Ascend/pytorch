#pragma once

#include <c10/core/Device.h>

#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_prof.h"

namespace c10_npu {
namespace acl {
typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xffff,
} aclrtEventWaitStatus;

typedef enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY  = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE = 1,
} aclrtEventRecordedStatus;

typedef enum aclrtStreamStatus {
    ACL_STREAM_STATUS_COMPLETE  = 0,
    ACL_STREAM_STATUS_NOT_READY = 1,
    ACL_STREAM_STATUS_RESERVED  = 0xFFFF,
} aclrtStreamStatus;

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

aclError AclProfilingInit(const char *profilerResultPath, size_t length);
aclError AclProfilingStart(const aclprofConfig *profilerConfig);
aclError AclProfilingStop(const aclprofConfig *profilerConfig);
aclError AclProfilingFinalize();
aclprofConfig * AclProfilingCreateConfig(
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

aclError AclrtSynchronizeStreamWithTimeout(aclrtStream stream);

aclError AclrtDestroyStreamForce(aclrtStream stream);

aclError AclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo);

aclError AclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy);

aclError AclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id);
} // namespace acl
} // namespace c10_npu
