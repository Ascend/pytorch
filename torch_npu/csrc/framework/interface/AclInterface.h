#ifndef __TORCH_NPU_INTERFACE_ACLINTERFACE__
#define __TORCH_NPU_INTERFACE_ACLINTERFACE__

#include "third_party/acl/inc/acl/acl_rt.h"
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_prof.h>
#include <third_party/acl/inc/acl/acl_op.h>

namespace at_npu {
namespace native {

typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xffff,
} aclrtEventWaitStatus;

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
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

aclError AclProfilingInit(const char *profilerResultPath, size_t length);
aclError AclProfilingStart(const aclprofConfig *profilerConfig);
aclError AclProfilingStop(const aclprofConfig *profilerConfig);
aclError AclProfilingFinalize();
aclprofConfig* AclProfilingCreateConfig(
    uint32_t *deviceIdList,
    uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics,
    aclprofAicoreEvents *aicoreEvents,
    uint64_t dataTypeConfig);
aclError AclProfilingDestroyConfig(const aclprofConfig *profilerConfig);

aclError AclopStartDumpArgs(uint32_t dumpType, const char *path);

aclError AclopStopDumpArgs(uint32_t dumpType);

} // namespace native
} // namespace at_npu

#endif // __TORCH_NPU_INTERFACE_ACLINTERFACE__