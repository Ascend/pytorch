#include "third_party/hccl/inc/hccl/hccl.h"

namespace at_npu {
namespace hccl {

/**
 * @ingroup AscendCL
 * @brief get hccl comm name
 *
 * @param commHandle [IN]    query hccl commHandle
 * @param commName [OUT]     hccl come name
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
extern HcclResult HcclGetCommNameFace(HcclComm commHandle, char* commName);

extern HcclResult HcclCommResumeFace(HcclComm comm);

/**
 * @ingroup AscendCL
 * @brief checkout hccl config Feature Supported
 *
 * @param configParameter [IN] config Feature enum
 * @param bool [OUT] feature supported status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
extern bool isHcclFeatureSupported(HcclCommConfigCapability configParameter);

HcclResult HcclCommSetMemoryRangeFace(HcclComm comm, void *virPtr, size_t size, size_t alignment, uint64_t flags);
HcclResult HcclCommUnsetMemoryRangeFace(HcclComm comm, void *virPtr);
HcclResult HcclCommActivateCommMemoryFace(HcclComm comm, void *virPtr, size_t size, size_t offset,
                                          aclrtDrvMemHandle handle, uint64_t flags);
HcclResult HcclCommDeactivateCommMemoryFace(HcclComm comm, void *virPtr);

} // namespace native
} // namespace at_npu