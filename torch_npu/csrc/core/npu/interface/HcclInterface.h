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


} // namespace native
} // namespace at_npu