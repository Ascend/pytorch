#ifndef __PLUGIN_NATIVE_NPU_INTERFACE_HCCLINTERFACE__
#define __PLUGIN_NATIVE_NPU_INTERFACE_HCCLINTERFACE__

#include "third_party/hccl/inc/hccl/hccl.h"

namespace at_npu {
namespace native {
namespace hccl {
/**
 * @ingroup AscendCL
 * @brief set hccl config option value
 *
 * @param config [IN]      hccl set config type
 * @param configValue [IN]   hccl set config value
 *
 * @return HcclResult
 */
extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);

} // namespace hccl
} // namespace native
} // namespace at_npu
#endif