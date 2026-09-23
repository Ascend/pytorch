#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_BASE_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_BASE_SYMBOL_H_
#include <string>
#include "acl/acl_base.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt::device::ascend {
ORIGIN_METHOD_WITH_SIMU(aclCreateDataBuffer, aclDataBuffer*, void*, size_t);
ORIGIN_METHOD_WITH_SIMU(aclCreateTensorDesc, aclTensorDesc*, aclDataType, int, const int64_t*, aclFormat);
ORIGIN_METHOD_WITH_SIMU(aclDataTypeSize, size_t, aclDataType);
ORIGIN_METHOD_WITH_SIMU(aclDestroyDataBuffer, aclError, const aclDataBuffer*);
ORIGIN_METHOD_WITH_SIMU(aclDestroyTensorDesc, void, const aclTensorDesc*);
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescDimV2, aclError, const aclTensorDesc*, size_t, int64_t*);
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescNumDims, size_t, const aclTensorDesc*)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorConst, aclError, aclTensorDesc*, void*, size_t)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorDescName, void, aclTensorDesc*, const char*)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorFormat, aclError, aclTensorDesc*, aclFormat)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorPlaceMent, aclError, aclTensorDesc*, aclMemType)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorShape, aclError, aclTensorDesc*, int, const int64_t*)
ACLRT_GET_SOC_NAME_WITH_SIMU(aclrtGetSocName, const char*)
ORIGIN_METHOD_WITH_SIMU(aclUpdateDataBuffer, aclError, aclDataBuffer*, void*, size_t)
ORIGIN_METHOD_WITH_SIMU(aclGetDataBufferAddr, void*, const aclDataBuffer*)
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescSize, size_t, const aclTensorDesc*)
ORIGIN_METHOD_WITH_SIMU(aclGetRecentErrMsg, const char*)

void LoadAclBaseApiSymbol(const std::string& ascendPath);
void LoadSimulationAclBaseApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_BASE_SYMBOL_H_
