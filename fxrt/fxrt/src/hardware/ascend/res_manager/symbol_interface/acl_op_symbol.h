#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
#include <string>
#include "acl/acl_op.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(aclopCreateAttr, aclopAttr*)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrBool, aclError, aclopAttr*, const char*, uint8_t)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrDataType, aclError, aclopAttr*, const char*, aclDataType)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrFloat, aclError, aclopAttr*, const char*, float)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrInt, aclError, aclopAttr*, const char*, int64_t)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListBool, aclError, aclopAttr*, const char*, int, const uint8_t*)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListDataType, aclError, aclopAttr*, const char*, int, const aclDataType[])
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListFloat, aclError, aclopAttr*, const char*, int, const float*)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListInt, aclError, aclopAttr*, const char*, int, const int64_t*)
ORIGIN_METHOD_WITH_SIMU(
    aclopSetAttrListListInt,
    aclError,
    aclopAttr*,
    const char*,
    int,
    const int*,
    const int64_t* const[])
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListString, aclError, aclopAttr*, const char*, int, const char**)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrString, aclError, aclopAttr*, const char*, const char*)
ORIGIN_METHOD_WITH_SIMU(aclopSetModelDir, aclError, const char*)

extern aclopCreateAttrFunObj aclopCreateAttr_;
extern aclopSetAttrBoolFunObj aclopSetAttrBool_;
extern aclopSetAttrDataTypeFunObj aclopSetAttrDataType_;
extern aclopSetAttrFloatFunObj aclopSetAttrFloat_;
extern aclopSetAttrIntFunObj aclopSetAttrInt_;
extern aclopSetAttrListBoolFunObj aclopSetAttrListBool_;
extern aclopSetAttrListDataTypeFunObj aclopSetAttrListDataType_;
extern aclopSetAttrListFloatFunObj aclopSetAttrListFloat_;
extern aclopSetAttrListIntFunObj aclopSetAttrListInt_;
extern aclopSetAttrListListIntFunObj aclopSetAttrListListInt_;
extern aclopSetAttrListStringFunObj aclopSetAttrListString_;
extern aclopSetAttrStringFunObj aclopSetAttrString_;
extern aclopSetModelDirFunObj aclopSetModelDir_;

void LoadAclOpApiSymbol(const std::string& ascendPath);
void LoadSimulationAclOpApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
