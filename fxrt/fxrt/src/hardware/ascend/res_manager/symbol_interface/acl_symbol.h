#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_base_symbol.h"

namespace fxrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(aclInit, aclError, const char*);
ORIGIN_METHOD_WITH_SIMU(aclFinalize, aclError);

extern aclInitFunObj aclInit_;
extern aclFinalizeFunObj aclFinalize_;

void LoadAclApiSymbol(const std::string& ascendPath);
void LoadSimulationAclApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
