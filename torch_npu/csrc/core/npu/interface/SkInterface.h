#pragma once

#include "third_party/acl/inc/acl/acl_sk.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_mdl.h"

namespace c10_npu {
namespace skapi {
/**
 * This API is used to call aclskOptimizeMdlRI.
*/
aclError AclskOptimize(aclmdlRI model, const aclskOptions *options);

/**
 * This API is used to call aclskScopeBegin.
*/
aclError AclskScopeBegin(const char *scopeName, aclrtStream stream);

/**
 * This API is used to call aclskScopeBegin.
*/
aclError AclskScopeEnd(const char *scopeName, aclrtStream stream);
} // namespace skapi
} // namespace c10_npu