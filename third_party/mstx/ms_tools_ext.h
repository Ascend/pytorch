#ifndef INC_EXTERNAL_MS_TOOLS_EXT_H_
#define INC_EXTERNAL_MS_TOOLS_EXT_H_

#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MSTX_INVALID_ID 0

typedef uint64_t mstxRangeId;

ACL_FUNC_VISIBILITY void mstxMarkA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY mstxRangeId mstxRangeStartA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY void mstxRangeEnd(mstxRangeId id);

#ifdef __cplusplus
}
#endif

#endif
