#ifndef INC_EXTERNAL_MS_TOOLS_EXT_H_
#define INC_EXTERNAL_MS_TOOLS_EXT_H_

#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MSTX_INVALID_ID 0

typedef uint64_t  mstxRangeId;

struct mstxDomainRegistration_st;
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t* mstxDomainhandle_t;

ACL_FUNC_VISIBILITY void mstxMarkA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY mstxRangeId mstxRangeStartA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY void mstxRangeEnd(mstxRangeId id);

ACL_FUNC_VISIBILITY mstxDomainhandle_t mstxDomainCreateA(const char* name);

ACL_FUNC_VISIBILITY void mstxDomainDestroy(mstxDomainhandle_t handle);

ACL_FUNC_VISIBILITY void mstxDomainMarkA(mstxDomainhandle_t handle, const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY mstxRangeId mstxDomainRangeStartA(mstxDomainhandle_t handle, const char* message,
                                                      aclrtStream stream);

ACL_FUNC_VISIBILITY void mstxDomainRangeEnd(mstxDomainhandle_t handle, mstxRangeId id);

#ifdef __cplusplus
}
#endif

#endif
