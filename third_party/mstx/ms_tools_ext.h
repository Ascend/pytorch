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
typedef mstxDomainRegistration_t* mstxDomainHandle_t;

struct mstxMemHeap_st;
typedef struct mstxMemHeap_st mstxMemHeap_t;
typedef mstxMemHeap_t* mstxMemHeapHandle_t;

struct mstxMemRegion_st;
typedef struct mstxMemRegion_st mstxMemRegion_t;
typedef mstxMemRegion_t* mstxMemRegionHandle_t;

typedef struct mstxMemVirtualRangeDesc_t {
    uint32_t deviceId;
    const void* ptr;
    uint64_t size;
} mstxMemVirtualRangeDesc_t;

typedef enum mstxMemHeapUsageType {
    MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0,
} mstxMemHeapUsageType;

typedef enum mstxMemType {
    MSTX_MEM_TYPE_VIRTUAL_ADDRESS = 0,
} mstxMemType;

typedef struct mstxMemHeapDesc_t {
    mstxMemHeapUsageType usage;
    mstxMemType type;
    const void* typeSpecificDesc;
} mstxMemHeapDesc_t;

typedef struct mstxMemRegionsRegisterBatch_t {
    mstxMemHeapHandle_t heap;
    mstxMemType regionType;
    size_t regionCount;
    const void* regionDescArray;
    mstxMemRegionHandle_t* regionHandleArrayOut;
} mstxMemRegionsRegisterBatch_t;

typedef enum mstxMemRegionRefType {
    MSTX_MEM_REGION_REF_TYPE_POINTER = 0,
    MSTX_MEM_REGION_REF_TYPE_HANDLE
} mstxMemRegionRefType;

typedef struct mstxMemRegionRef_t {
    mstxMemRegionRefType refType;
    union {
        const void* pointer;
        mstxMemRegionHandle_t handle;
    };
} mstxMemRegionRef_t;

typedef struct mstxMemRegionsUnregisterBatch_t {
    size_t refCount;
    const mstxMemRegionRef_t* refArray;
} mstxMemRegionsUnregisterBatch_t;

ACL_FUNC_VISIBILITY void mstxMarkA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY mstxRangeId mstxRangeStartA(const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY void mstxRangeEnd(mstxRangeId id);

ACL_FUNC_VISIBILITY mstxDomainHandle_t mstxDomainCreateA(const char* name);

ACL_FUNC_VISIBILITY void mstxDomainDestroy(mstxDomainHandle_t handle);

ACL_FUNC_VISIBILITY void mstxDomainMarkA(mstxDomainHandle_t handle, const char* message, aclrtStream stream);

ACL_FUNC_VISIBILITY mstxRangeId mstxDomainRangeStartA(mstxDomainHandle_t handle, const char* message,
                                                      aclrtStream stream);

ACL_FUNC_VISIBILITY void mstxDomainRangeEnd(mstxDomainHandle_t handle, mstxRangeId id);

ACL_FUNC_VISIBILITY mstxMemHeapHandle_t mstxMemHeapRegister(mstxDomainHandle_t domain, const mstxMemHeapDesc_t* desc);

ACL_FUNC_VISIBILITY void mstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);

ACL_FUNC_VISIBILITY void mstxMemRegionsRegister(mstxDomainHandle_t domain, const mstxMemRegionsRegisterBatch_t* desc);

ACL_FUNC_VISIBILITY void mstxMemRegionsUnregister(mstxDomainHandle_t domain, const mstxMemRegionsUnregisterBatch_t* desc);

#ifdef __cplusplus
}
#endif

#endif
