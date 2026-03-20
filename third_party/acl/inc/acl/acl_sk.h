#ifndef ACL_SUPERKERNEL_H
#define ACL_SUPERKERNEL_H

#include <cstdint>
#include <cstddef>
#include "acl.h"

enum class aclskOptionType : uint32_t {
    PRELOAD_CODE = 0,
    SPLIT_MODE = 1,
    STREAM_FUSION = 2,
    DEBUG_DCCI_DISABLE_ON_KERNEL = 3,
    DEBUG_SYNC_ALL = 4,
    KERNEL_MAP = 5,
    SK_OPTION_MAX = 0xFFFFFFFF
};

struct aclskPreloadOption {
    uint32_t preloadMode;
};

struct aclskSplitModeOption {
    uint32_t splitCnt;
};

struct aclskStreamFusionOption {
    uint32_t streamFusion;
};

struct aclskDcciOption {
    char** kernelNames;
    size_t kernelCnt;
};

struct aclskDebugSyncAllOption {
    uint32_t debugSyncAll;
};

struct aclskDebugDcciAllOption {
    uint32_t debugDcciAll;
};

struct aclskKernelMap {
    char* globalName;
    char* sknlNames[4];
};

struct aclskKernelMapOption {
    aclskKernelMap* kernelMaps;
    size_t numKernels;
};

struct aclskOption {
    aclskOptionType optionType;
    union {
        aclskPreloadOption preload;
        aclskSplitModeOption splitMode;
        aclskStreamFusionOption streamFusion;
        aclskDcciOption dcciBeforeKernelStart;
        aclskDcciOption dcciAfterKernelEnd;
        aclskDcciOption disableKernelDcci;
        aclskDebugSyncAllOption debugSync;
        aclskDebugDcciAllOption debugDcci;
        aclskKernelMapOption kernelMap;
    };
};

typedef struct aclskOptions {
    aclskOption *options;
    size_t numOptions;
};

ACL_FUNC_VISIBILITY aclError aclskOptimize(aclmdlRI model, aclskOptions *options);

#endif