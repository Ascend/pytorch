#ifndef ACL_SUPERKERNEL_H
#define ACL_SUPERKERNEL_H

#include <cstdint>
#include <cstddef>
#include "acl.h"

enum class aclskOptionType : uint32_t {
    PRELOAD_CODE = 0,
    SPLIT_MODE = 1,
    STREAM_FUSION = 2,
    DCCI_DISABLE_ON_KERNEL = 3,
    DEBUG_SYNC_ALL = 4,
    KERNEL_MAP = 5,
    CONSTANT_CODEGEN = 6,  // 常量化代码生成选项
    AUTO_OP_PARALLEL = 7,  // 优化多流算子排布
    DCCI_BEFORE_KERNEL_START = 8,
    DEBUG_OP_EXEC_TRACE = 9,
    DEBUG_CROSS_CORE_SYNC_CHECK = 10,
    OPT_EXTEND_OPTION = 11,   // 扩展选项，预留后续使用
    DEBUG_EXTEND_OPTION = 12, // 扩展选项，预留后续使用
    DCCI_AFTER_KERNEL_END = 13,
    AGGRESSIVE_OPT_STRATEGIES = 14, // value memory wait breaker policy
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

struct aclskAutoOpParallelOption {
    uint32_t enableAutoOpParallel;
};

struct aclskDebugOpExecTraceOption {
    uint32_t enableOpExecTrace;
};

struct aclskDebugCrossCoreSyncCheckOption {
    uint32_t enableCrossCoreSyncCheck;
};

struct aclskExtendOption {
    char* value;
};

/**
 * aggressiveOpts.valueBreakerBypass is a bitmask-style value-memory wait policy:
 *  - ACLSK_VALUE_BREAKER_BYPASS_NONE (0b00): reject write/wait pairing relation, keep wait unfusible
 *  - ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT (0b01): for paired write+wait, existing rule must pass or SK exits
 *  - ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT (0b10): for waits whose writes are outside modelRI, allow wait fusion
 */
enum aclskValueBreakerBypassFlag : uint32_t {
    ACLSK_VALUE_BREAKER_BYPASS_NONE = 0b00U,
    ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT = 0b01U,
    ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT = 0b10U
};

struct aclskAggressiveOptStrategies {
    uint32_t eventBreakerBypass;
    uint32_t valueBreakerBypass;
    uint32_t taskBreakerBypass;
};

/**
 * 常量化代码生成选项
 * enableConstant: 1 启用常量化, 0 禁用常量化
 */
struct aclskConstantCodegenOption {
    uint32_t enableConstant;
};

struct aclskOption {
    aclskOptionType optionType;
    union {
        aclskPreloadOption preload;
        aclskSplitModeOption splitMode;
        aclskStreamFusionOption streamFusion;
        aclskDcciOption disableKernelDcci;
        aclskDebugSyncAllOption debugSync;
        aclskKernelMapOption kernelMap;
        aclskConstantCodegenOption constantCodegen;
        aclskAutoOpParallelOption autoOpParallel;
        aclskDcciOption dcciBeforeKernelStart;
        aclskDebugOpExecTraceOption debugOpExecTrace;
        aclskDebugCrossCoreSyncCheckOption debugCrossCoreSyncCheck;
        aclskExtendOption optExtend;
        aclskExtendOption debugExtend;
        aclskDcciOption dcciAfterKernelEnd;
        aclskAggressiveOptStrategies aggressiveOpts;
    };
};

typedef struct aclskOptions {
    aclskOption *options;
    size_t numOptions;
};

ACL_FUNC_VISIBILITY aclError aclskOptimize(aclmdlRI model, aclskOptions *options);

#endif