/**
 * @file prof_common.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2024. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */
#ifndef MSPROFILER_PROF_COMMON_H
#define MSPROFILER_PROF_COMMON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define MSPROF_DATA_HEAD_MAGIC_NUM  0x5A5AU
#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU
#define MSPROF_TASK_TIME_L0 0x00000800ULL  // mean PROF_TASK_TIME
#define MSPROF_EVENT_FLAG 0xFFFFFFFFFFFFFFFFULL
typedef void* VOID_PTR;
typedef const void* ConstVoidPtr;
typedef int32_t (*MsprofReportHandle)(uint32_t moduleId, uint32_t type, VOID_PTR data, uint32_t len);
typedef int32_t (*MsprofCtrlHandle)(uint32_t type, VOID_PTR data, uint32_t len);
typedef int32_t (*MsprofSetDeviceHandle)(VOID_PTR data, uint32_t len);
typedef int32_t (*MsprofCtrlCallback)(uint32_t type, void *data, uint32_t len);
typedef int32_t (*MsprofReporterCallback)(uint32_t moduleId, uint32_t type, void *data, uint32_t len);

/**
 * @name  ProfCommandHandle
 * @brief callback to start/stop profiling
 * @param type      [IN] enum call back type
 * @param data      [IN] callback data
 * @param len       [IN] callback data size
 * @return enum MsprofErrorCode
 */
typedef int32_t (*ProfCommandHandle)(uint32_t type, VOID_PTR data, uint32_t len);

/* Msprof report level */
#define MSPROF_REPORT_PYTORCH_LEVEL     30000U
#define MSPROF_REPORT_PTA_LEVEL         25000U
#define MSPROF_REPORT_ACL_LEVEL         20000U
#define MSPROF_REPORT_MODEL_LEVEL       15000U
#define MSPROF_REPORT_NODE_LEVEL        10000U
#define MSPROF_REPORT_AICPU_LEVEL       6000U
#define MSPROF_REPORT_HCCL_NODE_LEVEL   5500U
#define MSPROF_REPORT_RUNTIME_LEVEL     5000U
#define MSPROF_REPORT_PROF_LEVEL        4500U
#define MSPROF_REPORT_DPU_LEVEL         4000U

/* Msprof report type of acl(20000) level(acl), offset: 0x000000 */
#define MSPROF_REPORT_ACL_OP_BASE_TYPE            0x010000U
#define MSPROF_REPORT_ACL_MODEL_BASE_TYPE         0x020000U
#define MSPROF_REPORT_ACL_RUNTIME_BASE_TYPE       0x030000U
#define MSPROF_REPORT_ACL_OTHERS_BASE_TYPE        0x040000U


/* Msprof report type of acl(20000) level(host api), offset: 0x050000 */
#define MSPROF_REPORT_ACL_NN_BASE_TYPE            0x050000U
#define MSPROF_REPORT_ACL_ASCENDC_TYPE            0x060000U
#define MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE     0x070000U
#define MSPROF_REPORT_ACL_DVPP_BASE_TYPE          0x090000U
#define MSPROF_REPORT_ACL_GRAPH_BASE_TYPE         0x0A0000U

/* Msprof report type of model(15000) level, offset: 0x000000 */
#define MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE    0U         /* type info: graph_id_map */
#define MSPROF_REPORT_MODEL_EXECUTE_TYPE         1U         /* type info: execute */
#define MSPROF_REPORT_MODEL_LOAD_TYPE            2U         /* type info: load */
#define MSPROF_REPORT_MODEL_INPUT_COPY_TYPE      3U         /* type info: IntputCopy */
#define MSPROF_REPORT_MODEL_OUTPUT_COPY_TYPE     4U         /* type info: OutputCopy */
#define MSPROF_REPORT_MODEL_LOGIC_STREAM_TYPE    7U         /* type info: logic_stream_info */
#define MSPROF_REPORT_MODEL_EXEOM_TYPE           8U         /* type info: exeom */
#define MSPROF_REPORT_MODEL_UDF_BASE_TYPE        0x010000U  /* type info: udf_info */
#define MSPROF_REPORT_MODEL_AICPU_BASE_TYPE      0x020000U  /* type info: aicpu */

/* Msprof report type of node(10000) level, offset: 0x000000 */
#define MSPROF_REPORT_NODE_BASIC_INFO_TYPE       0U  /* type info: node_basic_info */
#define MSPROF_REPORT_NODE_TENSOR_INFO_TYPE      1U  /* type info: tensor_info */
#define MSPROF_REPORT_NODE_FUSION_OP_INFO_TYPE   2U  /* type info: funsion_op_info */
#define MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE  4U  /* type info: context_id_info */
#define MSPROF_REPORT_NODE_LAUNCH_TYPE           5U  /* type info: launch */
#define MSPROF_REPORT_NODE_TASK_MEMORY_TYPE      6U  /* type info: task_memory_info */
#define MSPROF_REPORT_NODE_HOST_OP_EXEC_TYPE     8U  /* type info: op exec */
#define MSPROF_REPORT_NODE_ATTR_INFO_TYPE        9U  /* type info: node_attr_info */
#define MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE    10U  /* type info: hccl_op_info */
#define MSPROF_REPORT_NODE_STATIC_OP_MEM_TYPE   11U  /* type info: static_op_mem */
#define MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE    12U  /* type info: mc2_comm_info */

/* Msprof report type of node(10000) level(ge api), offset: 0x010000 */
#define MSPROF_REPORT_NODE_GE_API_BASE_TYPE      0x010000U /* type info: ge api */
#define MSPROF_REPORT_NODE_HCCL_BASE_TYPE        0x020000U /* type info: hccl api */
#define MSPROF_REPORT_NODE_DVPP_API_BASE_TYPE    0x030000U /* type info: dvpp api */
/* Msprof report type of aicpu(6000), offset: 0x000000 */
#define MSPROF_REPORT_AICPU_NODE_TYPE               0U /* type info: DATA_PREPROCESS.AICPU */
#define MSPROF_REPORT_AICPU_DP_TYPE                 1U /* type info: DATA_PREPROCESS.DP */
#define MSPROF_REPORT_AICPU_MODEL_TYPE              2U /* type info: DATA_PREPROCESS.AICPU_MODEL */
#define MSPROF_REPORT_AICPU_MI_TYPE                 3U /* type info: DATA_PREPROCESS.AICPUMI */
#define MSPROF_REPORT_AICPU_MC2_EXECUTE_COMM_TIME   4U /* 通信时刻信息 */
#define MSPROF_REPORT_AICPU_MC2_EXECUTE_COMP_TIME   5U /* 计算时刻信息 */
#define MSPROF_REPORT_AICPU_MC2_HCCL_INFO           6U /* task信息 */

/* Msprof report type of hccl(5500) level(op api), offset: 0x010000 */
#define MSPROF_REPORT_HCCL_NODE_BASE_TYPE        0x010000U
#define MSPROF_REPORT_HCCL_MASTER_TYPE           0x010001U
#define MSPROF_REPORT_HCCL_SLAVE_TYPE            0x010002U

/* Msprof report type of hccl(4000U) level(dpu), offset: 0x000000 */
#define MSPROF_REPORT_DPU_TRACK_TYPE              0U /* type info: dpu_track */

/* use with AdprofCheckFeatureIsOn */
#define ADPROF_TASK_TIME_L0 0x00000008ULL
#define ADPROF_TASK_TIME_L1 0x00000010ULL
#define ADPROF_TASK_TIME_L2 0x00000020ULL

/* Msprof report type of profiling(4500) */
#define MSPROF_REPORT_DIAGNOSTIC_INFO_TYPE       0x010000U

enum ProfileCallbackType {
    PROFILE_CTRL_CALLBACK = 0,
    PROFILE_DEVICE_STATE_CALLBACK,
    PROFILE_REPORT_API_CALLBACK,
    PROFILE_REPORT_EVENT_CALLBACK,
    PROFILE_REPORT_COMPACT_CALLBACK,
    PROFILE_REPORT_ADDITIONAL_CALLBACK,
    PROFILE_REPORT_REG_TYPE_INFO_CALLBACK,
    PROFILE_REPORT_GET_HASH_ID_CALLBACK,
    PROFILE_HOST_FREQ_IS_ENABLE_CALLBACK,
    PROFILE_REPORT_API_C_CALLBACK,
    PROFILE_REPORT_EVENT_C_CALLBACK,
    PROFILE_REPORT_REG_TYPE_INFO_C_CALLBACK,
    PROFILE_REPORT_GET_HASH_ID_C_CALLBACK,
    PROFILE_HOST_FREQ_IS_ENABLE_C_CALLBACK,
};

enum MsprofDataTag {
    MSPROF_ACL_DATA_TAG = 0,            // acl data tag, range: 0~19
    MSPROF_GE_DATA_TAG_MODEL_LOAD = 20, // ge data tag, range: 20~39
    MSPROF_GE_DATA_TAG_FUSION = 21,
    MSPROF_GE_DATA_TAG_INFER = 22,
    MSPROF_GE_DATA_TAG_TASK = 23,
    MSPROF_GE_DATA_TAG_TENSOR = 24,
    MSPROF_GE_DATA_TAG_STEP = 25,
    MSPROF_GE_DATA_TAG_ID_MAP = 26,
    MSPROF_GE_DATA_TAG_HOST_SCH = 27,
    MSPROF_RUNTIME_DATA_TAG_API = 40,   // runtime data tag, range: 40~59
    MSPROF_RUNTIME_DATA_TAG_TRACK = 41,
    MSPROF_AICPU_DATA_TAG = 60,         // aicpu data tag, range: 60~79
    MSPROF_AICPU_MODEL_TAG = 61,
    MSPROF_HCCL_DATA_TAG = 80,          // hccl data tag, range: 80~99
    MSPROF_DP_DATA_TAG = 100,           // dp data tag, range: 100~119
    MSPROF_MSPROFTX_DATA_TAG = 120,     // hccl data tag, range: 120~139
    MSPROF_DATA_TAG_MAX = 65536,        // data tag value type is uint16_t
};

enum MsprofMindsporeNodeTag {
    GET_NEXT_DEQUEUE_WAIT = 1,
};

/**
 * @brief struct of mixed data
 */
#define MSPROF_MIX_DATA_RESERVE_BYTES 7
#define MSPROF_MIX_DATA_STRING_LEN 120
enum MsprofMixDataType {
    MSPROF_MIX_DATA_HASH_ID = 0,
    MSPROF_MIX_DATA_STRING,
};
struct MsprofMixData {
    uint8_t type;  // MsprofMixDataType
    uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
    union {
        uint64_t hashId;
        char dataStr[MSPROF_MIX_DATA_STRING_LEN];
    } data;
};

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

/**
 * @brief profiling command info
 */
#define MSPROF_MAX_DEV_NUM 64
struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MsprofCommandHandleParams params;
};

/**
 * @brief struct of data reported by acl
 */
#define MSPROF_ACL_DATA_RESERVE_BYTES 32
#define MSPROF_ACL_API_NAME_LEN 64
enum MsprofAclApiType {
    MSPROF_ACL_API_TYPE_OP = 1,
    MSPROF_ACL_API_TYPE_MODEL,
    MSPROF_ACL_API_TYPE_RUNTIME,
    MSPROF_ACL_API_TYPE_OTHERS,
};
struct MsprofAclProfData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_ACL_DATA_TAG;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t apiType;       // enum MsprofAclApiType
    uint64_t beginTime;
    uint64_t endTime;
    uint32_t processId;
    uint32_t threadId;
    char apiName[MSPROF_ACL_API_NAME_LEN];
    uint8_t  reserve[MSPROF_ACL_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by GE
 */
#define MSPROF_GE_MODELLOAD_DATA_RESERVE_BYTES 104
struct MsprofGeProfModelLoadData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_MODEL_LOAD;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t modelId;
    struct MsprofMixData modelName;
    uint64_t startTime;
    uint64_t endTime;
    uint8_t  reserve[MSPROF_GE_MODELLOAD_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_FUSION_DATA_RESERVE_BYTES 8
#define MSPROF_GE_FUSION_OP_NUM 8
struct MsprofGeProfFusionData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_FUSION;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t modelId;
    struct MsprofMixData fusionName;
    uint64_t inputMemSize;
    uint64_t outputMemSize;
    uint64_t weightMemSize;
    uint64_t workspaceMemSize;
    uint64_t totalMemSize;
    uint64_t fusionOpNum;
    uint64_t fusionOp[MSPROF_GE_FUSION_OP_NUM];
    uint8_t  reserve[MSPROF_GE_FUSION_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_INFER_DATA_RESERVE_BYTES 64
struct MsprofGeProfInferData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_INFER;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t modelId;
    struct MsprofMixData modelName;
    uint32_t requestId;
    uint32_t threadId;
    uint64_t inputDataStartTime;
    uint64_t inputDataEndTime;
    uint64_t inferStartTime;
    uint64_t inferEndTime;
    uint64_t outputDataStartTime;
    uint64_t outputDataEndTime;
    uint8_t  reserve[MSPROF_GE_INFER_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_TASK_DATA_RESERVE_BYTES 12
#define MSPROF_GE_OP_TYPE_LEN 56
enum MsprofGeTaskType {
    MSPROF_GE_TASK_TYPE_AI_CORE = 0,
    MSPROF_GE_TASK_TYPE_AI_CPU,
    MSPROF_GE_TASK_TYPE_AIV,
    MSPROF_GE_TASK_TYPE_WRITE_BACK,
    MSPROF_GE_TASK_TYPE_MIX_AIC,
    MSPROF_GE_TASK_TYPE_MIX_AIV,
    MSPROF_GE_TASK_TYPE_FFTS_PLUS,
    MSPROF_GE_TASK_TYPE_DSA,
    MSPROF_GE_TASK_TYPE_DVPP,
    MSPROF_GE_TASK_TYPE_HCCL,
    MSPROF_GE_TASK_TYPE_FUSION,
    MSPROF_GE_TASK_TYPE_INVALID
};

enum MsprofGeShapeType {
    MSPROF_GE_SHAPE_TYPE_STATIC = 0,
    MSPROF_GE_SHAPE_TYPE_DYNAMIC,
};
struct MsprofGeOpType {
    uint8_t type;  // MsprofMixDataType
    uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
    union {
        uint64_t hashId;
        char dataStr[MSPROF_GE_OP_TYPE_LEN];
    } data;
};
struct MsprofGeProfTaskData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_TASK;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t taskType;      // MsprofGeTaskType
    struct MsprofMixData opName;
    struct MsprofGeOpType opType;
    uint64_t curIterNum;
    uint64_t timeStamp;
    uint32_t shapeType;     // MsprofGeShapeType
    uint32_t blockDims;
    uint32_t modelId;
    uint32_t streamId;
    uint32_t taskId;
    uint32_t threadId;
    uint32_t contextId;
    uint8_t  reserve[MSPROF_GE_TASK_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_TENSOR_DATA_RESERVE_BYTES 8
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5
enum MsprofGeTensorType {
    MSPROF_GE_TENSOR_TYPE_INPUT = 0,
    MSPROF_GE_TENSOR_TYPE_OUTPUT,
};
struct MsprofGeTensorData {
    uint32_t tensorType;    // MsprofGeTensorType
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofGeProfTensorData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_TENSOR;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t modelId;
    uint64_t curIterNum;
    uint32_t streamId;
    uint32_t taskId;
    uint32_t tensorNum;
    struct MsprofGeTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
    uint8_t  reserve[MSPROF_GE_TENSOR_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_STEP_DATA_RESERVE_BYTES 27
enum MsprofGeStepTag {
    MSPROF_GE_STEP_TAG_BEGIN = 0,
    MSPROF_GE_STEP_TAG_END,
};
struct MsprofGeProfStepData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_STEP;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t modelId;
    uint32_t streamId;
    uint32_t taskId;
    uint64_t timeStamp;
    uint64_t curIterNum;
    uint32_t threadId;
    uint8_t  tag;           // MsprofGeStepTag
    uint8_t  reserve[MSPROF_GE_STEP_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_ID_MAP_DATA_RESERVE_BYTES 6
struct MsprofGeProfIdMapData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_ID_MAP;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t graphId;
    uint32_t modelId;
    uint32_t sessionId;
    uint64_t timeStamp;
    uint16_t mode;
    uint8_t  reserve[MSPROF_GE_ID_MAP_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_HOST_SCH_DATA_RESERVE_BYTES 24
struct MsprofGeProfHostSchData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_HOST_SCH;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t threadId;      // record in start event
    uint64_t element;
    uint64_t event;
    uint64_t startTime;     // record in start event
    uint64_t endTime;       // record in end event
    uint8_t  reserve[MSPROF_GE_HOST_SCH_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by RunTime
 */
#define MSPROF_AICPU_DATA_RESERVE_BYTES 9
struct MsprofAicpuProfData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_AICPU_DATA_TAG;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint16_t streamId;
    uint16_t taskId;
    uint64_t runStartTime;
    uint64_t runStartTick;
    uint64_t computeStartTime;
    uint64_t memcpyStartTime;
    uint64_t memcpyEndTime;
    uint64_t runEndTime;
    uint64_t runEndTick;
    uint32_t threadId;
    uint32_t deviceId;
    uint64_t submitTick;
    uint64_t scheduleTick;
    uint64_t tickBeforeRun;
    uint64_t tickAfterRun;
    uint32_t kernelType;
    uint32_t dispatchTime;
    uint32_t totalTime;
    uint16_t fftsThreadId;
    uint8_t  version;
    uint8_t  reserve[MSPROF_AICPU_DATA_RESERVE_BYTES];
};

struct MsprofAicpuModelProfData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_AICPU_MODEL_TAG;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t rsv;   // Ensure 8-byte alignment
    uint64_t timeStamp;
    uint64_t indexId;
    uint32_t modelId;
    uint16_t tagId;
    uint16_t rsv1;
    uint64_t eventId;
    uint8_t  reserve[24];
};

/**
 * @brief struct of data reported by DP
 */
#define MSPROF_DP_DATA_RESERVE_BYTES 16
#define MSPROF_DP_DATA_ACTION_LEN 16
#define MSPROF_DP_DATA_SOURCE_LEN 64
#define MSPROF_CTX_ID_MAX_NUM 55

struct MsprofDpProfData {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_DP_DATA_TAG;
#else
    uint16_t magicNumber;
    uint16_t dataTag;
#endif
    uint32_t rsv;   // Ensure 8-byte alignment
    uint64_t timeStamp;
    char action[MSPROF_DP_DATA_ACTION_LEN];
    char source[MSPROF_DP_DATA_SOURCE_LEN];
    uint64_t index;
    uint64_t size;
    uint8_t  reserve[MSPROF_DP_DATA_RESERVE_BYTES];
};

struct MsprofAicpuNodeAdditionalData {
    uint16_t streamId;
    uint16_t taskId;
    uint64_t runStartTime;
    uint64_t runStartTick;
    uint64_t computeStartTime;
    uint64_t memcpyStartTime;
    uint64_t memcpyEndTime;
    uint64_t runEndTime;
    uint64_t runEndTick;
    uint32_t threadId;
    uint32_t deviceId;
    uint64_t submitTick;
    uint64_t scheduleTick;
    uint64_t tickBeforeRun;
    uint64_t tickAfterRun;
    uint32_t kernelType;
    uint32_t dispatchTime;
    uint32_t totalTime;
    uint16_t fftsThreadId;
    uint8_t version;
    uint8_t reserve[MSPROF_AICPU_DATA_RESERVE_BYTES];
};

struct MsprofAicpuModelAdditionalData {
    uint64_t indexId;
    uint32_t modelId;
    uint16_t tagId;
    uint16_t rsv1;
    uint64_t eventId;
    uint8_t reserve[24];
};

struct MsprofAicpuDpAdditionalData {
    char action[MSPROF_DP_DATA_ACTION_LEN];
    char source[MSPROF_DP_DATA_SOURCE_LEN];
    uint64_t index;
    uint64_t size;
    uint8_t reserve[MSPROF_DP_DATA_RESERVE_BYTES];
};

struct MsprofAicpuMiAdditionalData {
    uint32_t nodeTag;  // MsprofMindsporeNodeTag:1
    uint32_t reserve;
    uint64_t queueSize;
    uint64_t runStartTime;
    uint64_t runEndTime;
};

// AICPU kfc算子执行时间
struct AicpuKfcProfCommTurn {
    uint64_t waitNotifyStartTime;  // 开始等待通信参数
    uint64_t kfcAlgExeStartTime;   // 开始通信算法执行
    uint64_t sendTaskStartTime;    // 开始下发task
    uint64_t waitActiveStartTime;  // 开始等待激活
    uint64_t acitveStartTime;      // 开始激活处理
    uint64_t waitExeEndStartTime;  // 开始等待任务执行结束
    uint64_t rtsqExeEndTime;       // 任务执行结束时间
    uint64_t dataLen;              // 本轮通信数据长度
    uint32_t deviceId;
    uint16_t streamId;
    uint16_t taskId;
    uint8_t version;
    uint8_t commTurn;  // 总通信轮次
    uint8_t currentTurn;
    uint8_t reserve[5];
};

// Aicore算子执行时间
struct AicpuKfcProfComputeTurn {
    uint64_t waitComputeStartTime;  // 开始等待计算
    uint64_t computeStartTime;      // 开始计算
    uint64_t computeExeEndTime;     // 计算执行结束
    uint64_t dataLen;               // 本轮计算数据长度
    uint32_t deviceId;
    uint16_t streamId;
    uint16_t taskId;
    uint8_t version;
    uint8_t computeTurn;  // 总计算轮次
    uint8_t currentTurn;
    uint8_t reserve[5];
};

/**
 * @brief struct of data reported by HCCL
 */
#pragma pack(4)
struct MsprofHcclProfNotify {
    uint32_t taskID;
    uint64_t notifyID;
    uint32_t stage;
    uint32_t remoteRank;
    uint32_t transportType;
    uint32_t role; // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfReduce {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint32_t op;            // {0: sum, 1: mul, 2: max, 3: min}
    uint32_t dataType;      // data type {0: INT8, 1: INT16, 2: INT32, 3: FP16, 4:FP32, 5:INT64, 6:UINT64}
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: SDMA, 1: RDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfRDMA {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t notifyID;
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: RDMA, 1:SDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    uint32_t type;          // RDMA type {0: RDMASendNotify, 1:RDMASendPayload}
    double durationEstimated;
};

struct MsprofHcclProfMemcpy {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t notifyID;
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: RDMA, 1:SDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfStageStep {
    uint32_t rank;
    uint32_t rankSize;
};

struct MsprofHcclProfFlag {
    uint64_t cclTag;
    uint64_t groupName;
    uint32_t localRank;
    uint32_t workFlowMode;
};

#define MSPROF_HCCL_INVALID_UINT 0xFFFFFFFFU
struct MsprofHcclInfo {
    uint64_t itemId;
    uint64_t cclTag;
    uint64_t groupName;
    uint32_t localRank;
    uint32_t remoteRank;
    uint32_t rankSize;
    uint32_t workFlowMode;
    uint32_t planeID;
    uint32_t ctxId;
    uint64_t notifyID;
    uint32_t stage;
    uint32_t role; // role {0: dst, 1:src}
    double durationEstimated;
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint64_t dataSize; // bytes
    uint32_t opType; // {0: sum, 1: mul, 2: max, 3: min}
    uint32_t dataType; // data type {0: INT8, 1: INT16, 2: INT32, 3: FP16, 4:FP32, 5:INT64, 6:UINT64}
    uint32_t linkType; // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t transportType; // transport type {0: SDMA, 1: RDMA, 2:LOCAL}
    uint32_t rdmaType; // RDMA type {0: RDMASendNotify, 1:RDMASendPayload}
    uint32_t reserve2;
#ifdef __cplusplus
    MsprofHcclInfo() : role(MSPROF_HCCL_INVALID_UINT), opType(MSPROF_HCCL_INVALID_UINT),
        dataType(MSPROF_HCCL_INVALID_UINT), linkType(MSPROF_HCCL_INVALID_UINT),
        transportType(MSPROF_HCCL_INVALID_UINT), rdmaType(MSPROF_HCCL_INVALID_UINT)
    {
    }
#endif
};

struct MsprofAicpuMC2HcclInfo {
    uint64_t itemId;
    uint64_t cclTag;
    uint64_t groupName;
    uint32_t localRank;
    uint32_t remoteRank;
    uint32_t rankSize;
    uint32_t workFlowMode;
    uint32_t planeID;
    uint32_t ctxId;
    uint64_t notifyID;
    uint32_t stage;
    uint32_t role; // role {0: dst, 1:src}
    double durationEstimated;
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint64_t dataSize; // bytes
    uint32_t opType; // {0: sum, 1: mul, 2: max, 3: min}
    uint32_t dataType; // data type {0: INT8, 1: INT16, 2: INT32, 3: FP16, 4:FP32, 5:INT64, 6:UINT64}
    uint32_t linkType; // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t transportType; // transport type {0: SDMA, 1: RDMA, 2:LOCAL}
    uint32_t rdmaType; // RDMA type {0: RDMASendNotify, 1:RDMASendPayload}
    uint32_t taskId;
    uint16_t streamId;
    uint16_t reserve[3];
};

struct ProfilingDeviceCommResInfo {
    uint64_t groupName; // 通信域
    uint32_t rankSize; // 通信域内rank总数
    uint32_t rankId; // 当前device rankId，通信域内编号
    uint32_t usrRankId; // 当前device rankId，全局编号
    uint32_t aicpuKfcStreamId; // MC2中launch aicpu kfc算子的stream
    uint32_t commStreamSize; // 当前device侧使用的通信stream数量
    uint32_t commStreamIds[8]; // 具体streamId
    uint32_t reserve;
};

#define MSPROF_MULTI_THREAD_MAX_NUM 25
struct MsprofMultiThread {
    uint32_t threadNum;
    uint32_t threadId[MSPROF_MULTI_THREAD_MAX_NUM];
};
#pragma pack()


#pragma pack(1)
struct MsprofNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};

enum AttrType {
    OP_ATTR = 0,
};

struct MsprofAttrInfo {
    uint64_t opName;
    uint32_t attrType;
    uint64_t hashId;
};

struct MsrofTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MsrofTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
};

struct MsprofHCCLOPInfo {  // for MsprofReportCompactInfo buffer data
    uint8_t relay : 1;     // 借轨通信
    uint8_t retry : 1;     // 重传标识
    uint8_t dataType;      // 跟HcclDataType类型保存一致
    uint16_t algType;      // algtype 每4位表示一个算法阶段
    uint64_t count;        // 发送数据个数
    uint64_t groupName;    // group hash id
};

struct ProfFusionOpInfo {
uint64_t opName;
uint32_t fusionOpNum;
uint64_t inputMemsize;
uint64_t outputMemsize;
uint64_t weightMemSize;
uint64_t workspaceMemSize;
uint64_t totalMemSize;
uint64_t fusionOpId[MSPROF_GE_FUSION_OP_NUM];
};

struct MsprofContextIdInfo {
    uint64_t opName;
    uint32_t ctxIdNum;
    uint32_t ctxIds[MSPROF_CTX_ID_MAX_NUM];
};

struct MsprofGraphIdInfo {
    uint64_t modelName;
    uint32_t graphId;
    uint32_t modelId;
};

struct MsprofMemoryInfo {
    uint64_t addr;
    int64_t size;
    uint64_t nodeId; // op name hash id
    uint64_t totalAllocateMemory;
    uint64_t totalReserveMemory;
    uint32_t deviceId;
    uint32_t deviceType;
};

/**
 * @name  MsprofStampInfo
 * @brief struct of data reported by msproftx
 */
struct MsprofStampInfo {
    uint16_t magicNumber;
    uint16_t dataTag;
    uint32_t processId;
    uint32_t threadId;
    uint32_t category;    // marker category
    uint32_t eventType;
    int32_t payloadType;
    union PayloadValue {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        uint32_t uiValue[2];
        int32_t iValue[2];
        float fValue[2];
    } payload;            // payload info for marker
    uint64_t startTime;
    uint64_t endTime;
    int32_t messageType;
    char message[128];
};

struct MsprofStaticOpMem {
    int64_t size;        // op memory size
    uint64_t opName;     // op name hash id
    uint64_t lifeStart;  // serial number of op memory used
    uint64_t lifeEnd;    // serial number of op memory used
    uint64_t totalAllocateMemory; // static graph total allocate memory
    uint64_t dynOpName;  // 0: invalid， other： dynamic op name of root
    uint32_t graphId;    // multipe model
};

#define MSPROF_PHYSIC_STREAM_ID_MAX_NUM 56
struct MsprofLogicStreamInfo {
    uint32_t logicStreamId;
    uint32_t physicStreamNum;
    uint32_t physicStreamId[MSPROF_PHYSIC_STREAM_ID_MAX_NUM];
};

struct MsprofExeomLoadInfo {
    uint32_t modelId;
    uint32_t reserve;
    uint64_t modelName; /* name hash */
};
#pragma pack()

struct MsprofApi { // for MsprofReportApi
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

struct MsprofEvent {  // for MsprofReportEvent
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t requestId; // 0xFFFF means single event
    uint64_t timeStamp;
#ifdef __cplusplus
    uint64_t eventFlag = MSPROF_EVENT_FLAG;
#else
    uint64_t eventFlag;
#endif
    uint64_t itemId;
};

struct MsprofRuntimeTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;       // task message hash id
};

struct MsprofDpuTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;   // high 4 bits, devType: dpu: 1, low 12 bits device id
    uint16_t streamId;
    uint32_t taskId;
    uint32_t taskType;    // task type enum
    uint32_t res;
    uint64_t startTime;   // start time
};

#define MSPROF_COMPACT_INFO_DATA_LENGTH (40)
struct MsprofCompactInfo {  // for MsprofReportCompactInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        struct MsprofRuntimeTrack runtimeTrack;
        struct MsprofNodeBasicInfo nodeBasicInfo;
        struct MsprofHCCLOPInfo hcclopInfo;
        struct MsprofDpuTrack dpuTack;
    } data;
};

#define MSPROF_ADDTIONAL_INFO_DATA_LENGTH (232)
struct MsprofAdditionalInfo {  // for MsprofReportAdditionalInfo buffer data
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t  data[MSPROF_ADDTIONAL_INFO_DATA_LENGTH];
};

/**
 * @name  MsprofErrorCode
 * @brief error code
 */
enum MsprofErrorCode {
    MSPROF_ERROR_NONE = 0,
    MSPROF_ERROR_MEM_NOT_ENOUGH,
    MSPROF_ERROR_GET_ENV,
    MSPROF_ERROR_CONFIG_INVALID,
    MSPROF_ERROR_ACL_JSON_OFF,
    MSPROF_ERROR,
    MSPROF_ERROR_UNINITIALIZE,
};

#define MSPROF_ENGINE_MAX_TAG_LEN (63)

/**
 * @name  ReporterData
 * @brief struct of data to report
 */
struct ReporterData {
    char tag[MSPROF_ENGINE_MAX_TAG_LEN + 1];  // the sub-type of the module, data with different tag will be writen
    int32_t deviceId;                         // the index of device
    size_t dataLen;                           // the length of send data
    uint8_t *data;                            // the data content
};

/**
 * @name  MsprofHashData
 * @brief struct of data to hash
 */
struct MsprofHashData {
    int32_t deviceId;                         // the index of device
    size_t dataLen;                           // the length of data
    uint8_t *data;                            // the data content
    uint64_t hashId;                          // the id of hashed data
};

enum MsprofConfigParamType {
    DEV_CHANNEL_RESOURCE = 0,          // device channel resource
    HELPER_HOST_SERVER                 // helper host server
};

/**
 * @name  MsprofConfigParam
 * @brief struct of set config
 */
struct MsprofConfigParam {
    uint32_t deviceId;                        // the index of device
    uint32_t type;                            // DEV_CHANNEL_RESOURCE; HELPER_HOST_SERVER
    uint32_t value;                           // DEV_CHANNEL_RESOURCE: 1 off; HELPER_HOST_SERVER: 1 on
};

/**
 * @name  MsprofReporterModuleId
 * @brief module id of data to report
 */
enum MsprofReporterModuleId {
    MSPROF_MODULE_DATA_PREPROCESS = 0,    // DATA_PREPROCESS
    MSPROF_MODULE_HCCL,                   // HCCL
    MSPROF_MODULE_ACL,                    // AclModule
    MSPROF_MODULE_FRAMEWORK,              // Framework
    MSPROF_MODULE_RUNTIME,                // runtime
    MSPROF_MODULE_MSPROF                  // msprofTx
};

/**
 * @name  MsprofReporterCallbackType
 * @brief reporter callback request type
 */
enum MsprofReporterCallbackType {
    MSPROF_REPORTER_REPORT = 0,           // report data
    MSPROF_REPORTER_INIT,                 // init reporter
    MSPROF_REPORTER_UNINIT,               // uninit reporter
    MSPROF_REPORTER_DATA_MAX_LEN,         // data max length for calling report callback
    MSPROF_REPORTER_HASH                  // hash data to id
};

#define MSPROF_OPTIONS_DEF_LEN_MAX (2048U)

/**
 * @name  MsprofGeOptions
 * @brief struct of MSPROF_CTRL_INIT_GE_OPTIONS
 */
struct MsprofGeOptions {
    char jobId[MSPROF_OPTIONS_DEF_LEN_MAX];
    char options[MSPROF_OPTIONS_DEF_LEN_MAX];
};

/**
 * @name  MsprofCtrlCallbackType
 * @brief ctrl callback request type
 */
enum MsprofCtrlCallbackType {
    MSPROF_CTRL_INIT_ACL_ENV = 0,           // start profiling with acl env
    MSPROF_CTRL_INIT_ACL_JSON = 1,          // start pro with acl.json
    MSPROF_CTRL_INIT_GE_OPTIONS = 2,        // start profiling with ge env and options
    MSPROF_CTRL_FINALIZE = 3,               // stop profiling
    MSPROF_CTRL_INIT_HELPER = 4,            // start profiling in helper device
    MSPROF_CTRL_INIT_PURE_CPU = 5,          // start profiling in pure cpu
    MSPROF_CTRL_INIT_DYNA = 0xFF,           // start profiling for dynamic profiling
};

enum MsprofCommandHandleType {
    PROF_COMMANDHANDLE_TYPE_INIT = 0,
    PROF_COMMANDHANDLE_TYPE_START,
    PROF_COMMANDHANDLE_TYPE_STOP,
    PROF_COMMANDHANDLE_TYPE_FINALIZE,
    PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MAX
};

enum MsprofConfigType {
    MSPROF_CONFIG_HELPER_HOST = 0
};

/**
 * @brief   profiling command type
 */
enum ProfCtrlType {
    PROF_CTRL_INVALID = 0,
    PROF_CTRL_SWITCH,
    PROF_CTRL_REPORTER,
    PROF_CTRL_STEPINFO,
    PROF_CTRL_BUTT
};

/**
 * @brief   Prof Chip ID
 */
enum Prof_Chip_ID {
    PROF_CHIP_ID0 = 0
};

/**
 * @brief  the struct of profiling set setp info
 */
typedef struct ProfStepInfoCmd {
    uint64_t index_id;
    uint16_t tag_id;
    void *stream;
} ProfStepInfoCmd_t;

#ifdef __cplusplus
}
#endif
#endif  // MSPROFILER_PROF_COMMON_H_
