/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TYPES_H_
#define HCCL_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief HCCL functions return value definition
 */
typedef enum {
    HCCL_SUCCESS = 0,               /**< success */
    HCCL_E_PARA = 1,                /**< parameter error */
    HCCL_E_PTR = 2,                 /**< empty pointer */
    HCCL_E_MEMORY = 3,              /**< memory error */
    HCCL_E_INTERNAL = 4,            /**< internal error */
    HCCL_E_NOT_SUPPORT = 5,         /**< not support feature */
    HCCL_E_NOT_FOUND = 6,           /**< not found specific resource */
    HCCL_E_UNAVAIL = 7,             /**< resource unavailable */
    HCCL_E_SYSCALL = 8,             /**< call system interface error */
    HCCL_E_TIMEOUT = 9,             /**< timeout */
    HCCL_E_OPEN_FILE_FAILURE = 10,  /**< open file fail */
    HCCL_E_TCP_CONNECT = 11,        /**< tcp connect fail */
    HCCL_E_ROCE_CONNECT = 12,       /**< roce connect fail */
    HCCL_E_TCP_TRANSFER = 13,       /**< tcp transfer fail */
    HCCL_E_ROCE_TRANSFER = 14,      /**< roce transfer fail */
    HCCL_E_RUNTIME = 15,            /**< call runtime api fail */
    HCCL_E_DRV = 16,                /**< call driver api fail */
    HCCL_E_PROFILING = 17,          /**< call profiling api fail */
    HCCL_E_CCE = 18,                /**< call cce api fail */
    HCCL_E_NETWORK = 19,            /**< call network api fail */
    HCCL_E_AGAIN = 20,              /**< try again */
    HCCL_E_REMOTE = 21,             /**< error cqe */
    HCCL_E_SUSPENDING = 22,         /**< error communicator suspending */
    HCCL_E_OPRETRY_FAIL = 23,       /**< retry constraint */
    HCCL_E_OOM = 24,                /**< out of memory */
    HCCL_E_RESERVED                 /**< reserved */
} HcclResult;

/**
 * @brief handle to HCCL communicator
 */
typedef void *HcclComm;

/**
 * @brief HCCL Reduction opperation
 */
typedef enum {
    HCCL_REDUCE_SUM = 0,    /**< sum */
    HCCL_REDUCE_PROD = 1,   /**< prod */
    HCCL_REDUCE_MAX = 2,    /**< max */
    HCCL_REDUCE_MIN = 3,    /**< min */
    HCCL_REDUCE_RESERVED = 255 /**< reserved */
} HcclReduceOp;

/**
 * @brief HCCL data type
 */
typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,    /**< int8 */
    HCCL_DATA_TYPE_INT16 = 1,   /**< int16 */
    HCCL_DATA_TYPE_INT32 = 2,   /**< int32 */
    HCCL_DATA_TYPE_FP16 = 3,    /**< fp16 */
    HCCL_DATA_TYPE_FP32 = 4,    /**< fp32 */
    HCCL_DATA_TYPE_INT64 = 5,    /**< int64 */
    HCCL_DATA_TYPE_UINT64 = 6,    /**< uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,    /**< uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,   /**< uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,   /**< uint32 */
    HCCL_DATA_TYPE_FP64 = 10,    /**< fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,    /**< bfp16 */
    HCCL_DATA_TYPE_INT128 = 12,   /**< int128 */
    HCCL_DATA_TYPE_HIF8 = 14,     /**< hif8 */
    HCCL_DATA_TYPE_FP8E4M3 = 15,  /**< fp8e4m3 */
    HCCL_DATA_TYPE_FP8E5M2 = 16,  /**< fp8e5m2 */
    HCCL_DATA_TYPE_FP8E8M0 = 17,  /**< fp8e8m0 */
    HCCL_DATA_TYPE_RESERVED = 255 /**< reserved */
} HcclDataType;

typedef enum {
    HCCL_DETERMINISTIC = 0,     /**< 0: non-deterministic, 1: deterministic */
#ifndef OPEN_BUILD_PROJECT
    HCCL_ACCELERATOR,           /**< 0: default, 1: CCU, 2: AIV, 3: AICPU_TS, 4: HOSTCPU_TS, 5: AICPU */
#endif
    HCCL_CONFIG_RESERVED
} HcclConfig;

union HcclConfigValue {
    int32_t value;
};

const uint32_t HCCL_ROOT_INFO_BYTES =  4108; // 4108: root info length
const uint32_t BUFFER_NAME_MAX_LENGTH = 128; // cclbuffer name max length
const uint32_t COMM_NAME_MAX_LENGTH = 128; // group name max length
const uint32_t UDI_MAX_LENGTH = 128; // UDI max length
const uint32_t HCCL_COMM_ALGO_MAX_LENGTH = 1600; // hccl algo max length
const uint32_t HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50; // hccl_retry_enable max length
const uint32_t HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128; // hccl_retry_params max length
/**
 * @brief HCCL root info
 */
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

const uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
const uint32_t HCCL_COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;
const uint32_t HCCL_COMM_CONFIG_VERSION = 10;
const uint32_t HCCL_COMM_DEFAULT_BUFFSIZE = 200;
const uint32_t HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_DEFAULT_DETERMINISTIC = 0;
const uint32_t HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0;
// 0xffffffff表示用户未配置TC或SL
const uint32_t HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xffffffff;
const int32_t HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET = 0xffffffff;

typedef struct HcclCommConfigDef {
    char reserved[HCCL_COMM_CONFIG_INFO_BYTES];
    uint32_t hcclBufferSize;
    uint32_t hcclDeterministic;
    char hcclCommName[COMM_NAME_MAX_LENGTH];
    char hcclUdi[UDI_MAX_LENGTH];
    uint32_t hcclOpExpansionMode;   // 0:默认值  1:host  2:aicpu  3:aiv
    uint32_t hcclRdmaTrafficClass;
    uint32_t hcclRdmaServiceLevel;
    uint32_t hcclWorldRankID;
    uint64_t hcclJobID;
    uint8_t aclGraphZeroCopyEnable; // 只有Reduce类算子(单算子和AclGraph下算法选择不一致)受此配置影响 0:默认值，关闭aclgraph零拷贝(结果与单算子一致优先) 1:开启aclgraph零拷贝(性能优先)
    int32_t hcclExecTimeOut; // hccl执行超时时间
    char hcclAlgo[HCCL_COMM_ALGO_MAX_LENGTH];
    char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
    char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
    char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
    uint32_t hcclQos;
} HcclCommConfig;

typedef enum {
    HCCL_COMM_CONFIG_BUFFER_SIZE = 0,
    HCCL_COMM_CONFIG_DETERMINISTIC = 1,
    HCCL_COMM_CONFIG_COMM_NAME = 2,
    HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3,
    HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV = 4,
    HCCL_COMM_CONFIG_WORLD_RANKID = 5,
    HCCL_COMM_CONFIG_JOBID = 6,
    HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE = 7,
    HCCL_COMM_CONFIG_EXEC_TIMEOUT = 8,
    HCCL_COMM_CONFIG_ALGO = 9,
    HCCL_COMM_CONFIG_RETRY = 10,
    HCCL_COMM_CONFIG_BUFFER_NAME = 11,
    HCCL_COMM_CONFIG_RESERVED
} HcclCommConfigCapability;

typedef enum {
    HCCL_SEND = 0,
    HCCL_RECV = 1,
    HCCL_SEND_RECV_RESERVED
} HcclSendRecvType;

typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;
    void *buf;
    uint64_t count;
    HcclDataType dataType;
    uint32_t remoteRank;
} HcclSendRecvItem;

typedef enum {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL,
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC,
    HCCL_CMD_INIT,
    HCCL_CMD_BARRIER,
    HCCL_CMD_MAX
} HcclCMDType;
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_TYPES_H_
