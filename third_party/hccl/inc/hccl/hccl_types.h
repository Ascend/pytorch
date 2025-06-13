/**
 * @file hccl_types.h
 * @brief HCCL data type definition 
 * 
 */
 
#ifndef HCCL_TYPES_H_
#define HCCL_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
const uint32_t HCCL_COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;
const uint32_t HCCL_COMM_CONFIG_VERSION = 5;
const uint32_t HCCL_COMM_DEFAULT_BUFFSIZE = 200;                // 200MB buffer size
const uint32_t HCCL_COMM_DEFAULT_DETERMINISTIC = 0;             // Disable deterministic calculations
const uint32_t COMM_NAME_MAX_LENGTH = 128;
const uint32_t UDI_MAX_LENGTH = 128;
const uint32_t HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xffffffff;
const uint32_t HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0;

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
    HCCL_REDUCE_RESERVED    /**< reserved */
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
    HCCL_DATA_TYPE_INT64 = 5,   /**< int64 */
    HCCL_DATA_TYPE_UINT64 = 6,  /**< uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,   /**< uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,  /**< uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,  /**< uint32 */
    HCCL_DATA_TYPE_FP64 = 10,   /**< fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,  /**< bfp16 */
    HCCL_DATA_TYPE_RESERVED     /**< reserved */
} HcclDataType;

const uint32_t HCCL_ROOT_INFO_BYTES =  4108; // 4108: root info length

/**
 * @brief HCCL root info
 */
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

/**
 * @brief HCCL set config type
 */
typedef enum {
    HCCL_DETERMINISTIC = 0,
    HCCL_CONFIG_RESERVED
} HcclConfig;

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

union  HcclConfigValue {
    int32_t value;
};

typedef struct HcclCommConfigDef {
    char reserved[HCCL_COMM_CONFIG_INFO_BYTES];
    uint32_t hcclBufferSize;
    uint32_t hcclDeterministic;
    char hcclCommName[COMM_NAME_MAX_LENGTH];
    char hcclUdi[UDI_MAX_LENGTH];
    uint32_t hcclOpExpansionMode;
    uint32_t hcclRdmaTrafficClass;
    uint32_t hcclRdmaServiceLevel;
} HcclCommConfig;

typedef enum {
    HCCL_COMM_CONFIG_BUFFER_SIZE = 0,
    HCCL_COMM_CONFIG_DETERMINISTIC = 1,
    HCCL_COMM_CONFIG_COMM_NAME = 2,
    HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3,
    HCCL_COMM_CONFIG_RESERVED,
} HcclCommConfigCapability;
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_TYPES_H_
