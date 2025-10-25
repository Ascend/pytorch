/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include "shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup group_macros Macros
 * @{
*/

/// \def SHMEM_XXX_VERSION
/// \brief macros that define current version info
#define SHMEM_MAX_IP_PORT_LEN 64
/**@} */  // end of group_macros

/**
 * @defgroup group_enums Enumerations
 * @{
*/

/**
 * @brief Error code for the SHMEM library.
*/
enum shmem_error_code_t : int {
    SHMEM_SUCCESS = 0,         ///< Task execution was successful.
    SHMEM_INVALID_PARAM = -1,  ///< There is a problem with the parameters.
    SHMEM_INVALID_VALUE = -2,  ///< There is a problem with the range of the value of the parameter.
    SHMEM_SMEM_ERROR = -3,     ///< There is a problem with SMEM.
    SHMEM_INNER_ERROR = -4,    ///< This is a problem caused by an internal error.
    SHMEM_NOT_INITED = -5,     ///< This is a problem caused by an uninitialization.
};

/**
 * @brief The state of the SHMEM library initialization.
*/
enum shmem_init_status_t {
    SHMEM_STATUS_NOT_INITIALIZED = 0,  ///< Uninitialized.
    SHMEM_STATUS_SHM_CREATED,          ///< Shared memory heap creation is complete.
    SHMEM_STATUS_IS_INITIALIZED,       ///< Initialization is complete.
    SHMEM_STATUS_INVALID = INT_MAX,    ///< Invalid status code.
};

/**@} */  // end of group_enums

/**
 * @defgroup group_structs Structs
 * @{
*/

constexpr uint16_t SHMEM_UNIQUE_ID_INNER_LEN = 60;

typedef struct {
    int32_t version;
    char internal[SHMEM_UNIQUE_ID_INNER_LEN];
} shmem_uniqueid_t;

/**
 * @struct shmem_init_optional_attr_t
 * @brief Optional parameter for the attributes used for initialization.
 *
 * - int version: version
 * - data_op_engine_type_t data_op_engine_type: data_op_engine_type
 * - uint32_t shm_init_timeout: shm_init_timeout
 * - uint32_t shm_create_timeout: shm_create_timeout
 * - uint32_t control_operation_timeout: control_operation_timeout
*/
typedef struct {
    int version;
    data_op_engine_type_t data_op_engine_type;
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
} shmem_init_optional_attr_t;

/**
 * @struct shmem_init_attr_t
 * @brief Mandatory parameter for attributes used for initialization.
 *
 * - int my_rank: The rank of the current process.
 * - int n_ranks: The total rank number of all processes.
 * - char ip_port[SHMEM_MAX_IP_PORT_LEN]: The ip and port of the communication server. The port must not conflict
 *   with other modules and processes.
 * - uint64_t local_mem_size: The size of shared memory currently occupied by current rank.
 * - shmem_init_optional_attr_t option_attr: Optional Parameters.
*/
typedef struct {
    int my_rank;
    int n_ranks;
    char ip_port[SHMEM_MAX_IP_PORT_LEN];
    uint64_t local_mem_size;
    shmem_init_optional_attr_t option_attr;
} shmem_init_attr_t;

#ifdef __cplusplus
}
#endif

#endif