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
#include "third_party/shmem/include/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup group_macros Macros
 * @{
*/
/// \def SHMEM_HOST_API
/// \brief A macro that identifies a function on the host side.
#define SHMEM_HOST_API   __attribute__((visibility("default")))
/**@} */ // end of group_macros

/**
 * @defgroup group_enums Enumerations
 * @{
*/

/**
 * @brief Error code for the SHMEM library.
*/
enum shmem_error_code_t : int {
    SHMEM_SUCCESS = 0,          ///< Task execution was successful.
    SHMEM_INVALID_PARAM = -1,   ///< There is a problem with the parameters.
    SHMEM_INVALID_VALUE = -2,   ///< There is a problem with the range of the value of the parameter.
    SHMEM_SMEM_ERROR = -3,      ///< There is a problem with SMEM.
    SHMEM_INNER_ERROR = -4,     ///< This is a problem caused by an internal error.
    SHMEM_NOT_INITED = -5,      ///< This is a problem caused by an uninitialization.
};

/**
 * @brief init flags
*/
enum aclshmemx_bootstrap_t : int {
    ACLSHMEMX_INIT_WITH_DEFAULT = 1 << 0,
    ACLSHMEMX_INIT_WITH_MPI = 1 << 1,
    ACLSHMEMX_INIT_WITH_UNIQUEID = 1 << 3,
    ACLSHMEMX_INIT_MAX = 1 << 31
};

/**
 * @brief The state of the SHMEM library initialization.
*/
enum shmem_init_status_t{
    SHMEM_STATUS_NOT_INITIALIZED = 0,    ///< Uninitialized.
    SHMEM_STATUS_SHM_CREATED,           ///< Shared memory heap creation is complete.
    SHMEM_STATUS_IS_INITIALIZED,         ///< Initialization is complete.
    SHMEM_STATUS_INVALID = INT_MAX,     ///< Invalid status code.
};

constexpr uint16_t SHMEM_UNIQUE_ID_INNER_LEN = 60;
/// \brief Inner length of the unique ID buffer for ACLSHMEM
constexpr uint16_t ACLSHMEM_UNIQUE_ID_INNER_LEN = 124;
/// \brief Default timeout value (in seconds) for ACLSHMEM operations
constexpr int DEFAULT_TIMEOUT = 120;

/// \def ACLSHMEM_MAX_IP_PORT_LEN
/// \brief Maximum length of the IP and port string in ACLSHMEM (including null terminator)
#define ACLSHMEM_MAX_IP_PORT_LEN 64

typedef struct {
    int32_t version;
    char internal[SHMEM_UNIQUE_ID_INNER_LEN];
} shmem_uniqueid_t;

constexpr int32_t SHMEM_UNIQUEID_VERSION = (1 << 16) + sizeof(shmem_uniqueid_t);

#define SHMEM_UNIQUEID_INITIALIZER                      \
    {                                                   \
        SHMEM_UNIQUEID_VERSION,                         \
        {                                               \
            0                                           \
        }                                               \
    }                                                   \

/**
 * @struct aclshmemx_uniqueid_t
 * @brief Structure required for SHMEM unique ID (uid) initialization
 *
 * - int32_t version: version.
 * - int my_pe: The pe of the current process.
 * - int n_pes: The total pe number of all processes.
 * - char internal[ACLSHMEM_UNIQUE_ID_INNER_LEN]: Internal information of uid.
*/
typedef struct {
    int32_t version;
    int my_pe;
    int n_pes;
    char internal[ACLSHMEM_UNIQUE_ID_INNER_LEN];
} aclshmemx_uniqueid_t;

/// \brief Version number of the ACLSHMEM unique ID structure
constexpr int32_t ACLSHMEM_UNIQUEID_VERSION = (1 << 16) + sizeof(aclshmemx_uniqueid_t);
/**@} */ // end of group_constants

/**
 * @addtogroup group_macros
 * @{
*/
/// \def ACLSHMEM_UNIQUEID_INITIALIZER
/// \brief Initializer macro for the ACLSHMEM unique ID structure
#define ACLSHMEM_UNIQUEID_INITIALIZER                   \
    {                                                   \
        ACLSHMEM_UNIQUEID_VERSION,                      \
        {                                               \
            0                                           \
        }                                               \
    }

/**@} */ // end of group_enums

/**
 * @defgroup group_structs Structs
 * @{
*/

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
 * - const char* ip_port: The ip and port of the communication server. The port must not conflict with other modules and processes.
 * - uint64_t local_mem_size: The size of shared memory currently occupied by current rank.
 * - shmem_init_optional_attr_t option_attr: Optional Parameters.
*/
typedef struct {
    int my_rank;
    int n_ranks;
    const char* ip_port;
    uint64_t local_mem_size;
    shmem_init_optional_attr_t option_attr;
} shmem_init_attr_t;

/**
 * @struct aclshmem_init_optional_attr_t
 * @brief Optional parameter for the attributes used for initialization.
 *
 * - int version: version
 * - data_op_engine_type_t data_op_engine_type: data_op_engine_type
 * - uint32_t shm_init_timeout: shm_init_timeout
 * - uint32_t shm_create_timeout: shm_create_timeout
 * - uint32_t control_operation_timeout: control_operation_timeout
 * - int32_t sockFd: sock_fd for apply port in advance
*/
typedef struct {
    int version;
    data_op_engine_type_t data_op_engine_type;
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
    int32_t sockFd;
} aclshmem_init_optional_attr_t;
/**
 * @struct aclshmemx_init_attr_t
 * @brief Mandatory parameter for attributes used for initialization.
 *
 * - int my_pe: The pe of the current process.
 * - int n_pes: The total pe number of all processes.
 * - char ip_port[ACLSHMEM_MAX_IP_PORT_LEN]: The ip and port of the communication server. The port must not conflict
 *   with other modules and processes.
 * - uint64_t local_mem_size: The size of shared memory currently occupied by current pe.
 * - aclshmem_init_optional_attr_t option_attr: Optional Parameters.
*/
typedef struct {
    int my_pe;
    int n_pes;
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN];
    uint64_t local_mem_size;
    aclshmem_init_optional_attr_t option_attr = {(1 << 16) + sizeof(aclshmem_init_optional_attr_t), ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    void *comm_args;
} aclshmemx_init_attr_t;

/**
 * @brief Callback function of private key password decryptor, see shmem_register_decrypt_handler
 *
 * @param cipherText       [in] the encrypted text(private password)
 * @param cipherTextLen    [in] the length of encrypted text
 * @param plainText        [out] the decrypted text(private password)
 * @param plaintextLen     [out] the length of plainText
 */
typedef int (*shmem_decrypt_handler)(const char *cipherText, size_t cipherTextLen, char *plainText, size_t &plainTextLen);

/**@} */ // end of group_structs

#ifdef __cplusplus
}
#endif

#endif