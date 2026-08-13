/* -------------------------------------------------------------------------
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is part of the MindStudio project.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *    http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
*/

#ifndef MSPTI_RESULT_H
#define MSPTI_RESULT_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif
#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility push(default)
#endif

/**
 * @brief MSPTI result codes.
 *
 * Error and result codes returned by MSPTI functions.
 */
typedef enum {
    MSPTI_SUCCESS                                       = 0,
    MSPTI_ERROR_INVALID_PARAMETER                       = 1,
    MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED      = 2,
    MSPTI_ERROR_MAX_LIMIT_REACHED                       = 3,
    MSPTI_ERROR_DEVICE_OFFLINE                          = 4,
    MSPTI_ERROR_QUEUE_EMPTY                             = 5,
    MSPTI_ERROR_WITHOUT_LD_PRELOAD                      = 6,
    MSPTI_ERROR_INNER                                   = 999,
    MSPTI_ERROR_FORCE_INT                               = 0x7fffffff
} msptiResult;

#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility pop
#endif
#if defined(__cplusplus)
}
#endif

#endif
