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

#ifndef MSPTI_CALLBACK_H
#define MSPTI_CALLBACK_H

#include <stdint.h>
#include "mspti_result.h"

#if defined(__cplusplus)
extern "C" {
#endif
#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility push(default)
#endif

/**
 * @brief A callback subscriber.
 */
typedef struct msptiSubscriber_st *msptiSubscriberHandle;

typedef uint32_t msptiCallbackId;

/**
* @brief Callback domains.
*
* Callback domains. Each domain represents callback points for a
* group of related API functions or CANN driver activity.
*/
typedef enum {
    /**
   * Invalid domain.
   */
    MSPTI_CB_DOMAIN_INVALID           = 0,
    /**
   * Domain containing callback points for all runtime API functions.
   */
    MSPTI_CB_DOMAIN_RUNTIME           = 1,
    MSPTI_CB_DOMAIN_HCCL              = 2,
    MSPTI_CB_DOMAIN_SIZE,
    MSPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
} msptiCallbackDomain;

/**
 * @brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in @p cbdata depends on the @p domain. If @p domain is
 * MSPTI_CB_DOMAIN_RUNTIME the type
 * of @p cbdata will be msptiCallbackData.
 *
 * @param userdata User data supplied at subscription of the callback
 * @param domain The domain of the callback
 * @param cbid The ID of the callback
 * @param cbdata Data passed to the callback.
 */
/* cbdata is const msptiCallbackData* in the SDK. torch_npu never
 * registers a callback, so the payload type is never dereferenced and
 * is declared as a void pointer here to avoid carrying the definition.
 */
typedef void (*msptiCallbackFunc)(
    void *userdata,
    msptiCallbackDomain domain,
    msptiCallbackId cbid,
    const void *cbdata);

/**
 * @brief Initialize a callback subscriber with a callback function
 * and user data.
 *
 * Initializes a callback subscriber with a callback function and
 * (optionally) a pointer to user data. The returned subscriber handle
 * can be used to enable and disable the callback for specific domains
 * and callback IDs.
 * @note Only a single subscriber can be registered at a time. To ensure
 * that no other MSPTI client interrupts the profiling session, it's the
 * responsibility of all the MSPTI clients to call this function before
 * starting the profiling session.
 * @note This function does not enable any callbacks.
 * @note @b Thread-safety: this function is thread safe.
 *
 * @param subscriber [in] handle to initialize subscriber
 * @param callback [in] The callback function
 * @param userdata [in] A pointer to user data. This data will be passed to
 * the callback function via the @p userdata parameter.
 *
 * @return MSPTI_SUCCESS on success
 * @return MSPTI_ERROR_INNER if unable to initialize MSPTI
 * @return MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if there is already a MSPTI subscriber
 * @return MSPTI_ERROR_INVALID_PARAMETER if @p subscriber is NULL
 */
msptiResult msptiSubscribe(msptiSubscriberHandle *subscriber, msptiCallbackFunc callback, void* userdata);

/**
 * @brief Unregister a callback subscriber.
 *
 * Removes a callback subscriber so that no future callbacks will be
 * issued to that subscriber.
 *
 * @param subscriber [in] Handle to the initialize subscriber
 *
 * @return MSPTI_SUCCESS on success
 * @return MSPTI_ERROR_INVALID_PARAMETER if @p subscriber is NULL or not initialized
 */
msptiResult msptiUnsubscribe(msptiSubscriberHandle subscriber);

#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility pop
#endif
#if defined(__cplusplus)
}
#endif

#endif
