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

#ifndef MSPTI_ACTIVITY_H
#define MSPTI_ACTIVITY_H

#define ACTIVITY_STRUCT_ALIGNMENT 8
#if defined(_WIN32)
#define START_PACKED_ALIGNMENT __pragma(pack(push, 1))
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_STRUCT_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__)
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__((__packed__)) __attribute__((aligned(ACTIVITY_STRUCT_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "mspti_result.h"

#define MSPTI_INVALID_DEVICE_ID ((uint32_t) 0xFFFFFFFFU)
#define MSPTI_INVALID_STREAM_ID ((uint32_t) 0xFFFFFFFFU)
#define MSPTI_INVALID_CORRELATION_ID ((uint64_t) 0)

#if defined(__cplusplus)
extern "C" {
#endif
#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility push(default)
#endif

/**
 * @brief The kinds of activity records.
 *
 * Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 */
typedef enum {
    /**
    * The activity record is invalid.
    */
    MSPTI_ACTIVITY_KIND_INVALID = 0,
    MSPTI_ACTIVITY_KIND_MARKER = 1,
    MSPTI_ACTIVITY_KIND_KERNEL = 2,
    MSPTI_ACTIVITY_KIND_API = 3,
    MSPTI_ACTIVITY_KIND_HCCL = 4,
    MSPTI_ACTIVITY_KIND_MEMORY = 5,
    MSPTI_ACTIVITY_KIND_MEMSET = 6,
    MSPTI_ACTIVITY_KIND_MEMCPY = 7,
    MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 8,
    MSPTI_ACTIVITY_KIND_COMMUNICATION = 9,
    MSPTI_ACTIVITY_KIND_ACL_API = 10,
    MSPTI_ACTIVITY_KIND_NODE_API = 11,
    MSPTI_ACTIVITY_KIND_RUNTIME_API = 12,
    MSPTI_ACTIVITY_KIND_COUNT,
    MSPTI_ACTIVITY_KIND_FORCE_INT = 0x7fffffff
} msptiActivityKind;

/**
 * @brief The kind of external APIs supported for correlation.
 *
 * Custom correlation kinds are reserved for usage in external tools.
 */
typedef enum {
    MSPTI_EXTERNAL_CORRELATION_KIND_INVALID = 0,
    MSPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 1,
    MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 2,
    MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1 = 3,
    MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2 = 4,
    MSPTI_EXTERNAL_CORRELATION_KIND_SIZE,
    MSPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT = 0x7fffffff,
} msptiExternalCorrelationKind;

typedef enum {
    MSPTI_ACTIVITY_COMMUNICATION_INT8 = 0,
    MSPTI_ACTIVITY_COMMUNICATION_INT16 = 1,
    MSPTI_ACTIVITY_COMMUNICATION_INT32 = 2,
    MSPTI_ACTIVITY_COMMUNICATION_FP16 = 3,
    MSPTI_ACTIVITY_COMMUNICATION_FP32 = 4,
    MSPTI_ACTIVITY_COMMUNICATION_INT64 = 5,
    MSPTI_ACTIVITY_COMMUNICATION_UINT64 = 6,
    MSPTI_ACTIVITY_COMMUNICATION_UINT8 = 7,
    MSPTI_ACTIVITY_COMMUNICATION_UINT16 = 8,
    MSPTI_ACTIVITY_COMMUNICATION_UINT32 = 9,
    MSPTI_ACTIVITY_COMMUNICATION_FP64 = 10,
    MSPTI_ACTIVITY_COMMUNICATION_BFP16 = 11,
    MSPTI_ACTIVITY_COMMUNICATION_INT128 = 12,
    MSPTI_ACTIVITY_COMMUNICATION_HIF8 = 14,
    MSPTI_ACTIVITY_COMMUNICATION_FP8E4M3 = 15,
    MSPTI_ACTIVITY_COMMUNICATION_FP8E5M2 = 16,
    MSPTI_ACTIVITY_COMMUNICATION_FP8E8M0 = 17,
    MSPTI_ACTIVITY_COMMUNICATION_INVALID_TYPE = 0x0000FFFF
} msptiCommunicationDataType;

START_PACKED_ALIGNMENT

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
} msptiActivity;

typedef struct PACKED_ALIGNMENT {
    /**
    * The activity record kind, must be MSPTI_ACTIVITY_KIND_KERNEL.
    */
    msptiActivityKind kind;

    /**
    * The start timestamp for the kernel, in ns.
    */
    uint64_t start;

    /**
    * The end timestamp for the kernel, in ns.
    */
    uint64_t end;

    /**
    * A stream object requires that we identify device and stream ID.
    */
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;

    /**
    * The correlation ID of the kernel.
    */
    uint64_t correlationId;

    /**
    * The kernel type.
    */
    const char *type;

    /**
    * The kernel name.
    */
    const char *name;
} msptiActivityKernel;

typedef struct PACKED_ALIGNMENT {
    /**
    * The activity record kind, must be MSPTI_ACTIVITY_COMMUNICATION_KIND.
    */
    msptiActivityKind kind;

    /**
    * Communication data type, Works with
    * msptiCommunicationDataType
    */
    msptiCommunicationDataType dataType;

    /**
    * Count of communication data
    */
    uint64_t count;

    /**
    * A stream object requires that we identify device and stream ID.
    */
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;

    /**
    * The start timestamp on device for the communication, in ns.
    */
    uint64_t start;

    /**
    * The end timestamp on device for the communication, in ns.
    */
    uint64_t end;

    /**
    * The type of algorithm used for communication
    */
    const char* algType;

    /**
    * The communication op name.
    */
    const char* name;

    /**
    * The name of communication group where the communication operator is executed
    */
    const char* commName;

    /**
    * The correlation ID of the communication op.
    */
    uint64_t correlationId;
} msptiActivityCommunication;

typedef struct PACKED_ALIGNMENT {
    /**
    * The activity record kind, must be MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION.
    */
    msptiActivityKind kind;
    /**
    * The kind of external API this record correlated to.
    */
    msptiExternalCorrelationKind externalKind;
    /**
    * The correlation ID of the associated non-MSPTI API record.
    */
    uint64_t externalId;
    /**
    * The correlation ID of the associated MSPTI driver or runtime API record.
    */
    uint64_t correlationId;
} msptiActivityExternalCorrelation;

END_PACKED_ALIGNMENT

/**
 * @brief Function type for callback used by MSPTI to request an empty
 * buffer for storing activity records.
 *
 * This callback function signals the MSPTI client that an activity
 * buffer is needed by MSPTI. The activity buffer is used by MSPTI to
 * store activity records. The callback function can decline the
 * request by setting **buffer to NULL. In this case MSPTI may drop
 * activity records.
 *
 * @param buffer Returns the new buffer. If set to NULL then no buffer
 * is returned.
 * @param size Returns the size of the returned buffer.
 * @param maxNumRecords Returns the maximum number of records that
 * should be placed in the buffer. If 0 then the buffer is filled with
 * as many records as possible. If > 0 the buffer is filled with at
 * most that many records before it is returned.
 */
typedef void(*msptiBuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

/**
 * @brief Function type for callback used by MSPTI to return a buffer
 * of activity records.
 *
 * This callback function returns to the MSPTI client a buffer
 * containing activity records.  The buffer contains @p validSize
 * bytes of activity records which should be read using
 * msptiActivityGetNextRecord. After this call MSPTI
 * relinquished ownership of the buffer and will not use it
 * anymore. The client may return the buffer to MSPTI using the
 * msptiBuffersCallbackRequestFunc callback.
 *
 * @param buffer The activity record buffer.
 * @param size The total size of the buffer in bytes as set in
 * MSPTI_BuffersCallbackRequestFunc.
 * @param validSize The number of valid bytes in the buffer.
 */
typedef void(*msptiBuffersCallbackCompleteFunc)(uint8_t *buffer, size_t size, size_t validSize);

/**
 * @brief Registers callback functions with MSPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested/completed callbacks from MSPTI.
 *
 * @param funcBufferRequested [in] callback which is invoked when an empty
 * buffer is requested by MSPTI
 * @param funcBufferCompleted [in] callback which is invoked when a buffer
 * containing activity records is available from MSPTI
 *
 * @return MSPTI_SUCCESS
 * @return MSPTI_ERROR_INVALID_PARAMETER if either
 * funcBufferRequested or funcBufferCompleted is NULL
 */
msptiResult msptiActivityRegisterCallbacks(
    msptiBuffersCallbackRequestFunc funcBufferRequested, msptiBuffersCallbackCompleteFunc funcBufferCompleted);

/**
 * @brief Enable collection of a specific kind of activity record.
 *
 * Enable collection of a specific kind of activity record. Multiple
 * kinds can be enabled by calling this function multiple times.
 * By default, the collection of all activity types is inactive.
 *
 * @param kind [in] The kind of activity record to collect
 *
 * @return MSPTI_SUCCESS
 */
msptiResult msptiActivityEnable(msptiActivityKind kind);

/**
 * @brief Disable collection of a specific kind of activity record.
 *
 * Disable collection of a specific kind of activity record. Multiple
 * kinds can be disabled by calling this function multiple times.
 * By default, the collection of all activity types is inactive.
 *
 * @param kind [in] The kind of activity record to stop collecting
 *
 * @return MSPTI_SUCCESS
 */
msptiResult msptiActivityDisable(msptiActivityKind kind);

/**
 * @brief Iterate over the activity records in a buffer.
 *
 * This is a function to iterate over the activity records in buffer.
 *
 * @param buffer [in] The buffer containing activity records
 * @param validBufferSizeBytes [in] The number of valid bytes in the buffer.
 * @param record [in] Inputs the previous record returned by
 * msptiActivityGetNextRecord and returns the next activity record
 * from the buffer. If input value is NULL, returns the first activity
 * record in the buffer.
 *
 * @return MSPTI_SUCCESS
 * @return MSPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * @return MSPTI_ERROR_INVALID_PARAMETER if buffer is NULL.
 */
msptiResult msptiActivityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes, msptiActivity **record);

/**
 * @brief Request to deliver activity records via the buffer completion callback.
 *
 * This function returns the activity records associated with all contexts/streams
 * (and the global buffers not associated with any stream) to the MSPTI client
 * using the callback registered in msptiActivityRegisterCallbacks. It return all
 * activity buffers that contain completed activity records, even if these buffers are not completely filled.
 *
 * Before calling this function, the buffer handling callback api must be activated
 * by calling msptiActivityRegisterCallbacks.
 *
 * @param flag [in] Reserved for internal use.
 *
 * @return MSPTI_SUCCESS
 */
msptiResult msptiActivityFlushAll(uint32_t flag);

/**
 * @brief Push an external correlation id for the calling thread.
 *
 * This function notifies MSPTI that the calling thread is entering an external API region.
 * When a MSPTI activity API record is created while within an external API region and
 * MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION is enabled, the activity API record will
 * be preceded by a msptiActivityExternalCorrelation record for each msptiExternalCorrelationKind.
 *
 * @param kind [in] The kind of external API activities should be correlated with.
 * @param id [in] External correlation id.
 *
 * @return MSPTI_SUCCESS
 * @return MSPTI_ERROR_INVALID_PARAMETER
 */
msptiResult msptiActivityPushExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t id);

/**
 * @brief Pop an external correlation id for the calling thread.
 *
 * This function notifies MSPTI that the calling thread is leaving an external API region.
 *
 * @param kind [in] The kind of external API activities should be correlated with.
 * @param lastId [in] If the function returns successful, contains the last external correlation id for this kind,
 * can be NULL.
 *
 * @return MSPTI_SUCCESS
 * @return MSPTI_ERROR_INVALID_PARAMETER
 * @return MSPTI_ERROR_QUEUE_EMPTY
 */
msptiResult msptiActivityPopExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t *lastId);

/** Query whether collection of the specified activity kind is enabled. */
bool msptiActivityIsEnabled(msptiActivityKind kind);

#if defined(__GNUC__) && defined(MSPTI_LIB)
#pragma GCC visibility pop
#endif
#if defined(__cplusplus)
}
#endif

#endif
