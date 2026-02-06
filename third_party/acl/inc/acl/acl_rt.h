/**
* @file acl_rt.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>
#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Current version is 1.15.0
#define ACL_MAJOR_VERSION              1
#define ACL_MINOR_VERSION              15
#define ACL_PATCH_VERSION              0
#define ACL_EVENT_SYNC                    0x00000001U
#define ACL_EVENT_CAPTURE_STREAM_PROGRESS 0x00000002U
#define ACL_EVENT_TIME_LINE               0x00000008U
#define ACL_EVENT_EXTERNAL                0x00000020U
#define ACL_EVENT_IPC                     0x00000040U

// for create stream
#define ACL_STREAM_FAST_LAUNCH  0x00000001U
#define ACL_STREAM_FAST_SYNC    0x00000002U
#define ACL_STREAM_PERSISTENT   0x00000004U
#define ACL_STREAM_HUGE         0x00000008U
#define ACL_STREAM_CPU_SCHEDULE 0x00000010U

#define ACL_STREAM_WAIT_VALUE_GEQ 0x00000000U
#define ACL_STREAM_WAIT_VALUE_EQ  0x00000001U
#define ACL_STREAM_WAIT_VALUE_AND 0x00000002U
#define ACL_STREAM_WAIT_VALUE_NOR 0x00000003U

#define ACL_CONTINUE_ON_FAILURE 0x00000000U
#define ACL_STOP_ON_FAILURE     0x00000001U

// for device get capability
#define ACL_DEV_FEATURE_SUPPORT     0x00000001
#define ACL_DEV_FEATURE_NOT_SUPPORT 0x00000000

#define ACL_RT_NOTIFY_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_NOTIFY_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x02UL

#define MAX_MODULE_NUM 128
#define ACL_RT_NOTIFY_IMPORT_FLAG_DEFAULT            0x0UL
#define ACL_RT_NOTIFY_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x02UL

#define ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x1UL

#define ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT            0x0UL
#define ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x1UL

#define ACL_RT_VMM_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x1UL

#define ACL_HOST_REG_MAPPED 0x2UL
#define ACL_HOST_REG_PINNED 0X10000000UL

#define ACL_VALUE_WAIT_EQ                0x1

#define ACL_IPC_EVENT_HANDLE_SIZE        64U

constexpr int32_t DEVICE_UTILIZATION_NOT_SUPPORT = -1;

typedef enum aclrtRunMode {
    ACL_DEVICE,
    ACL_HOST,
} aclrtRunMode;

typedef enum aclrtTsId {
    ACL_TS_ID_AICORE   = 0,
    ACL_TS_ID_AIVECTOR = 1,
    ACL_TS_ID_RESERVED = 2,
} aclrtTsId;

typedef enum aclrtEventStatus {
    ACL_EVENT_STATUS_COMPLETE  = 0,
    ACL_EVENT_STATUS_NOT_READY = 1,
    ACL_EVENT_STATUS_RESERVED  = 2,
} aclrtEventStatus;

typedef enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE = 1,
} aclrtEventRecordedStatus;

typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xFFFF,
} aclrtEventWaitStatus;

typedef enum aclrtStreamStatus {
    ACL_STREAM_STATUS_COMPLETE  = 0,
    ACL_STREAM_STATUS_NOT_READY = 1,
    ACL_STREAM_STATUS_RESERVED  = 0xFFFF,
} aclrtStreamStatus;

typedef enum aclrtCallbackBlockType {
    ACL_CALLBACK_NO_BLOCK,
    ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
    ACL_MEM_MALLOC_HUGE1G_ONLY,
    ACL_MEM_MALLOC_HUGE1G_ONLY_P2P,
    ACL_MEM_TYPE_LOW_BAND_WIDTH   = 0x0100,
    ACL_MEM_TYPE_HIGH_BAND_WIDTH  = 0x1000,
    ACL_MEM_ACCESS_USER_SPACE_READONLY = 0x100000,
} aclrtMemMallocPolicy;

typedef enum {
    ACL_HOST_REGISTER_MAPPED = 0,
} aclrtHostRegisterType;

typedef enum {
    ACL_RT_MEM_ATTR_RSV = 0,
    ACL_RT_MEM_ATTR_MODULE_ID,
    ACL_RT_MEM_ATTR_DEVICE_ID,
    ACL_RT_MEM_ATTR_VA_FLAG,
} aclrtMallocAttrType;

typedef union {
    uint16_t moduleId;
    uint32_t deviceId;
    uint32_t vaFlag;
    uint8_t rsv[8];
} aclrtMallocAttrValue;

typedef struct {
    aclrtMallocAttrType attr;
    aclrtMallocAttrValue value;
} aclrtMallocAttribute;

typedef struct {
    aclrtMallocAttribute* attrs;
    size_t numAttrs;
} aclrtMallocConfig;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
    ACL_HBM_MEM_HUGE1G,
    ACL_HBM_MEM_P2P_HUGE1G,
} aclrtMemAttr;

typedef enum aclrtGroupAttr {
    ACL_GROUP_AICORE_INT,
    ACL_GROUP_AIV_INT,
    ACL_GROUP_AIC_INT,
    ACL_GROUP_SDMANUM_INT,
    ACL_GROUP_ASQNUM_INT,
    ACL_GROUP_GROUPID_INT
} aclrtGroupAttr;

typedef enum aclrtFloatOverflowMode {
    ACL_RT_OVERFLOW_MODE_SATURATION = 0,
    ACL_RT_OVERFLOW_MODE_INFNAN,
    ACL_RT_OVERFLOW_MODE_UNDEF,
} aclrtFloatOverflowMode;

typedef enum {
    ACL_RT_STREAM_WORK_ADDR_PTR = 0, /**< pointer to model work addr */
    ACL_RT_STREAM_WORK_SIZE, /**< pointer to model work size */
    ACL_RT_STREAM_FLAG,
    ACL_RT_STREAM_PRIORITY,
} aclrtStreamConfigAttr;

typedef struct aclrtStreamConfigHandle {
    void* workptr;
    size_t workSize;
    size_t flag;
    uint32_t priority;
} aclrtStreamConfigHandle;

typedef struct aclrtUtilizationExtendInfo aclrtUtilizationExtendInfo;

typedef struct aclrtUtilizationInfo {
    int32_t cubeUtilization;
    int32_t vectorUtilization;
    int32_t aicpuUtilization;
    int32_t memoryUtilization;
    aclrtUtilizationExtendInfo *utilizationExtend; /**< reserved parameters, current version needs to be null */
} aclrtUtilizationInfo;

typedef struct tagRtGroupInfo aclrtGroupInfo;

typedef struct rtExceptionInfo aclrtExceptionInfo;

typedef enum aclrtMemLocationType {
    ACL_MEM_LOCATION_TYPE_HOST = 0, /**< reserved enum, current version not support */
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEM_LOCATION_TYPE_UNREGISTERED,
} aclrtMemLocationType;

typedef struct aclrtMemLocation {
    uint32_t id;
    aclrtMemLocationType type;
} aclrtMemLocation;

typedef struct aclrtPtrAttributes {
    aclrtMemLocation location;
    uint32_t pageSize;
    uint32_t rsv[4];
} aclrtPtrAttributes;

typedef struct aclrtMemUsageInfo {
    char name[32];
    uint64_t curMemSize;
    uint64_t memPeakSize;
    size_t reserved[8];
} aclrtMemUsageInfo;

typedef enum aclrtMemAllocationType {
    ACL_MEM_ALLOCATION_TYPE_PINNED = 0,
} aclrtMemAllocationType;

typedef enum aclrtMemHandleType {
    ACL_MEM_HANDLE_TYPE_NONE = 0,
} aclrtMemHandleType;

typedef struct aclrtPhysicalMemProp {
    aclrtMemHandleType handleType;
    aclrtMemAllocationType allocationType;
    aclrtMemAttr memAttr;
    aclrtMemLocation location;
    uint64_t reserve;
} aclrtPhysicalMemProp;

typedef enum aclrtMemGranularityOptions {
    ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
    ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    ACL_RT_MEM_ALLOC_GRANULARITY_UNDEF = 0xFFFF,
} aclrtMemGranularityOptions;

typedef void* aclrtDrvMemHandle;

typedef void (*aclrtCallback)(void *userData);

typedef void (*aclrtHostFunc)(void *args);

typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

typedef enum aclrtDeviceStatus {
    ACL_RT_DEVICE_STATUS_NORMAL = 0,
    ACL_RT_DEVICE_STATUS_ABNORMAL,
    ACL_RT_DEVICE_STATUS_END = 0xFFFF,
} aclrtDeviceStatus;

typedef struct aclrtUuid {
    char bytes[16];
} aclrtUuid;

typedef void* aclrtBinary;
typedef void* aclrtBinHandle;
typedef void* aclrtFuncHandle;
typedef void* aclrtArgsHandle;
typedef void* aclrtParamHandle;

#define MAX_MEM_UCE_INFO_ARRAY_SIZE 128
#define UCE_INFO_RESERVED_SIZE 14

typedef struct aclrtMemUceInfo {
    void* addr;
    size_t len;
    size_t reserved[UCE_INFO_RESERVED_SIZE];
} aclrtMemUceInfo;

typedef enum aclrtCmoType {
    ACL_RT_CMO_TYPE_PREFETCH = 0,
    ACL_RT_CMO_TYPE_WRITEBACK,
    ACL_RT_CMO_TYPE_INVALID,
    ACL_RT_CMO_TYPE_FLUSH,
} aclrtCmoType;

typedef enum aclrtLastErrLevel {
    ACL_RT_THREAD_LEVEL = 0,
} aclrtLastErrLevel;

#define ACL_RT_BINARY_MAGIC_ELF_AICORE      0x43554245U
#define ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE 0x41415246U
#define ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE   0x41494343U

typedef enum aclrtBinaryLoadOptionType {
    ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD = 1,
    ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE = 3,
} aclrtBinaryLoadOptionType;

typedef union aclrtBinaryLoadOptionValue {
    uint32_t isLazyLoad;
    uint32_t magic;
    int32_t cpuKernelMode;
    uint32_t rsv[4];
} aclrtBinaryLoadOptionValue;

typedef struct {
    aclrtBinaryLoadOptionType type;
    aclrtBinaryLoadOptionValue value;
} aclrtBinaryLoadOption;

typedef struct aclrtBinaryLoadOptions {
    aclrtBinaryLoadOption *options;
    size_t numOpt;
} aclrtBinaryLoadOptions;

typedef enum {
    ACL_RT_ENGINE_TYPE_AIC = 0,
    ACL_RT_ENGINE_TYPE_AIV,
} aclrtEngineType;

typedef enum aclrtLaunchKernelAttrId {
    ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1,
    ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE = 3,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH,
    ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP,
    ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT,
} aclrtLaunchKernelAttrId;

typedef union aclrtLaunchKernelAttrValue {
    uint8_t schemMode;
    uint32_t localMemorySize;
    aclrtEngineType engineType;
    uint32_t blockDimOffset;
    uint8_t isBlockTaskPrefetch;
    uint8_t isDataDump;
    uint16_t timeout;
    uint32_t rsv[4];
} aclrtLaunchKernelAttrValue;

typedef struct aclrtLaunchKernelAttr {
    aclrtLaunchKernelAttrId id;
    aclrtLaunchKernelAttrValue value;
} aclrtLaunchKernelAttr;

typedef struct aclrtLaunchKernelCfg {
    aclrtLaunchKernelAttr *attrs;
    size_t numAttrs;
} aclrtLaunchKernelCfg;

typedef enum {
    ACL_STREAM_ATTR_FAILURE_MODE         = 1,
    ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK = 2,
    ACL_STREAM_ATTR_USER_CUSTOM_TAG      = 3,
    ACL_STREAM_ATTR_CACHE_OP_IFNO        = 4,
} aclrtStreamAttr;

typedef union {
    uint64_t failureMode;
    uint32_t overflowSwitch;
    uint32_t userCustomTag;
    uint32_t cacheOpInfoSwitch;
    uint32_t reserve[4];
} aclrtStreamAttrValue;

typedef enum {
    ACL_DEV_ATTR_AICPU_CORE_NUM  = 1,    // aicpu number
    ACL_DEV_ATTR_AICORE_CORE_NUM = 101,  // aicore number
    ACL_DEV_ATTR_VECTOR_CORE_NUM = 201,  // vector core number
} aclrtDevAttr;

typedef enum {
    ACL_FEATURE_TSCPU_TASK_UPDATE_SUPPORT_AIC_AIV = 1,
    ACL_FEATURE_SYSTEM_MEMQ_EVENT_CROSS_DEV       = 21,
} aclrtDevFeatureType;

typedef enum {
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM   = 10,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX   = 11,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN   = 12,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL = 13,
} aclrtReduceKind;

typedef enum {
    ACL_RT_DEV_RES_CUBE_CORE = 0,
    ACL_RT_DEV_RES_VECTOR_CORE,
} aclrtDevResLimitType;

typedef enum {
    ACL_RT_EQUAL = 0,
    ACL_RT_NOT_EQUAL,
    ACL_RT_GREATER,
    ACL_RT_GREATER_OR_EQUAL,
    ACL_RT_LESS,
    ACL_RT_LESS_OR_EQUAL
} aclrtCondition;

typedef enum {
    ACL_RT_SWITCH_INT32 = 0,
    ACL_RT_SWITCH_INT64 = 1,
} aclrtCompareDataType;

typedef struct {
    aclrtCmoType cmoType;
    uint32_t barrierId;
} aclrtBarrierCmoInfo;

#define ACL_RT_CMO_MAX_BARRIER_NUM 6U

typedef struct {
    size_t barrierNum;
    aclrtBarrierCmoInfo cmoInfo[ACL_RT_CMO_MAX_BARRIER_NUM];
} aclrtBarrierTaskInfo;

#define ACL_RT_DEVS_TOPOLOGY_HCCS     0x01ULL
#define ACL_RT_DEVS_TOPOLOGY_PIX      0x02ULL
#define ACL_RT_DEVS_TOPOLOGY_PIB      0x04ULL
#define ACL_RT_DEVS_TOPOLOGY_PHB      0x08ULL
#define ACL_RT_DEVS_TOPOLOGY_SYS      0x10ULL
#define ACL_RT_DEVS_TOPOLOGY_SIO      0x20ULL
#define ACL_RT_DEVS_TOPOLOGY_HCCS_SW  0x40ULL

typedef struct {
    aclrtMemLocation dstLoc;
    aclrtMemLocation srcLoc;
    uint8_t rsv[16];
} aclrtMemcpyBatchAttr;

typedef struct aclrtIpcEventHandle {
    char reserved[ACL_IPC_EVENT_HANDLE_SIZE];
} aclrtIpcEventHandle;

/**
 * @ingroup AscendCL
 * @brief peek at last error by level
 *
 * @param level [IN] error level
 *
 * @retval Runtime error code
 */
ACL_FUNC_VISIBILITY aclError aclrtPeekAtLastError(aclrtLastErrLevel level);

/**
 * @ingroup AscendCL
 * @brief get last error by level
 *
 * @param level [IN] error level
 *
 * @retval Runtime error code
 */
ACL_FUNC_VISIBILITY aclError aclrtGetLastError(aclrtLastErrLevel level);


/**
 * @ingroup AscendCL
 * @brief Set a callback function to handle exception information
 *
 * @param callback [IN] callback function to handle exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);

/**
 * @ingroup AscendCL
 * @brief Get task id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The task id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get stream id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The stream id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get thread id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get device id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get error code from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The error code from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the callback function on the Stream
 *
 * @param threadId [IN] thread ID
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *
 * @param fn [IN]   Specify the callback function to be added
 *                  The function prototype of the callback function is:
 *                  typedef void (*aclrtCallback)(void *userData);
 * @param userData [IN]   User data to be passed to the callback function
 * @param blockType [IN]  callback block type
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType,
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger callback processing
 *
 * @par Function
 *  The thread processing callback specified by
 *  the aclrtSubscribeReport interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeReport
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the callback function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param threadId [IN]   thread ID
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create context and associates it with the calling thread
 *
 * @par Function
 * The following use cases are supported:
 * @li If you don't call the aclrtCreateContext interface
 * to explicitly create the context,
 * the system will use the default context, which is implicitly created
 * when the aclrtSetDevice interface is called.
 * @li If multiple contexts are created in a process
 * (there is no limit on the number of contexts),
 * the current thread can only use one of them at the same time.
 * It is recommended to explicitly specify the context of the current thread
 * through the aclrtSetCurrentContext interface to increase.
 * the maintainability of the program.
 *
 * @param  context [OUT]    point to the created context
 * @param  deviceId [IN]    device to create context on
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetDevice | aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief destroy context instance
 *
 * @par Function
 * Can only destroy context created through aclrtCreateContext interface
 *
 * @param  context [IN]   the context to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief set the context of the thread
 *
 * @par Function
 * The following scenarios are supported:
 * @li If the aclrtCreateContext interface is called in a thread to explicitly
 * create a Context (for example: ctx1), the thread's Context can be specified
 * without calling the aclrtSetCurrentContext interface.
 * The system uses ctx1 as the context of thread1 by default.
 * @li If the aclrtCreateContext interface is not explicitly created,
 * the system uses the default context as the context of the thread.
 * At this time, the aclrtDestroyContext interface cannot be used to release
 * the default context.
 * @li If the aclrtSetCurrentContext interface is called multiple times to
 * set the thread's Context, the last one prevails.
 *
 * @par Restriction
 * @li If the cevice corresponding to the context set for the thread
 * has been reset, you cannot set the context as the context of the thread,
 * otherwise a business exception will result.
 * @li It is recommended to use the context created in a thread.
 * If the aclrtCreateContext interface is called in thread A to create a context,
 * and the context is used in thread B,
 * the user must guarantee the execution order of tasks in the same stream
 * under the same context in two threads.
 *
 * @param  context [IN]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext | aclrtDestroyContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief get the context of the thread
 *
 * @par Function
 * If the user calls the aclrtSetCurrentContext interface
 * multiple times to set the context of the current thread,
 * then the last set context is obtained
 *
 * @param  context [OUT]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context);

/**
 * @ingroup AscendCL
 * @brief get system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[OUT] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[IN] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief get system param option value in current process
 *
 * @param opt[IN] system option
 * @param value[OUT] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current process
 *
 * @param opt[IN] system option
 * @param value[IN] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief Specify the device to use for the operation
 * implicitly create the default context and the default stream
 *
 * @par Function
 * The following use cases are supported:
 * @li Device can be specified in the process or thread.
 * If you call the aclrtSetDevice interface multiple
 * times to specify the same device,
 * you only need to call the aclrtResetDevice interface to reset the device.
 * @li The same device can be specified for operation
 *  in different processes or threads.
 * @li Device is specified in a process,
 * and multiple threads in the process can share this device to explicitly
 * create a Context (aclrtCreateContext interface).
 * @li In multi-device scenarios, you can switch to other devices
 * through the aclrtSetDevice interface in the process.
 *
 * @param  deviceId [IN]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtResetDevice |aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Reset the current operating Device and free resources on the device,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aclrtDestroyEvent interface to release Event or
 * call aclrtDestroyStream interface to release explicitly created Stream->
 * call aclrtDestroyContext to release explicitly created Context->
 * call aclrtResetDevice interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief get target device of current thread
 *
 * @param deviceId [OUT]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId);

/**
 * @ingroup AscendCL
 * @brief set stream failure mode
 *
 * @param stream [IN]  the stream to set
 * @param mode [IN]  stream failure mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);

/**
 * @ingroup AscendCL
 * @brief get target side
 *
 * @param runMode [OUT]    the run mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetRunMode(aclrtRunMode *runMode);

/**
 * @ingroup AscendCL
 * @brief Wait for compute device to finish
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDevice(void);

/**
 * @ingroup AscendCL
 * @brief Set Scheduling TS
 *
 * @param tsId [IN]   the ts id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetTsDevice(aclrtTsId tsId);

/**
 * @ingroup AscendCL
 * @brief get total device number.
 *
 * @param count [OUT]    the device number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEvent(aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag, event can be reused naturally
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy event instance
 *
 * @par Function
 *  Only events created through the aclrtCreateEvent interface can be
 *  destroyed, synchronous interfaces. When destroying an event,
 *  the user must ensure that the tasks involved in the aclrtSynchronizeEvent
 *  interface or the aclrtStreamWaitEvent interface are completed before
 *  they are destroyed.
 *
 * @param  event [IN]   event to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtSynchronizeEvent | aclrtStreamWaitEvent
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Record an Event in the Stream
 *
 * @param event [IN]    event to record
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Reset an event
 *
 * @par Function
 *  Users need to make sure to wait for the tasks in the Stream
 *  to complete before resetting the Event
 *
 * @param event [IN]    event to reset
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);

 /**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclrtQueryEvent is deprecated, use aclrtQueryEventStatus instead")
ACL_FUNC_VISIBILITY aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status);

/**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event recorded status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);

/**
* @ingroup AscendCL
* @brief Queries an event's wait-status
*
* @param  event [IN]    event to query
* @param  status [OUT]  event wait-status
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status);

/**
 * @ingroup AscendCL
 * @brief get an interprocess handle for a previously allocated event.
 *
 * @param [in]  event  event allocated with ACL_EVENT_IPC flags
 * @param [out] handle handle for interprocess
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle);

/**
 * @ingroup AscendCL
 * @brief opens an interprocess event handle for user in the current process.
 *
 * @param [in]  handle  interprocess handle to open
 * @param [out] event   returns the imported event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief computes the elapsed time between events.
 *
 * @param ms [OUT]     time between start and end in ms
 * @param start [IN]   starting event
 * @param end [IN]     ending event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtRecordEvent | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes and padded with 32 bytes
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMalloc interface needs to be released
 * through the aclrtFree interface.
 * @li Before calling the media data processing interface,
 * if you need to apply memory on the device to store input or output data,
 * you need to call acldvppMalloc to apply for memory.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | acldvppMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes with no padding
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocAlign32 interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocAlign32(void **devPtr,
                                                size_t size,
                                                aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief allocate memory on device with cache
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocCached interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocCached(void **devPtr,
                                               size_t size,
                                               aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief get memory attribute, host or device
 *
 * @param ptr [IN]         memory pointer
 * @param attributes [OUT] a buffer to store attributes
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtPointerGetAttributes(const void *ptr,
                                                       aclrtPtrAttributes *attributes);

/**
 * @ingroup AscendCL
 * @brief register an existing host memory range
 *
 * @param ptr [IN]     host pointer to memory to page-lock
 * @param size [IN]    size in bytes of the address range to page-lock in bytes
 * @param flag [IN]    flag for allocation request
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief flush cache data to ddr
 *
 * @param devPtr [IN]  the pointer that flush data to ddr
 * @param size [IN]    flush size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemFlush(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief invalidate cache data
 *
 * @param devPtr [IN]  pointer to invalidate cache data
 * @param size [IN]    invalidate size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemInvalidate(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free device memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief alloc memory on host
 *
 * @par Restriction
 * @li The requested memory cannot be used in the Device
 * and needs to be explicitly copied to the Device.
 * @li The memory requested by the aclrtMallocHost interface
 * needs to be released through the aclrtFreeHost interface.
 *
 * @param  hostPtr [OUT] pointer to pointer to allocated memory on the host
 * @param  size [IN]     alloc memory size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreeHost
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief allocate host memory with config
 *
 * @param  ptr [OUT]    pointer to allocated memory
 * @param  size [IN]    alloc memory size
 * @param  cfg [IN]     memory alloc config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHostWithCfg(void **ptr,
                                                    uint64_t size,
                                                    aclrtMallocConfig *cfg);

/**
 * @ingroup AscendCL
 * @brief free host memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocHost
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Performs a batch of memory copies synchronous.
 * @param [in] dsts         Array of destination pointers.
 * @param [in] destMax      Array of sizes for memcpy operations.
 * @param [in] srcs         Array of memcpy source pointers.
 * @param [in] sizes        Array of sizes for src memcpy operations.
 * @param [in] numBatches   Size of dsts, srcs and sizes arrays.
 * @param [in] attrs        Array of memcpy attributes.
 * @param [in] attrsIndexes Array of indices to specify which copies each entry in the attrs array applies to.
 *                          The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k]
 *                          through attrsIdxs[k+1] - 1. Also attrs[numAttrs-1] will apply to copies starting from
 *                          attrsIdxs[numAttrs-1] through count - 1.
 * @param [in] numAttrs     Size of attrs and attrsIdxs arrays.
 * @param [out] failIdx     Pointer to a location to return the index of the copy where a failure was encountered.
 *                          The value will be SIZE_MAX if the error doesn't pertain to any specific copy.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                                              size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                                              size_t numAttrs, size_t *failIdx);


/**
 * @ingroup AscendCL
 * @brief Performs a batch of memory copies synchronous.
 * @param [in] dsts         Array of destination pointers.
 * @param [in] destMax      Array of sizes for memcpy operations.
 * @param [in] srcs         Array of memcpy source pointers.
 * @param [in] sizes        Array of sizes for src memcpy operations.
 * @param [in] numBatches   Size of dsts, srcs and sizes arrays.
 * @param [in] attrs        Array of memcpy attributes.
 * @param [in] attrsIdxs    Array of indices to specify which copies each entry in the attrs array applies to.
 *                          The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k]
 *                          through attrsIdxs[k+1] - 1. Also attrs[numAttrs-1] will apply to copies starting from
 *                          attrsIdxs[numAttrs-1] through count - 1.
 * @param [in] numAttrs     Size of attrs and attrsIdxs arrays.
 * @param [out] failIdx     Pointer to a location to return the index of the copy where a failure was encountered.
 *                          The value will be SIZE_MAX if the error doesn't pertain to any specific copy.
 * @param [in] stream       stream handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatchAsync(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                                                   size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                                                   size_t numAttrs, size_t *failIndex, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device, would
 *         be synchronous if memory is not allocated via calling acl or rts api.
 *
 * @par Function
 *  After calling this interface and memory is allocated via calling acl or rts api,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsyncWithCondition(void *dst,
                                                           size_t destMax,
                                                           const void *src,
                                                           size_t count,
                                                           aclrtMemcpyKind kind,
                                                           aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Performs a batch of memory copies synchronous.
 * @param [in] dsts         Array of destination pointers.
 * @param [in] destMax      Array of sizes for memcpy operations.
 * @param [in] srcs         Array of memcpy source pointers.
 * @param [in] sizes        Array of sizes for src memcpy operations.
 * @param [in] numBatches   Size of dsts, srcs and sizes arrays.
 * @param [in] attrs        Array of memcpy attributes.
 * @param [in] attrsIndexes Array of indices to specify which copies each entry in the attrs array applies to.
 *                          The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k]
 *                          through attrsIdxs[k+1] - 1. Also attrs[numAttrs-1] will apply to copies starting from
 *                          attrsIdxs[numAttrs-1] through count - 1.
 * @param [in] numAttrs     Size of attrs and attrsIdxs arrays.
 * @param [out] failIdx     Pointer to a location to return the index of the copy where a failure was encountered.
 *                          The value will be SIZE_MAX if the error doesn't pertain to any specific copy.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                                              size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                                              size_t numAttrs, size_t *failIdx);


/**
 * @ingroup AscendCL
 * @brief Performs a batch of memory copies synchronous.
 * @param [in] dsts         Array of destination pointers.
 * @param [in] destMax      Array of sizes for memcpy operations.
 * @param [in] srcs         Array of memcpy source pointers.
 * @param [in] sizes        Array of sizes for src memcpy operations.
 * @param [in] numBatches   Size of dsts, srcs and sizes arrays.
 * @param [in] attrs        Array of memcpy attributes.
 * @param [in] attrsIdxs    Array of indices to specify which copies each entry in the attrs array applies to.
 *                          The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k]
 *                          through attrsIdxs[k+1] - 1. Also attrs[numAttrs-1] will apply to copies starting from
 *                          attrsIdxs[numAttrs-1] through count - 1.
 * @param [in] numAttrs     Size of attrs and attrsIdxs arrays.
 * @param [out] failIdx     Pointer to a location to return the index of the copy where a failure was encountered.
 *                          The value will be SIZE_MAX if the error doesn't pertain to any specific copy.
 * @param [in] stream       stream handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatchAsync(void **dsts, size_t *destMax, void **srcs, size_t *sizes,
                                                   size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                                                   size_t numAttrs, size_t *failIndex, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2d(void *dst,
                                           size_t dpitch,
                                           const void *src,
                                           size_t spitch,
                                           size_t width,
                                           size_t height,
                                           aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief asynchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 * @param stream [IN]    asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2dAsync(void *dst,
                                                size_t dpitch,
                                                const void *src,
                                                size_t spitch,
                                                size_t width,
                                                size_t height,
                                                aclrtMemcpyKind kind,
                                                aclrtStream stream);

/**
* @ingroup AscendCL
* @brief Asynchronous initialize memory
* and set contents of memory to specified value async
*
* @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
* @param devPtr [IN]      destination address pointer
* @param maxCount [IN]    Max length of destination address memory
* @param value [IN]       set value
* @param count [IN]       the number of byte to set
* @param stream [IN]      asynchronized task stream
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*
* @see aclrtSynchronizeStream
*/
ACL_FUNC_VISIBILITY aclError aclrtMemsetAsync(void *devPtr,
                                              size_t maxCount,
                                              int32_t value,
                                              size_t count,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Allocate an address range reservation
 *
 * @param virPtr [OUT]    Resulting pointer to start of virtual address range allocated
 * @param size [IN]       Size of the reserved virtual address range requested
 * @param alignment [IN]  Alignment of the reserved virtual address range requested
 * @param expectPtr [IN]  Fixed starting address range requested, must be nullptr
 * @param flags [IN]      Flag of page type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReleaseMemAddress | aclrtMallocPhysical | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr,
                                                    size_t size,
                                                    size_t alignment,
                                                    void *expectPtr,
                                                    uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Free an address range reservation
 *
 * @param virPtr [IN]  Starting address of the virtual address range to free
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReserveMemAddress
 */
ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief Create a memory handle representing a memory allocation of a given
 * size described by the given properties
 *
 * @param handle [OUT]  Value of handle returned. All operations on this
 * allocation are to be performed using this handle.
 * @param size [IN]     Size of the allocation requested
 * @param prop [IN]     Properties of the allocation to create
 * @param flags [IN]    Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreePhysical | aclrtReserveMemAddress | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle,
                                                 size_t size,
                                                 const aclrtPhysicalMemProp *prop,
                                                 uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Release a memory handle representing a memory allocation which was
 * previously allocated through aclrtMallocPhysical
 *
 * @param handle [IN]  Value of handle which was returned previously by aclrtMallocPhysical
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle);

/**
 * @ingroup AscendCL
 * @brief Maps an allocation handle to a reserved virtual address range
 *
 * @param virPtr [IN]  Address where memory will be mapped
 * @param size [IN]    Size of the memory mapping
 * @param offset [IN]  Offset into the memory represented by handle from which to start mapping
 * @param handle [IN]  Handle to a shareable memory
 * @param flags [IN]   Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtUnmapMem | aclrtReserveMemAddress | aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr,
                                         size_t size,
                                         size_t offset,
                                         aclrtDrvMemHandle handle,
                                         uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Unmap the backing memory of a given address range
 *
 * @param virPtr [IN]  Starting address for the virtual address range to unmap
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream);

/**
 * @ingroup AscendCL
 * @brief  create stream instance with param
 *
 * @par Function
 * Can create fast streams through the aclrtCreateStreamWithConfig interface
 *
 * @param  stream [OUT]   the created stream
 * @param  priority [IN]   the priority of stream, value range:0~7
 * @param  flag [IN]   indicate the function for stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @par Restriction
 * Before calling the aclrtDestroyStream interface to destroy
 * the specified Stream, you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the Stream have been completed.
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance by force
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamForce(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is complete immediately
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Query a stream for completion status.
 *
 * @param  stream [IN]   the stream to query
 * @param  status [OUT]  stream status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);

/**
 * @ingroup AscendCL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed.
 * Support for multiple streams waiting for the same event.
 *
 * @param  stream [IN]   the wait stream If using thedefault Stream, set NULL
 * @param  event [IN]    the event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief set group
 *
 * @par Function
 *  set the task to the corresponding group
 *
 * @param groupId [IN]   group id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo | aclrtGetGroupInfoDetail
 */
ACL_FUNC_VISIBILITY aclError aclrtSetGroup(int32_t groupId);

/**
 * @ingroup AscendCL
 * @brief get the number of group
 *
 * @par Function
 *  get the number of group. if the number of group is zero,
 *  it means that group is not supported or group is not created.
 *
 * @param count [OUT]   the number of group
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create group information
 *
 * @retval null for failed.
 * @retval OtherValues success.
 *
 * @see aclrtDestroyGroupInfo
 */
ACL_FUNC_VISIBILITY aclrtGroupInfo *aclrtCreateGroupInfo();

/**
 * @ingroup AscendCL
 * @brief destroy group information
 *
 * @param groupInfo [IN]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get all group information
 *
 * @param groupInfo [OUT]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount
 */
ACL_FUNC_VISIBILITY aclError aclrtGetAllGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get detail information of group
 *
 * @param groupInfo [IN]    pointer to group information
 * @param groupIndex [IN]   group index value
 * @param attr [IN]         group attribute
 * @param attrValue [OUT]   pointer to attribute value
 * @param valueLen [IN]     length of attribute value
 * @param paramRetSize [OUT]   pointer to real length of attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupInfoDetail(const aclrtGroupInfo *groupInfo,
                                                     int32_t groupIndex,
                                                     aclrtGroupAttr attr,
                                                     void *attrValue,
                                                     size_t valueLen,
                                                     size_t *paramRetSize);

/**
 * @ingroup AscendCL
 * @brief checking whether current device and peer device support the p2p feature
 *
 * @param canAccessPeer [OUT]   pointer to save the checking result
 * @param deviceId [IN]         current device id
 * @param peerDeviceId [IN]     peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceEnablePeerAccess | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief enable the peer device to support the p2p feature
 *
 * @param peerDeviceId [IN]   the peer device id
 * @param flags [IN]   reserved field, now it must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);

/**
 * @ingroup AscendCL
 * @brief disable the peer device to support the p2p function
 *
 * @param peerDeviceId [IN]   the peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceEnablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief Obtain the free memory and total memory of specified attribute.
 * the specified memory include normal memory and huge memory.
 *
 * @param attr [IN]    the memory attribute of specified device
 * @param free [OUT]   the free memory of specified device
 * @param total [OUT]  the total memory of specified device.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for waitting of op
 *
 * @param timeout [IN]   op wait timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for op executing
 *
 * @param timeout [IN]   op execute timeout, the unit is seconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief enable or disable overflow switch on some stream
 * @param stream [IN]   set overflow switch on this stream
 * @param flag [IN]  0 : disable 1 : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief get overflow switch on some stream
 * @param stream [IN]   get overflow switch on this stream
 * @param flag [OUT]  current overflow switch, 0 : disable others : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag);

/**
 * @ingroup AscendCL
 * @brief Query the comprehensive usage rate of device
 * @param deviceId [IN] the need query's deviceId
 * @param utilizationInfo [IN] the need query's device unit switch
 * @param utilizationInfo [OUT] the usage rate of device
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo);

/**
 * @ingroup AscendCL
 * @brief set saturation mode
 * @param mode [IN]   target saturation mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode);

/**
 * @ingroup AscendCL
 * @brief get overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtGetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputAddr [IN/OUT]  output device addr to store overflow status
 * @param outputSize [IN]  output addr size
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetOverflowStatus(void *outputAddr, size_t outputSize, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief reset overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtResetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetOverflowStatus(aclrtStream stream);

/**
* @ingroup AscendCL
* @brief cache manager operation
* @param [in] src  device memory address
* @param [in] size  memory size
* @param [in] cmoType  type of operation, currently, only ACL_RT_CMO_TYPE_PREFETCH is supported
* @param [in] stream   stream handle
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCmoAsync(void *src, size_t size, aclrtCmoType cmoType, aclrtStream stream);

/**`
 * @ingroup AscendCL
 * @brief get the mem uce info
 * @param [in] deviceId
 * @param [in/out] memUceInfoArray
 * @param [in] arraySize
 * @param [out] retSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray,
                                                size_t arraySize, size_t *retSize);

/**`
 * @ingroup AscendCL
 * @brief get the mem usage info
 * @param [in] deviceId
 * @param [in/out] memUsageInfo
 * @param [in] inputNum
 * @param [out] outputNum
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemUsageInfo(uint32_t deviceId, aclrtMemUsageInfo *memUsageInfo, size_t inputNum, size_t *outputNum);

/**
 * @ingroup AscendCL
 * @brief stop the task on specified device
 * @param [in] deviceId
 * @param [in] timeout
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceTaskAbort(int32_t deviceId, uint32_t timeout);

/**`
 * @ingroup AscendCL
 * @brief repair the mem uce
 * @param [in] deviceId
 * @param [in/out] memUceInfoArray
 * @param [in] arraySize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray, size_t arraySize);

/**`
 * @ingroup AscendCL
 * @brief abort unexecuted tasks and pause executing tasks on the stream
 * @param [in] stream  stream to be aborted, cannot be null
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamAbort(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Get the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @param [in] type      resources type
 * @param [out] value    resources limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief Set the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @param [in] type      resource type
 * @param [in] value     resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value);

/**
 * @ingroup AscendCL
 * @brief Reset the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDeviceResLimit(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Get the value of the limited resources of the specified stream
 * @param [in] stream   the stream handle
 * @param [in] type     resource type
 * @param [out] value   resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief Reset the value of the limited resources of the specified stream
 * @param [in] stream   the stream handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetStreamResLimit(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Use stream resource in current thread
 * @param stream [in]  stream to use
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUseStreamResInCurrentThread(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Not use stream resource in current thread
 * @param stream [in]  stream to not use
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnuseStreamResInCurrentThread(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Get the value of the limited resources of the current thread
 * @param type [in]   resource type
 * @param value [out] resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief Set the operation execution timeout for the current thread
 * @param [in]  timeout        Desired operation timeout value (in microsecond)
 * @param [out] actualTimeout  Pointer to a uint64_t variable to store the actual timeout value applied
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout);

/**
 * @ingroup AscendCL
 * @brief set stream attribute
 * @param [in] stream       stream handle
 * @param [in] stmAttrType  stream attribute type, which value can be:
 *                             ACL_STREAM_ATTR_FAILURE_MODE, ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK
 *                             or ACL_STREAM_ATTR_USER_CUSTOM_TAG
 * @param [in] value        stream attribute value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType,
    aclrtStreamAttrValue *value);

/**
 * @ingroup AscendCL
 * @brief get uuid of device by device id
 * @param [in] deviceId        device id
 * @param [out] uuid           16-byte Universally Unique Identifier for
 *                             globally unique identification of an NPU device.
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetUuid(int32_t deviceId, aclrtUuid *uuid);

/**
 * @ingroup AscendCL
 * @brief Write data to the specified memory. Asynchronous Interface.
 *
 * @param devAddr [IN]  Memory address on the Device side.
 * @param value [IN]   The data to be written into the memory.
 * @param flag [IN]   Reserved parameter, currently fixed to 0.
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtValueWrite(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Unblock after the data in the specified memory meets certain conditions. Asynchronous Interface.
 *
 * @param devAddr [IN]  Memory address on the Device side.
 * @param value [IN]   The value to be compared with the data in the memory.
 * @param flag [IN]    comparison logic.
 * @param stream [IN]  stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtValueWait(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *        Difference between this api and "aclrtLaunchCallback" is that
 *        thread will be created and registered inside this interface
 *        automatically, while "aclrtLaunchCallback" need manual registration.
 *        For details please refer to official API document
 * @param [in] stream   the stream to launch callback func
 * @param [in] func     callback func to launch
 *                      The function prototype of the callback function is:
 *                      typedef void (*aclrtHostFunc)(void *args);
 * @param [in] args     args for callback func
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc func, void *args);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_H_

