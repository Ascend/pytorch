/**
* @file acl_prof.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2021. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_PROF_H_
#define INC_EXTERNAL_ACL_PROF_H_

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_PROF_ACL_API                0x0001ULL
#define ACL_PROF_TASK_TIME              0x0002ULL
#define ACL_PROF_AICORE_METRICS         0x0004ULL
#define ACL_PROF_AICPU                  0x0008ULL
#define ACL_PROF_L2CACHE                0x0010ULL
#define ACL_PROF_HCCL_TRACE             0x0020ULL
#define ACL_PROF_TRAINING_TRACE         0x0040ULL
#define ACL_PROF_MSPROFTX               0x0080ULL
#define ACL_PROF_RUNTIME_API            0x0100ULL
#define ACL_PROF_TASK_TIME_L0           0x0800ULL
#define ACL_PROF_TASK_MEMORY            0x1000ULL

/**
 * @deprecated please use aclprofGetOpTypeLen and aclprofGetOpTNameLen instead
 */
#define ACL_PROF_MAX_OP_NAME_LEN        257
#define ACL_PROF_MAX_OP_TYPE_LEN        65

typedef enum {
    ACL_AICORE_ARITHMETIC_UTILIZATION = 0,
    ACL_AICORE_PIPE_UTILIZATION = 1,
    ACL_AICORE_MEMORY_BANDWIDTH = 2,
    ACL_AICORE_L0B_AND_WIDTH = 3,
    ACL_AICORE_RESOURCE_CONFLICT_RATIO = 4,
    ACL_AICORE_MEMORY_UB = 5,
    ACL_AICORE_L2_CACHE = 6,
    ACL_AICORE_NONE = 0xFF
} aclprofAicoreMetrics;

typedef enum {
    ACL_STEP_START = 0, // step  start
    ACL_STEP_END = 1   // step  end
} aclprofStepTag;


typedef struct aclprofConfig aclprofConfig;
typedef struct aclprofStopConfig aclprofStopConfig;
typedef struct aclprofAicoreEvents aclprofAicoreEvents;
typedef struct aclprofSubscribeConfig aclprofSubscribeConfig;
typedef struct aclprofStepInfo aclprofStepInfo;

/**
 * @ingroup AscendCL
 * @brief profiling initialize
 *
 * @param  profilerResultPath [IN]  path of profiling result
 * @param  length [IN]              length of profilerResultPath
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofFinalize
 */
MSVP_PROF_API aclError aclprofInit(const char *profilerResultPath, size_t length);

/**
 * @ingroup AscendCL
 * @brief profiling finalize
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofInit
 */
MSVP_PROF_API aclError aclprofFinalize();

/**
 * @ingroup AscendCL
 * @brief Start profiling modules by profilerConfig
 *
 * @param  profilerConfig [IN]  config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofStop
 */
MSVP_PROF_API aclError aclprofStart(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclprofConfig
 *
 * @param  deviceIdList [IN]      list of device id
 * @param  deviceNums [IN]        number of devices
 * @param  aicoreMetrics [IN]     type of aicore metrics
 * @param  aicoreEvents [IN]      pointer to aicore events, only support NULL now
 * @param  dataTypeConfig [IN]    config modules need profiling
 *
 * @retval the aclprofConfig pointer
 *
 * @see aclprofDestroyConfig
 */
MSVP_PROF_API aclprofConfig *aclprofCreateConfig(uint32_t *deviceIdList, uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics, aclprofAicoreEvents *aicoreEvents, uint64_t dataTypeConfig);

/**
 * @ingroup AscendCL
 * @brief Destroy data of type aclprofConfig
 *
 * @param  profilerConfig [IN]  config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofCreateConfig
 */
MSVP_PROF_API aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief stop profiling modules by stopProfilingConfig
 *
 * @param  profilerConfig [IN]  pointer to stop config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofStart
 */
MSVP_PROF_API aclError aclprofStop(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief subscribe profiling data of model
 *
 * @param  modelId [IN]              the model id subscribed
 * @param  profSubscribeConfig [IN]  pointer to config of model subscribe
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofModelUnSubscribe
 */
MSVP_PROF_API aclError aclprofModelSubscribe(uint32_t modelId,
    const aclprofSubscribeConfig *profSubscribeConfig);

/**
 * @ingroup AscendCL
 * @brief unsubscribe profiling data of model
 *
 * @param  modelId [IN]  the model id unsubscribed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofModelSubscribe
 */
MSVP_PROF_API aclError aclprofModelUnSubscribe(uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief create subscribe config
 *
 * @param  timeInfoSwitch [IN] switch whether get time info from model
 * @param  aicoreMetrics [IN]  aicore metrics
 * @param  fd [IN]             pointer to write pipe
 *
 * @retval the aclprofSubscribeConfig pointer
 *
 * @see aclprofDestroySubscribeConfig
 */
MSVP_PROF_API aclprofSubscribeConfig *aclprofCreateSubscribeConfig(int8_t timeInfoSwitch,
    aclprofAicoreMetrics aicoreMetrics, void *fd);

/**
 * @ingroup AscendCL
 * @brief destroy subscribe config
 *
 * @param  profSubscribeConfig [IN]  subscribe config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofCreateSubscribeConfig
 */
MSVP_PROF_API aclError aclprofDestroySubscribeConfig(const aclprofSubscribeConfig *profSubscribeConfig);

/**
 * @ingroup AscendCL
 * @brief create subscribe config
 *
 * @param  opDescSize [OUT]  size of op desc
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpDescSize(size_t *opDescSize);

/**
 * @ingroup AscendCL
 * @brief get op number from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  opNumber [OUT]  op number of subscription data
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpNum(const void *opInfo, size_t opInfoLen, uint32_t *opNumber);

/**
 * @ingroup AscendCL
 * @brief get length op type from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opTypeLen [OUT]  actual length of op type string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpTypeLen(const void *opInfo, size_t opInfoLen, uint32_t index,
    size_t *opTypeLen);

/**
 * @ingroup AscendCL
 * @brief get op type from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opType [OUT]     obtained op type string
 * @param  opTypeLen [IN]   obtained length of op type string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpType(const void *opInfo, size_t opInfoLen, uint32_t index,
    char *opType, size_t opTypeLen);

/**
 * @ingroup AscendCL
 * @brief get length op name from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opNameLen [OUT]  actual length of op name string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpNameLen(const void *opInfo, size_t opInfoLen, uint32_t index,
    size_t *opNameLen);

/**
 * @ingroup AscendCL
 * @brief get op type from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opName [OUT]     obtained op name string
 * @param  opNameLen [IN]   obtained length of op name string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
MSVP_PROF_API aclError aclprofGetOpName(const void *opInfo, size_t opInfoLen, uint32_t index,
    char *opName, size_t opNameLen);

/**
 * @ingroup AscendCL
 * @brief get start time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval start time(us) of specified op with timestamp
 * @retval 0 for failed
 */
MSVP_PROF_API uint64_t aclprofGetOpStart(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get end time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval end time(us) of specified op with timestamp
 * @retval 0 for failed
 */
MSVP_PROF_API uint64_t aclprofGetOpEnd(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get excution time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval execution time(us) of specified op with timestamp
 * @retval 0 for failed
 */
MSVP_PROF_API uint64_t aclprofGetOpDuration(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get model id from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 *
 * @retval model id of subscription data
 * @retval 0 for failed
 */
MSVP_PROF_API size_t aclprofGetModelId(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief
 *
 * @param  stepInfo [IN]     pointer to stepInfo data
 * @param  aclprofstepTag [IN] start or end flag
 * @param  stream [IN] steam info
 *
 * @retval 0 for failed
 */
MSVP_PROF_API aclError aclprofGetStepTimestamp(aclprofStepInfo* stepInfo, aclprofStepTag tag, aclrtStream stream);

 /**
 * @ingroup AscendCL
 * @brief create pointer to aclprofStepInfo data
 *
 *
 * @retval aclprofStepInfo pointer
 */
MSVP_PROF_API aclprofStepInfo* aclprofCreateStepInfo();

 /**
 * @ingroup AscendCL
 * @brief destroy aclprofStepInfo pointer
 *
 *
 * @retval void
 */
MSVP_PROF_API void aclprofDestroyStepInfo(aclprofStepInfo* stepinfo);

/**
* @ingroup AscendCL
* @brief create pointer to aclprofstamp
*
*
* @retval aclprofStamp pointer
*/
MSVP_PROF_API void *aclprofCreateStamp();

/**
* @ingroup AscendCL
* @brief destory stamp pointer
*
*
* @retval void
*/
MSVP_PROF_API void aclprofDestroyStamp(void *stamp);

/**
* @ingroup AscendCL
* @brief Record push timestamp
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
MSVP_PROF_API aclError aclprofPush(void *stamp);

/**
* @ingroup AscendCL
* @brief Record pop timestamp
*
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
MSVP_PROF_API aclError aclprofPop();

/**
* @ingroup AscendCL
* @brief Record range start timestamp
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
MSVP_PROF_API aclError aclprofRangeStart(void *stamp, uint32_t *rangeId);

/**
* @ingroup AscendCL
* @brief Record range end timestamp
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
MSVP_PROF_API aclError aclprofRangeStop(uint32_t rangeId);

/**
* @ingroup AscendCL
* @brief set message to stamp
*
*
* @retval void
*/
MSVP_PROF_API aclError aclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen);

/**
* @ingroup AscendCL
* @brief Record mark timestamp
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
MSVP_PROF_API aclError aclprofMark(void *stamp);
#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_PROF_H_
