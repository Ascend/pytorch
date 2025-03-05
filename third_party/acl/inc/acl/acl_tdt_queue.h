/**
* @file acl_tdt_queue.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_TDT_QUEUE_H_
#define INC_EXTERNAL_ACL_ACL_TDT_QUEUE_H_

#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_TDT_QUEUE_PERMISSION_MANAGE 1
#define ACL_TDT_QUEUE_PERMISSION_DEQUEUE 2
#define ACL_TDT_QUEUE_PERMISSION_ENQUEUE 4

#define ACL_TDT_QUEUE_ROUTE_UNBIND 0
#define ACL_TDT_QUEUE_ROUTE_BIND 1
#define ACL_TDT_QUEUE_ROUTE_BIND_ABNORMAL 2

typedef void *acltdtBuf;
typedef struct tagMemQueueAttr acltdtQueueAttr;
typedef struct acltdtQueueRouteList acltdtQueueRouteList;
typedef struct acltdtQueueRouteQueryInfo acltdtQueueRouteQueryInfo;
typedef struct acltdtQueueRoute acltdtQueueRoute;

typedef enum {
    ACL_TDT_QUEUE_NAME_PTR = 0,
    ACL_TDT_QUEUE_DEPTH_UINT32
} acltdtQueueAttrType;

typedef enum {
    ACL_TDT_QUEUE_ROUTE_SRC_UINT32 = 0,
    ACL_TDT_QUEUE_ROUTE_DST_UINT32,
    ACL_TDT_QUEUE_ROUTE_STATUS_INT32
} acltdtQueueRouteParamType;

typedef enum {
    ACL_TDT_QUEUE_ROUTE_QUERY_SRC = 0,
    ACL_TDT_QUEUE_ROUTE_QUERY_DST = 1,
    ACL_TDT_QUEUE_ROUTE_QUERY_SRC_AND_DST = 2,
    ACL_TDT_QUEUE_ROUTE_QUERY_ABNORMAL = 100
} acltdtQueueRouteQueryMode;

typedef enum {
    ACL_TDT_QUEUE_ROUTE_QUERY_MODE_ENUM = 0,
    ACL_TDT_QUEUE_ROUTE_QUERY_SRC_ID_UINT32,
    ACL_TDT_QUEUE_ROUTE_QUERY_DST_ID_UINT32
} acltdtQueueRouteQueryInfoParamType;

typedef enum {
    ACL_TDT_NORMAL_MEM = 0,
    ACL_TDT_DVPP_MEM
} acltdtAllocBufType;

/**
 * @ingroup AscendCL
 * @brief create queue
 *
 * @param attr [IN] pointer to the queue attr
 * @param qid [OUT] pointer to the qid
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtDestroyQueue
 */
ACL_FUNC_VISIBILITY aclError acltdtCreateQueue(const acltdtQueueAttr *attr, uint32_t *qid);

/**
 * @ingroup AscendCL
 * @brief destroy queue
 *
 * @param qid [IN] qid which to be destroyed
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueue
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyQueue(uint32_t qid);

/**
 * @ingroup AscendCL
 * @brief enqueue function
 *
 * @param qid [IN] qid
 * @param buf [IN] acltdtBuf
 * @param timeout [IN] timeout, -1 means blocking
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtDequeue
 */
ACL_FUNC_VISIBILITY aclError acltdtEnqueue(uint32_t qid, acltdtBuf buf, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief dequeue function
 *
 * @param qid [IN] qid
 * @param buf [OUT] pointer to the acltdtBuf
 * @param timeout [IN] timeout, -1 means blocking
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtEnqueue
 */
ACL_FUNC_VISIBILITY aclError acltdtDequeue(uint32_t qid, acltdtBuf *buf, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief enqueue function
 *
 * @param qid [IN] qid
 * @param data [IN] the pointer to data buf
 * @param dataSize [IN] the size of data buf
 * @param userData [IN] the pointer to user data buf
 * @param userDataSize [IN] the size of user data buf
 * @param timeout [IN] timeout, -1 means blocking
 * @param rsv [IN] reserved param
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtDequeueData
 */
ACL_FUNC_VISIBILITY aclError acltdtEnqueueData(uint32_t qid, const void *data, size_t dataSize,
    const void *userData, size_t userDataSize, int32_t timeout, uint32_t rsv);

/**
 * @ingroup AscendCL
 * @brief dequeue function
 *
 * @param qid [IN] qid
 * @param data [IN|OUT] the pointer to data buf
 * @param dataSize [IN] the size of data buf
 * @param retDataSize [OUT] the return size of data buf
 * @param userData [IN|OUT] the pointer to user data buf
 * @param userDataSize [IN] the size of user data buf
 * @param timeout [IN] timeout, -1 means blocking
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtEnqueueData
 */
ACL_FUNC_VISIBILITY aclError acltdtDequeueData(uint32_t qid, void *data, size_t dataSize, size_t *retDataSize,
    void *userData, size_t userDataSize, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief grant queue to other process
 *
 * @param qid [IN] qid
 * @param pid [IN] pid of dst process
 * @param permission [IN] permission of queue
 * @param timeout [IN] timeout, -1 means blocking
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see ACL_TDT_QUEUE_PERMISSION_MANAGE | ACL_TDT_QUEUE_PERMISSION_DEQUEUE | ACL_TDT_QUEUE_PERMISSION_ENQUEUE
 */
ACL_FUNC_VISIBILITY aclError acltdtGrantQueue(uint32_t qid, int32_t pid, uint32_t permission, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief attach queue in current process
 *
 * @param qid [IN] qid
 * @param timeout [IN] timeout, -1 means blocking
 * @param permission [OUT] permission of queue
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtGrantQueue
 */
ACL_FUNC_VISIBILITY aclError acltdtAttachQueue(uint32_t qid, int32_t timeout, uint32_t *permission);

/**
 * @ingroup AscendCL
 * @brief bind queue routes
 *
 * @param qRouteList [IN|OUT] pointer to the route list
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acltdtBindQueueRoutes(acltdtQueueRouteList *qRouteList);

/**
 * @ingroup AscendCL
 * @brief unbind queue routes
 *
 * @param qRouteList [IN|OUT] pointer to the route list
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acltdtUnbindQueueRoutes(acltdtQueueRouteList *qRouteList);

/**
 * @ingroup AscendCL
 * @brief query queue routes according to query mode
 *
 * @param queryInfo [IN] pointer to the queue route query info
 * @param qRouteList [IN|OUT] pointer to the route list
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acltdtQueryQueueRoutes(const acltdtQueueRouteQueryInfo *queryInfo,
                                                    acltdtQueueRouteList *qRouteList);

/**
 * @ingroup AscendCL
 * @brief alloc acltdtBuf
 *
 * @param size [IN] alloc buf size
 * @param type [IN] reserved parameters, need to set zero currently
 * @param buf [OUT] pointer to the acltdtBuf
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtFreeBuf
 */
ACL_FUNC_VISIBILITY aclError acltdtAllocBuf(size_t size, uint32_t type, acltdtBuf *buf);

/**
 * @ingroup AscendCL
 * @brief free acltdtBuf
 *
 * @param buf [IN] pointer to the acltdtBuf
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtAllocBuf
 */
ACL_FUNC_VISIBILITY aclError acltdtFreeBuf(acltdtBuf buf);

/**
 * @ingroup AscendCL
 * @brief get data buf address
 *
 * @param buf [IN] acltdtBuf
 * @param dataPtr [OUT] pointer to the data ptr which is acquired from acltdtBuf
 * @param size [OUT] pointer to the size
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtAllocBuf
 */
ACL_FUNC_VISIBILITY aclError acltdtGetBufData(const acltdtBuf buf, void **dataPtr, size_t *size);

/**
 * @ingroup AscendCL
 * @brief set data buf effective len
 *
 * @param buf [IN] acltdtBuf
 * @param len [IN] set effective len to data buf which must be smaller than size acquired by acltdtGetBufData
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtGetBufData acltdtGetBufDataLen
 */
ACL_FUNC_VISIBILITY aclError acltdtSetBufDataLen(acltdtBuf buf, size_t len);

/**
 * @ingroup AscendCL
 * @brief get data buf effective len
 *
 * @param buf [IN] acltdtBuf
 * @param len [OUT] get effective len which is set by acltdtSetBufDataLen
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtSetBufDataLen
 */
ACL_FUNC_VISIBILITY aclError acltdtGetBufDataLen(acltdtBuf buf, size_t *len);

/**
 * @ingroup AscendCL
 * @brief append acltdtBuf to acltdtBuf chain
 *
 * @param headBuf [IN] acltdtBuf chain head
 * @param buf [IN] acltdtBuf will be appended
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtAppendBufChain(acltdtBuf headBuf, acltdtBuf buf);

/**
 * @ingroup AscendCL
 * @brief get acltdtBuf chain total size
 *
 * @param headBuf [IN] acltdtBuf chain head
 * @param num [OUT] acltdtBuf chain total size
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtAppendBufChain
 */
ACL_FUNC_VISIBILITY aclError acltdtGetBufChainNum(acltdtBuf headBuf, uint32_t *num);

/**
 * @ingroup AscendCL
 * @brief get acltdtBuf from acltdtBuf chain by index
 *
 * @param headBuf [IN] acltdtBuf chain head
 * @param index [IN] the index which is smaller than num acquired from acltdtGetBufChainNum
 * @param buf [OUT] the acltdtBuf from acltdtBuf on index
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtAppendBufChain acltdtGetBufChainNum
 */
ACL_FUNC_VISIBILITY aclError acltdtGetBufFromChain(acltdtBuf headBuf, uint32_t index, acltdtBuf *buf);

/**
 * @ingroup AscendCL
 * @brief get private data buf address and size
 *
 * @param buf [IN] acltdtBuf
 * @param dataPtr [IN/OUT] pointer to the user ptr
 * @param size [IN] the current private data area size, less than or equal to 96B
 * @param offset [IN] address offset, less than or equal to 96B
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtGetBufUserData
 */
ACL_FUNC_VISIBILITY aclError acltdtGetBufUserData(const acltdtBuf buf, void *dataPtr, size_t size, size_t offset);

/**
 * @ingroup AscendCL
 * @brief set private data buf address and size
 *
 * @param buf [OUT] acltdtBuf
 * @param dataPtr [IN] pointer to the user ptr
 * @param size [IN] the current private data area size, less than or equal to 96B
 * @param offset [IN] address offset, less than or equal to 96B
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtSetBufUserData
 */
ACL_FUNC_VISIBILITY aclError acltdtSetBufUserData(acltdtBuf buf, const void *dataPtr, size_t size, size_t offset);

/**
 * @ingroup AscendCL
 * @brief copy buf ref
 *
 * @param buf [IN] acltdtBuf
 * @param newBuf [OUT] Make a reference copy of the data area of buf and
 *                     create a new buf header pointing to the same data area
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCopyBufRef
 */
ACL_FUNC_VISIBILITY aclError acltdtCopyBufRef(const acltdtBuf buf, acltdtBuf *newBuf);

/**
 * @ingroup AscendCL
 * @brief Create the queue attr
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyQueueAttr
 */
ACL_FUNC_VISIBILITY acltdtQueueAttr *acltdtCreateQueueAttr();

/**
 * @ingroup AscendCL
 * @brief Destroy the queue attr
 *
 * @param attr [IN]  pointer to the queue attr
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueAttr
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyQueueAttr(const acltdtQueueAttr *attr);

/**
 * @ingroup AscendCL
 * @brief Set parameter for queue attr
 *
 * @param attr [IN|OUT] pointer to the queue attr
 * @param type [IN]    parameter type
 * @param len [IN]       parameter length
 * @param param [IN]        pointer to parameter value
 *
 * @retval ACL_SUCCESS for success, other for failure
 *
 * @see acltdtCreateQueueAttr
 */
ACL_FUNC_VISIBILITY aclError acltdtSetQueueAttr(acltdtQueueAttr *attr,
                                                acltdtQueueAttrType type,
                                                size_t len,
                                                const void *param);

/**
 * @ingroup AscendCL
 *
 * @brief Get parameter for queue attr.
 *
 * @param attr [IN]   pointer to the queue attr
 * @param type [IN]     parameter type
 * @param len [IN]        parameter length
 * @param paramRetSize [OUT] pointer to parameter real length
 * @param param [OUT]        pointer to parameter value
 *
 * @retval ACL_SUCCESS for success, other for failure
 *
 * @see acltdtCreateQueueAttr
 */
ACL_FUNC_VISIBILITY aclError acltdtGetQueueAttr(const acltdtQueueAttr *attr,
                                                acltdtQueueAttrType type,
                                                size_t len,
                                                size_t *paramRetSize,
                                                void *param);

/**
 * @ingroup AscendCL
 * @brief Create the queue route
 *
 * @param srcId [IN]   src id of queue route
 * @param dstId [IN]   dst id of queue route
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyQueueRoute
 */
ACL_FUNC_VISIBILITY acltdtQueueRoute* acltdtCreateQueueRoute(uint32_t srcId, uint32_t dstId);

/**
 * @ingroup AscendCL
 * @brief Destroy the queue attr
 *
 * @param route [IN]  pointer to the queue route
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueRoute
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyQueueRoute(const acltdtQueueRoute *route);

/**
 * @ingroup AscendCL
 *
 * @brief Get parameter for queue route.
 *
 * @param route [IN]   pointer to the queue route
 * @param type [IN]     parameter type
 * @param len [IN]        parameter length
 * @param paramRetSize [OUT] pointer to parameter real length
 * @param param [OUT]        pointer to parameter value
 *
 * @retval ACL_SUCCESS for success, other for failure
 *
 * @see acltdtCreateQueueRoute
 */
ACL_FUNC_VISIBILITY aclError acltdtGetQueueRouteParam(const acltdtQueueRoute *route,
                                                      acltdtQueueRouteParamType type,
                                                      size_t len,
                                                      size_t *paramRetSize,
                                                      void *param);

/**
 * @ingroup AscendCL
 * @brief Create the queue route list
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyQueueRouteList
 */
ACL_FUNC_VISIBILITY acltdtQueueRouteList* acltdtCreateQueueRouteList();

/**
 * @ingroup AscendCL
 * @brief Destroy the queue route list
 *
 * @param routeList [IN]  pointer to the queue route list
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueRouteList
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyQueueRouteList(const acltdtQueueRouteList *routeList);

/**
 * @ingroup AscendCL
 * @brief add queue route to the route list
 *
 * @param routeList [IN|OUT]  pointer to the queue route list
 * @param route [IN]  pointer to the queue route
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueRouteList | acltdtCreateQueueRoute
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtAddQueueRoute(acltdtQueueRouteList *routeList, const acltdtQueueRoute *route);

/**
 * @ingroup AscendCL
 * @brief get queue route from route list
 *
 * @param routeList [IN]  pointer to the queue route list
 * @param index [IN]  index of queue route in route list
 * @param route [IN|OUT]  pointer to the queue route
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueRouteList | acltdtCreateQueueRoute
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtGetQueueRoute(const acltdtQueueRouteList *routeList,
                                                 size_t index,
                                                 acltdtQueueRoute *route);

/**
 * @ingroup AscendCL
 * @brief get queue route num from route list
 *
 * @param routeList [IN]  pointer to the queue route list
 *
 * @retval the number of queue route
 *
 */
ACL_FUNC_VISIBILITY size_t acltdtGetQueueRouteNum(const acltdtQueueRouteList *routeList);

/**
 * @ingroup AscendCL
 * @brief Create the queue route query info
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyQueueRouteQueryInfo
 */
ACL_FUNC_VISIBILITY  acltdtQueueRouteQueryInfo* acltdtCreateQueueRouteQueryInfo();

/**
 * @ingroup AscendCL
 * @brief Destroy the queue route query info
 *
 * @param info [IN]  pointer to the queue route info
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateQueueRouteQueryInfo
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyQueueRouteQueryInfo(const acltdtQueueRouteQueryInfo *info);

/**
 * @ingroup AscendCL
 * @brief Set parameter for queue route info
 *
 * @param attr [IN|OUT] pointer to the queue route info
 * @param type [IN]    parameter type
 * @param len [IN]       parameter length
 * @param param [IN]        pointer to parameter value
 *
 * @retval ACL_SUCCESS for success, other for failure
 *
 * @see acltdtCreateQueueRouteQueryInfo
 */
ACL_FUNC_VISIBILITY aclError acltdtSetQueueRouteQueryInfo(acltdtQueueRouteQueryInfo *param,
                                                          acltdtQueueRouteQueryInfoParamType type,
                                                          size_t len,
                                                          const void *value);


#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_TDT_QUEUE_H_
