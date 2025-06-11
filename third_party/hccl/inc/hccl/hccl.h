/**
 * @file hccl.h
 * @brief HCCL API
 */

#ifndef HCCL_H_
#define HCCL_H_

#include "third_party/hccl/inc/hccl/hccl_types.h"
#include "third_party/acl/inc/acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Initialize HCCL.
 *
 * @param clusterInfo A string identifying the cluster info file path, include file name.
 * @param rank A integer identifying the identify for the rank.
 * @param comm A pointer identifying the initialized communication resource.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm);

/**
 * @brief Get hccl root info.
 *
 * @param rootInfo A pointer identifying the hccl root info.
 * @return HcclResult
 */
extern HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);

/**
 * @brief Initialize HCCL with root info.
 *
 * @param nRanks A integer identifying the rank size of the cluster.
 * @param rootInfo A struct identifying the hccl root info.
 * @param rank A integer identifying the identify for the rank.
 * @param comm A pointer identifying the initialized communication resource.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);

/**
 * @brief get hccl comm name
 *
 * @param commHandle [IN]    query hccl commHandle
 * @param commName [OUT]     hccl come name
 *
 * @return HcclResult
 */
extern HcclResult HcclGetCommName(HcclComm commHandle, char* commName);

/**
 * @brief AllReduce operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, float16, float32.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
extern HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, 
HcclReduceOp op, HcclComm comm, aclrtStream stream);

/**
 * @brief Broadcast operator.
 *
 * @param buf A pointer identifying the data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param root An integer(u32) identifying the the root rank in the operator.
 * @param comm A pointer identifying the communication resource based on
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
extern HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, 
aclrtStream stream);

/**
 * @brief ReduceScatter operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCount An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
extern HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, 
HcclReduceOp op, HcclComm comm, aclrtStream stream);

/**
 * @brief AllGather operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param sendCount An integer(u64) identifying the number of the input data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
extern HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, 
HcclComm comm, aclrtStream stream);

/**
 * @brief Destroy HCCL comm
 *
 * @param comm A pointer identifying the communication resource targetting
 * @return HcclResult
 * @see HcclCommInitClusterInfo()
 */
extern HcclResult HcclCommDestroy(HcclComm comm);

extern HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);

extern HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, 
    HcclComm comm, aclrtStream stream);

extern HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, 
    HcclComm comm, aclrtStream stream);

/**
 * @brief AlltoAllV operator.
 *
 * @param sendBuff A pointer identifying the input data address of the operator.
 * @param sendCounts Integer array, where entry i specifies the number of elements to send to rank i.
 * @param sdispls Integer array, where entry i specifies the displacement (offset from sendbuf, in units of sendtype) from which to send data to rank i.
 * @param sendType Datatype of send buffer elements, must be one of the following types: int8, int32, int64, uint64, float16, float32.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCounts Integer array, where entry j specifies the number of elements to receive from rank j.
 * @param rdispls Integer array, where entry j specifies the displacement (offset from recvbuf, in units of recvtype) to which data from rank j should be written.
 * @param recvType Datatype of receive buffer elements, must be one of the following types: int8, int32, int64, uint64, float16, float32.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, 
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, 
    HcclDataType recvType, HcclComm comm, aclrtStream stream);

extern HcclResult HcclAllGatherV(const void *sendBuf, uint64_t sendCount,
    const void *recvBuf, const void *recvCounts, const void *rdispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream);

extern HcclResult HcclReduceScatterV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    const void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream);

extern HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType sendType,
    HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream);

extern HcclResult HcclBatchSendRecv(HcclSendRecvItemDef *sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief set hccl config option value
 *
 * @param config [IN]      hccl set config type
 * @param configValue [IN]   hccl set config value
 *
 * @retval HCCL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);

extern HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream);

extern HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
    const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream);

extern HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms);

extern HcclResult HcclCommResume(HcclComm comm);

extern HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks);

/**
 * @brief Initialize the comm configuration.
 * @param config Pointer to the comm configuration that needs to be initialized.
*/
inline void HcclCommConfigInit(HcclCommConfig *config)
{
    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;

    configInfo_t *info = (configInfo_t *)config;

    info->size = sizeof(HcclCommConfig);
    info->magicWord = HCCL_COMM_CONFIG_MAGIC_WORD;
    info->version = HCCL_COMM_CONFIG_VERSION;
    info->reserved = 0;

    config->hcclBufferSize = HCCL_COMM_DEFAULT_BUFFSIZE;
    config->hcclDeterministic = HCCL_COMM_DEFAULT_DETERMINISTIC;
    config->hcclCommName[0] = '\0';
    config->hcclUdi[0] = '\0';
    config->hcclRdmaTrafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET;
    config->hcclRdmaServiceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET;
    config->hcclOpExpansionMode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE;
}

/**
 * @brief Get a number that represents the capability of comm configuration.
*/
extern uint32_t HcclGetCommConfigCapability();

/**
 * @brief Set the virtual memory range to HCCL communicator
 * @param comm A pointer identifying the communication resource based on.
 * @param baseVirPtr The base address of memory range
 * @param size The size of memory range
 * @param alignment Memory range alignment, now only support 0
 * @param flags The flag of this memory range, now only support 0
*/
extern HcclResult HcclCommSetMemoryRange(HcclComm comm, void *baseVirPtr, size_t size, size_t alignment, uint64_t flags);

/**
 * @brief Unset the virtual memory range to HCCL communicator
 * @param comm A pointer identifying the communication resource based on.
 * @param baseVirPtr The base address of memory range set by @ref HcclCommSetMemoryRange().
*/
extern HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr);

/**
 * @brief Activate memory by physical memory handle.
 * @param comm A pointer identifying the communication resource based on.
 * @param virPtr The virtual address memory range in @ref HcclCommSetMemoryRange()
 * @param size The length of activate memory
 * @param offset The offset of physical memory, now only support 0
 * @param handle The physical memory handle
 * @param flags The flag of physical memory, now only support 0
*/
extern HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr, size_t size, size_t offset,
                                             aclrtDrvMemHandle handle, uint64_t flags);

/**
 * @brief Deactivate memory.
 * @param comm A pointer identifying the communication resource based on.
 * @param virPtr The virtual address of activate memory by @ref HcclCommActivateCommMemory().
*/
extern HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void *virPtr);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_H_
