#include "hccl.h"

hcclResult_t HcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank) {return HCCL_SUCCESS;}
hcclResult_t HcclGetUniqueId(hcclUniqueId* id) {return HCCL_SUCCESS;}
hcclResult_t HcclAllReduce(void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t HcclBroadcast(void *ptr, u64 count, HcclDataType dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t HcclCommDestroy(hcclComm_t comm) {return HCCL_SUCCESS;}

hcclResult_t HcclReduceScatter(void *sendBuf, void *recvBuf, u64 recvCount, HcclDataType dataType,
                               HcclReduceOp op, HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclCommInitRootInfo(u32 nRanks, const HcclRootInfo *rootInfo,
                                  u32 rank, HcclComm *comm) {return HCCL_SUCCESS;}
hcclResult_t HcclCommInitRootInfoConfig(u32 nRanks, const HcclRootInfo *rootInfo,
                                        u32 rank, HcclCommConfig config, HcclComm *comm) {return HCCL_SUCCESS;}
hcclResult_t HcclCommInitClusterInfoConfig(const char *clusterInfo, u32 rank, HcclCommConfig *config,
    HcclComm *comm) {return HCCL_SUCCESS;}
hcclResult_t HcclCreateSubCommConfig(HcclComm *comm, u32 rankNum, u32 *rankIds, u64 subCommId, u32 subCommRankId,
    HcclCommConfig *config, HcclComm *subComm) {return HCCL_SUCCESS;}
hcclResult_t HcclGetCommName(HcclComm commHandle, char* commName) {return HCCL_SUCCESS;}

hcclResult_t HcclAllGather(void *sendBuf, void *recvBuf, u64 sendCount, HcclDataType dataType,
                           HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclRecv(void *recvBuf, u64 count, HcclDataType dataType, u32 srcRank,
                      HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclSend(void *sendBuf, u64 count, HcclDataType dataType, u32 destRank,
                      HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclGetRootInfo(HcclRootInfo *rootInfo) {return HCCL_SUCCESS;}
hcclResult_t HcclGetCommAsyncError(hcclComm_t comm, hcclResult_t* asyncError) {return HCCL_SUCCESS;}
hcclResult_t HcclScatter(void *sendBuf, void *recvBuf, u64 count, HcclDataType dataType, u32 root, HcclComm comm,
    aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclBatchSendRecv(HcclSendRecvItemDef* sendRecvInfo, u32 itemNum, hcclComm_t comm,
    aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclCommInitAll(u32 ndev, s32 *devices, hcclComm_t *comms) {return HCCL_SUCCESS;}
hcclResult_t HcclCommResume(hcclComm_t comm) {return HCCL_SUCCESS;}
hcclResult_t HcclCommWorkingDevNicSet(HcclComm comm, u32 *ranks, bool *useBackup, u32 nRanks){return HCCL_SUCCESS;}
