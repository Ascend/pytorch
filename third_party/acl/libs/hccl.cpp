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
hcclResult_t HcclAllGather(void *sendBuf, void *recvBuf, u64 sendCount, HcclDataType dataType,
                           HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclRecv(void *recvBuf, u64 count, HcclDataType dataType, u32 srcRank,
                      HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclSend(void *sendBuf, u64 count, HcclDataType dataType, u32 destRank,
                      HcclComm comm, aclrtStream stream) {return HCCL_SUCCESS;}
hcclResult_t HcclGetRootInfo(HcclRootInfo *rootInfo) {return HCCL_SUCCESS;}
