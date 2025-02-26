#ifndef LCAL_API_H
#define LCAL_API_H

#include "hccl/hccl.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *LcalCommPtr;

int LcalCommInitRankLocal(int rankSize, int rank, LcalCommPtr *comms);

int LcalCommInit(int rank, int rankSize, LcalCommPtr *comms);

int LcalCommInitAll(uint32_t ndev, int32_t *devices, LcalCommPtr *comms);

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, HcclDataType dataType, HcclReduceOp op,
                  LcalCommPtr comm, aclrtStream stream);

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, HcclDataType dataType, LcalCommPtr comm,
                  aclrtStream stream);

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                      LcalCommPtr comm, aclrtStream stream);

int LcclBroadcast(void *buf, int64_t count, HcclDataType dataType, int root, LcalCommPtr comm, aclrtStream stream);

int LcclCommDestroy(LcalCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // LCAL_API_H
