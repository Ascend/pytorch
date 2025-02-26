#pragma once

#include <string>
#include "hccl/hccl.h"
#include "hccl/lcal_api.h"

namespace at_npu {
namespace lccl {

using LcclDataType = HcclDataType;
using LcclReduceOp = HcclReduceOp;
using LcclComm = LcalCommPtr;

int LcclCommInitRankLocal(int rankSize, int rank, LcclComm *comms);

int LcclCommInit(int rank, int rankSize, LcclComm *comms);

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, LcclDataType dataType, LcclReduceOp op,
                  LcclComm comm, aclrtStream stream);

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, LcclDataType dataType, LcclComm comm,
                  aclrtStream stream);

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, LcclDataType dataType, LcclReduceOp op,
                      LcclComm comm, aclrtStream stream);

int LcclBroadcast(void *buf, int64_t count, LcclDataType dataType, int root, LcclComm comm,
                  aclrtStream stream);

int LcclCommDestroy(LcclComm comm);

}
}
