#include "LcclInterface.h"

#include <unordered_map>
#include <dlfcn.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace lccl {

#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(liblcal, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(liblcal, funcName)

TORCH_NPU_REGISTER_LIBRARY(liblcal)
TORCH_NPU_LOAD_FUNC(LcalCommInitRankLocal)
TORCH_NPU_LOAD_FUNC(LcalCommInit)
TORCH_NPU_LOAD_FUNC(LcclAllReduce)
TORCH_NPU_LOAD_FUNC(LcclAllGather)
TORCH_NPU_LOAD_FUNC(LcclReduceScatter)
TORCH_NPU_LOAD_FUNC(LcclBroadcast)
TORCH_NPU_LOAD_FUNC(LcclCommDestroy)

int LcclCommInitRankLocal(int rankSize, int rank, LcclComm *comms)
{
    typedef int(*lcalCommInitRankLocal)(int, int, LcclComm *);
    static lcalCommInitRankLocal func = nullptr;
    if (func == nullptr) {
        func = (lcalCommInitRankLocal)TORCH_NPU_GET_FUNC(LcalCommInitRankLocal);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcalCommInitRankLocal", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(rankSize, rank, comms);
}

int LcclCommInit(int rank, int rankSize, LcclComm *comms)
{
    typedef int(*lcalCommInit)(int, int, LcclComm *);
    static lcalCommInit func = nullptr;
    if (func == nullptr) {
        func = (lcalCommInit)TORCH_NPU_GET_FUNC(LcalCommInit);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcalCommInit", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(rank, rankSize, comms);
}

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, LcclDataType dataType, LcclReduceOp op,
                  LcclComm comm, aclrtStream stream)
{
    typedef int(*lcclAllReduce)(void *, void *, int64_t, LcclDataType, LcclReduceOp, LcclComm, aclrtStream);
    static lcclAllReduce func = nullptr;
    if (func == nullptr) {
        func = (lcclAllReduce)TORCH_NPU_GET_FUNC(LcclAllReduce);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcclAllReduce", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, LcclDataType dataType, LcclComm comm,
                  aclrtStream stream)
{
    typedef int(*lcclAllGather)(void *, void *, int64_t, LcclDataType, LcclComm, aclrtStream);
    static lcclAllGather func = nullptr;
    if (func == nullptr) {
        func = (lcclAllGather)TORCH_NPU_GET_FUNC(LcclAllGather);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcclAllGather", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, LcclDataType dataType, LcclReduceOp op,
                      LcclComm comm, aclrtStream stream)
{
    typedef int(*lcclReduceScatter)(void *, void *, int64_t, LcclDataType, LcclReduceOp, LcclComm, aclrtStream);
    static lcclReduceScatter func = nullptr;
    if (func == nullptr) {
        func = (lcclReduceScatter)TORCH_NPU_GET_FUNC(LcclReduceScatter);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcclReduceScatter", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
}

int LcclBroadcast(void *buf, int64_t count, LcclDataType dataType, int root, LcclComm comm,
                  aclrtStream stream)
{
    typedef int(*lcclBroadcast)(void *, int64_t, LcclDataType, int, LcclComm, aclrtStream);
    static lcclBroadcast func = nullptr;
    if (func == nullptr) {
        func = (lcclBroadcast)TORCH_NPU_GET_FUNC(LcclBroadcast);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcclBroadcast", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(buf, count, dataType, root, comm, stream);
}

int LcclCommDestroy(LcclComm comm)
{
    typedef int(*lcclCommDestroy)(LcclComm);
    static lcclCommDestroy func = nullptr;
    if (func == nullptr) {
        func = (lcclCommDestroy)TORCH_NPU_GET_FUNC(LcclCommDestroy);
        if (func == nullptr) {
            TORCH_CHECK(func, "Failed to find function ", "lcclCommDestroy", PTA_ERROR(ErrCode::NOT_FOUND));
            return -1;
        }
    }
    return func(comm);
}

}
}
