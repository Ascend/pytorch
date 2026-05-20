#pragma once

#include <c10/util/CallOnce.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace c10d_npu {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libhccl, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
  TORCH_NPU_GET_FUNCTION(libhccl, funcName)

TORCH_NPU_REGISTER_LIBRARY(libhccl)
TORCH_NPU_LOAD_FUNC(HcclAlltoAllV)
TORCH_NPU_LOAD_FUNC(HcclAllGatherV)
TORCH_NPU_LOAD_FUNC(HcclReduceScatterV)
TORCH_NPU_LOAD_FUNC(HcclReduce)
TORCH_NPU_LOAD_FUNC(HcclGetCommAsyncError)
TORCH_NPU_LOAD_FUNC(HcclScatter)
TORCH_NPU_LOAD_FUNC(HcclBatchSendRecv)
TORCH_NPU_LOAD_FUNC(HcclAlltoAll)
TORCH_NPU_LOAD_FUNC(HcclCommInitRootInfoConfig)
TORCH_NPU_LOAD_FUNC(HcclGetCommConfigCapability)
TORCH_NPU_LOAD_FUNC(HcclCommInitClusterInfoConfig)
TORCH_NPU_LOAD_FUNC(HcclCreateSubCommConfig)
TORCH_NPU_LOAD_FUNC(HcclCommWorkingDevNicSet)
TORCH_NPU_LOAD_FUNC(HcclCommRegister)
TORCH_NPU_LOAD_FUNC(HcclCommDeregister)
TORCH_NPU_LOAD_FUNC(HcclCommExchangeMem)
TORCH_NPU_LOAD_FUNC(HcclGetRootInfo)
TORCH_NPU_LOAD_FUNC(HcclCommDestroy)
TORCH_NPU_LOAD_FUNC(HcclSend)
TORCH_NPU_LOAD_FUNC(HcclRecv)
TORCH_NPU_LOAD_FUNC(HcclAllReduce)
TORCH_NPU_LOAD_FUNC(HcclBroadcast)
TORCH_NPU_LOAD_FUNC(HcclAllGather)
TORCH_NPU_LOAD_FUNC(HcclReduceScatter)
TORCH_NPU_LOAD_FUNC(HcclCommInitAll)
TORCH_NPU_LOAD_FUNC(HcclCommInitRootInfo)

TORCH_NPU_REGISTER_LIBRARY(libhcomm)
TORCH_NPU_REGISTER_FUNCTION(libhcomm, HcclGroupStart)
TORCH_NPU_REGISTER_FUNCTION(libhcomm, HcclGroupEnd)

extern HcclResult hcclGetRootInfo(HcclRootInfo *rootInfo)
{
    using HcclGetRootInfoFunc = HcclResult(*)(HcclRootInfo *);
    static HcclGetRootInfoFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclGetRootInfoFunc)TORCH_NPU_GET_FUNC(HcclGetRootInfo)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclGetRootInfo", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(rootInfo);
    return ret;
}

extern HcclResult hcclCommDestroy(HcclComm comm)
{
    using HcclCommDestroyFunc = HcclResult(*)(HcclComm);
    static HcclCommDestroyFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommDestroyFunc)TORCH_NPU_GET_FUNC(HcclCommDestroy)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommDestroy", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm);
    return ret;
}

extern HcclResult hcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
    HcclComm comm, aclrtStream stream)
{
    using HcclSendFunc = HcclResult(*)(
        void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static HcclSendFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclSendFunc)TORCH_NPU_GET_FUNC(HcclSend)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclSend", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, count, dataType, destRank, comm, stream);
    return ret;
}

extern HcclResult hcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
    HcclComm comm, aclrtStream stream)
{
    using HcclRecvFunc = HcclResult(*)(
        void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static HcclRecvFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclRecvFunc)TORCH_NPU_GET_FUNC(HcclRecv)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclRecv", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(recvBuf, count, dataType, srcRank, comm, stream);
    return ret;
}

extern HcclResult hcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    using HcclCommInitAllFunc = HcclResult(*)(
        uint32_t, int32_t *, HcclComm *);
    static HcclCommInitAllFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommInitAllFunc)TORCH_NPU_GET_FUNC(HcclCommInitAll)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitAll", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(ndev, devices, comms);
    return ret;
}

extern HcclResult hcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    using HcclAllGatherFunc = HcclResult(*)(
        void *, void *, uint64_t, HcclDataType, HcclComm, aclrtStream);
    static HcclAllGatherFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAllGatherFunc)TORCH_NPU_GET_FUNC(HcclAllGather)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAllGather", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    return ret;
}

extern HcclResult hcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    using HcclAllReduceFunc = HcclResult(*)(
        void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    static HcclAllReduceFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAllReduceFunc)TORCH_NPU_GET_FUNC(HcclAllReduce)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAllReduce", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, recvBuf, count, dataType, op, comm, stream);
    return ret;
}

extern HcclResult hcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    using HcclBroadcastFunc = HcclResult(*)(
        void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static HcclBroadcastFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclBroadcastFunc)TORCH_NPU_GET_FUNC(HcclBroadcast)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclBroadcast", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(buf, count, dataType, root, comm, stream);
    return ret;
}

extern HcclResult hcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    using HcclCommInitRootInfoFunc = HcclResult(*)(
        uint32_t, const HcclRootInfo *, uint32_t, HcclComm *);
    static HcclCommInitRootInfoFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommInitRootInfoFunc)TORCH_NPU_GET_FUNC(HcclCommInitRootInfo)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitRootInfo", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(nRanks, rootInfo, rank, comm);
    return ret;
}

extern HcclResult hcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    using HcclReduceScatterFunc = HcclResult(*)(
        void *, void *, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    static HcclReduceScatterFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclReduceScatterFunc)TORCH_NPU_GET_FUNC(HcclReduceScatter);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclReduceScatter", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    return ret;
}

extern HcclResult hcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    using HcclAlltoAllVFunc = HcclResult(*)(
        const void *, const void *, const void *, HcclDataType,
        const void *, const void *, const void *, HcclDataType,
        HcclComm, aclrtStream);
    static HcclAlltoAllVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAlltoAllVFunc)TORCH_NPU_GET_FUNC(HcclAlltoAllV);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAlltoAllV", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, sendCounts, sdispls, sendType,
        recvBuf, recvCounts, rdispls, recvType, comm, stream);
    return ret;
}

extern HcclResult hcclAllGatherV(const void *sendBuf, uint64_t sendCount,
    const void *recvBuf, const void *recvCounts, const void *rdispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    using HcclAllGatherVFunc = HcclResult(*)(
        const void *, uint64_t,
        const void *, const void *, const void *,
        HcclDataType, HcclComm, aclrtStream);
    static HcclAllGatherVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAllGatherVFunc)TORCH_NPU_GET_FUNC(HcclAllGatherV);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAllGatherV", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType, comm, stream);
    return ret;
}

extern HcclResult hcclReduceScatterV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    const void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    using HcclReduceScatterVFunc = HcclResult(*)(
        const void *, const void *, const void *,
        const void *, uint64_t,
        HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    static HcclReduceScatterVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclReduceScatterVFunc)TORCH_NPU_GET_FUNC(HcclReduceScatterV);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclReduceScatterV", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, sendCounts, sdispls, recvBuf, recvCount, dataType, op, comm, stream);
    return ret;
}

extern HcclResult hcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType sendType,
    HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
{
    using HcclReduceVFunc = HcclResult(*)(
        void *, void *, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
    static HcclReduceVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclReduceVFunc)TORCH_NPU_GET_FUNC(HcclReduce);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclReduce", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, recvBuf, count, sendType, op, root, comm, stream);
    return ret;
}

HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError)
{
    using HcclGetCommAsyncErrorVFunc = HcclResult(*)(HcclComm, HcclResult*);
    static HcclGetCommAsyncErrorVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclGetCommAsyncErrorVFunc)TORCH_NPU_GET_FUNC(HcclGetCommAsyncError);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclGetCommAsyncError", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, asyncError);
    return ret;
}

HcclResult hcclScatter(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    using HcclScatterVFunc = HcclResult(*)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static HcclScatterVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclScatterVFunc)TORCH_NPU_GET_FUNC(HcclScatter);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclScatter", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, recvBuf, count, dataType, root, comm, stream);
    return ret;
}

HcclResult hcclBatchIsendIrecv(void* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    using HcclBatchIsendIrecvVFunc = HcclResult(*)(
        void *, uint32_t, HcclComm, aclrtStream);
    static HcclBatchIsendIrecvVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclBatchIsendIrecvVFunc)TORCH_NPU_GET_FUNC(HcclBatchSendRecv);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclBatchSendRecv", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendRecvInfo, itemNum, comm, stream);
    return ret;
}

HcclResult hcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
    const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
    HcclComm comm, aclrtStream stream)
{
    using HcclAlltoAllFunc = HcclResult(*)(
        const void *, uint64_t, HcclDataType,
        const void *, uint64_t, HcclDataType,
        HcclComm, aclrtStream);
    static HcclAlltoAllFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAlltoAllFunc)TORCH_NPU_GET_FUNC(HcclAlltoAll);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAlltoAll", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(sendBuf, sendCount, sendType,
                    recvBuf, recvCount, recvType, comm, stream);
    return ret;
}

bool hcclCommInitRootInfoConfigExist()
{
    static c10::once_flag flag;
    static bool exist = false;
    c10::call_once(flag, [&]() {
        auto func = TORCH_NPU_GET_FUNC(HcclCommInitRootInfoConfig)
        if (func != nullptr) {
            exist = true;
        }
    });
    return exist;
}

bool hcclAllGatherVExist()
{
    static c10::once_flag flag;
    static bool exist = false;
    c10::call_once(flag, [&]() {
        auto func = TORCH_NPU_GET_FUNC(HcclAllGatherV)
        if (func != nullptr &&
            c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend310P1 &&
            c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) {
            exist = true;
        }
    });
    return exist;
}

bool hcclReduceScatterVExist()
{
    static c10::once_flag flag;
    static bool exist = false;
    c10::call_once(flag, [&]() {
        auto func = TORCH_NPU_GET_FUNC(HcclReduceScatterV)
        if (func != nullptr &&
            ((c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend310P1 &&
              c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
              c10_npu::GetSocVersion() == c10_npu::SocVersion::Ascend950)) {
            exist = true;
        }
    });
    return exist;
}

HcclResult hcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclCommConfig* config, HcclComm *comm)
{
    using HcclCommInitRootInfoConfigFunc = HcclResult(*)(
        uint32_t, const HcclRootInfo *, uint32_t, HcclCommConfig*, HcclComm *);
    static HcclCommInitRootInfoConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommInitRootInfoConfigFunc)TORCH_NPU_GET_FUNC(HcclCommInitRootInfoConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitRootInfoConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(nRanks, rootInfo, rank, config, comm);
    return ret;
}

bool isHcclFeatureSupported(HcclCommConfigCapability configParameter)
{
    using HcclGetCommConfigCapabilityFunc = uint32_t(*)();
    static HcclGetCommConfigCapabilityFunc func = (HcclGetCommConfigCapabilityFunc) TORCH_NPU_GET_FUNC(
            HcclGetCommConfigCapability);
    if (func == nullptr) {
        return false;
    }
    return configParameter < func();
}

bool hcclCommInitClusterInfoConfigExist()
{
    const static bool isClusterInitExist = []() -> bool {
        auto func = TORCH_NPU_GET_FUNC(HcclCommInitClusterInfoConfig)
        return func != nullptr;
    }();
    return isClusterInitExist;
}

HcclResult hcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)
{
    using HcclCommInitClusterInfoConfigFunc = HcclResult(*)(const char *, uint32_t, HcclCommConfig *, HcclComm *);
    static HcclCommInitClusterInfoConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommInitClusterInfoConfigFunc)TORCH_NPU_GET_FUNC(HcclCommInitClusterInfoConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitClusterInfoConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(clusterInfo, rank, config, comm);
    return ret;
}

bool hcclCreateSubCommConfigExist()
{
    const static bool isCreateSubCommExist = []() -> bool {
        auto func = TORCH_NPU_GET_FUNC(HcclCreateSubCommConfig)
        return func != nullptr;
    }();
    return isCreateSubCommExist;
}

HcclResult hcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm *subComm)
{
    using HcclCreateSubCommConfigFunc = HcclResult(*)(HcclComm *, uint32_t, uint32_t *, uint64_t, uint32_t, HcclCommConfig *, HcclComm *);
    static HcclCreateSubCommConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCreateSubCommConfigFunc)TORCH_NPU_GET_FUNC(HcclCreateSubCommConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCreateSubCommConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
    return ret;
}

bool hcclCommWorkingDevNicSetExist()
{
    const static bool isHcclCommWorkingDevNicSetExist = []() -> bool {
        auto func = TORCH_NPU_GET_FUNC(HcclCommWorkingDevNicSet)
        return func != nullptr;
    }();
    return isHcclCommWorkingDevNicSetExist;
}

HcclResult hcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks)
{
    using HcclCommWorkingDevNicSetFunc = HcclResult(*)(HcclComm, uint32_t *, bool *, uint32_t);
    static HcclCommWorkingDevNicSetFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommWorkingDevNicSetFunc)TORCH_NPU_GET_FUNC(HcclCommWorkingDevNicSet)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommWorkingDevNicSet", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, ranks, useBackup, nRanks);
    return ret;
}

HcclResult hcclCommRegister(HcclComm comm, void *addr, uint64_t size, void **handle, uint32_t flag)
{
    using HcclCommRegisterFunc = HcclResult(*)(HcclComm, void *, uint64_t, void **, uint32_t);
    static HcclCommRegisterFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommRegisterFunc)TORCH_NPU_GET_FUNC(HcclCommRegister)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommRegister", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, addr, size, handle, flag);
    return ret;
}

HcclResult hcclCommDeregister(HcclComm comm, void *handle)
{
    using HcclCommDeregisterFunc = HcclResult(*)(HcclComm, void *);
    static HcclCommDeregisterFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommDeregisterFunc)TORCH_NPU_GET_FUNC(HcclCommDeregister)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommDeregister", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, handle);
    return ret;
}

HcclResult hcclCommExchangeMem(HcclComm comm, void *windowHandle, uint32_t *peerRanks, uint32_t peerRankNum)
{
    using HcclCommExchangeMemFunc = HcclResult(*)(HcclComm, void *, uint32_t *, uint32_t);
    static HcclCommExchangeMemFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommExchangeMemFunc)TORCH_NPU_GET_FUNC(HcclCommExchangeMem)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommExchangeMem", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, windowHandle, peerRanks, peerRankNum);
    return ret;
}

HcclResult hcclGroupStart()
{
    using hcclGroupStartFunc = HcclResult(*)();
    static hcclGroupStartFunc func = nullptr;
    if (func == nullptr) {
        func = (hcclGroupStartFunc)TORCH_NPU_GET_FUNCTION(libhcomm, HcclGroupStart)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclGroupStart", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func();
    return ret;
}

HcclResult hcclGroupEnd()
{
    using hcclGroupEndFunc = HcclResult(*)();
    static hcclGroupEndFunc func = nullptr;
    if (func == nullptr) {
        func = (hcclGroupEndFunc)TORCH_NPU_GET_FUNCTION(libhcomm, HcclGroupEnd)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclGroupEnd", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func();
    return ret;
}
} // namespace c10d_npu
