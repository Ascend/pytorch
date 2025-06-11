#pragma once

#include <c10/util/CallOnce.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace c10d_npu {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclAlltoAllV)
LOAD_FUNCTION(HcclAllGatherV)
LOAD_FUNCTION(HcclReduceScatterV)
LOAD_FUNCTION(HcclReduce)
LOAD_FUNCTION(HcclGetCommAsyncError)
LOAD_FUNCTION(HcclScatter)
LOAD_FUNCTION(HcclBatchSendRecv)
LOAD_FUNCTION(HcclAlltoAll)
LOAD_FUNCTION(HcclCommInitRootInfoConfig)
LOAD_FUNCTION(HcclGetCommConfigCapability)
LOAD_FUNCTION(HcclCommInitClusterInfoConfig)
LOAD_FUNCTION(HcclCreateSubCommConfig)
LOAD_FUNCTION(HcclCommWorkingDevNicSet)


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
        func = (HcclAlltoAllVFunc)GET_FUNC(HcclAlltoAllV);
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
        func = (HcclAllGatherVFunc)GET_FUNC(HcclAllGatherV);
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
        func = (HcclReduceScatterVFunc)GET_FUNC(HcclReduceScatterV);
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
        func = (HcclReduceVFunc)GET_FUNC(HcclReduce);
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
        func = (HcclGetCommAsyncErrorVFunc)GET_FUNC(HcclGetCommAsyncError);
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
        func = (HcclScatterVFunc)GET_FUNC(HcclScatter);
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
        func = (HcclBatchIsendIrecvVFunc)GET_FUNC(HcclBatchSendRecv);
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
        func = (HcclAlltoAllFunc)GET_FUNC(HcclAlltoAll);
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
        auto func = GET_FUNC(HcclCommInitRootInfoConfig)
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
        auto func = GET_FUNC(HcclAllGatherV)
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
        auto func = GET_FUNC(HcclReduceScatterV)
        if (func != nullptr &&
            c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend310P1 &&
            c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) {
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
        func = (HcclCommInitRootInfoConfigFunc)GET_FUNC(HcclCommInitRootInfoConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitRootInfoConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(nRanks, rootInfo, rank, config, comm);
    return ret;
}

bool isHcclFeatureSupported(HcclCommConfigCapability configParameter)
{
    using HcclGetCommConfigCapabilityFunc = uint32_t(*)();
    static HcclGetCommConfigCapabilityFunc func = (HcclGetCommConfigCapabilityFunc) GET_FUNC(
            HcclGetCommConfigCapability);
    if (func == nullptr) {
        return false;
    }
    return configParameter < func();
}

bool hcclCommInitClusterInfoConfigExist()
{
    const static bool isClusterInitExist = []() -> bool {
        auto func = GET_FUNC(HcclCommInitClusterInfoConfig)
        return func != nullptr;
    }();
    return isClusterInitExist;
}

HcclResult hcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)
{
    using HcclCommInitClusterInfoConfigFunc = HcclResult(*)(const char *, uint32_t, HcclCommConfig *, HcclComm *);
    static HcclCommInitClusterInfoConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommInitClusterInfoConfigFunc)GET_FUNC(HcclCommInitClusterInfoConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommInitClusterInfoConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(clusterInfo, rank, config, comm);
    return ret;
}

bool hcclCreateSubCommConfigExist()
{
    const static bool isCreateSubCommExist = []() -> bool {
        auto func = GET_FUNC(HcclCreateSubCommConfig)
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
        func = (HcclCreateSubCommConfigFunc)GET_FUNC(HcclCreateSubCommConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCreateSubCommConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
    return ret;
}

bool hcclCommWorkingDevNicSetExist()
{
    const static bool isHcclCommWorkingDevNicSetExist = []() -> bool {
        auto func = GET_FUNC(HcclCommWorkingDevNicSet)
        return func != nullptr;
    }();
    return isHcclCommWorkingDevNicSetExist;
}

HcclResult hcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks)
{
    using HcclCommWorkingDevNicSetFunc = HcclResult(*)(HcclComm, uint32_t *, bool *, uint32_t);
    static HcclCommWorkingDevNicSetFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCommWorkingDevNicSetFunc)GET_FUNC(HcclCommWorkingDevNicSet)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCommWorkingDevNicSet", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, ranks, useBackup, nRanks);
    return ret;
}
} // namespace c10d_npu
