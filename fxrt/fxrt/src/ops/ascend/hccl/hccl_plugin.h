#ifndef OPS_ASCEND_HCCL_PLUGIN_H
#define OPS_ASCEND_HCCL_PLUGIN_H

#include <string>
#include <memory>
#include <map>
#include <functional>

#include "hccl/hccl.h"

#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

using OptionsType = std::map<std::string, std::string>;
using HExecCallBack = std::function<void(HcclResult)>;

ORIGIN_METHOD(HcclBroadcast, HcclResult, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclAllReduce, HcclResult, void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
ORIGIN_METHOD(
    HcclReduce,
    HcclResult,
    void*,
    void*,
    uint64_t,
    HcclDataType,
    HcclReduceOp,
    uint32_t,
    HcclComm,
    aclrtStream);
ORIGIN_METHOD(HcclScatter, HcclResult, void*, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclReduceScatter, HcclResult, void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclAllGather, HcclResult, void*, void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclSend, HcclResult, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclRecv, HcclResult, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(
    HcclAlltoAllV,
    HcclResult,
    const void*,
    const void*,
    const void*,
    HcclDataType,
    const void*,
    const void*,
    const void*,
    HcclDataType,
    HcclComm,
    aclrtStream);
ORIGIN_METHOD(
    HcclAllGatherV,
    HcclResult,
    void*,
    uint64_t,
    void*,
    const void*,
    const void*,
    HcclDataType,
    HcclComm,
    aclrtStream);
ORIGIN_METHOD(
    HcclReduceScatterV,
    HcclResult,
    void*,
    const void*,
    const void*,
    void*,
    uint64_t,
    HcclDataType,
    HcclReduceOp,
    HcclComm,
    aclrtStream);

ORIGIN_METHOD(
    HcclAlltoAll,
    HcclResult,
    const void*,
    uint64_t,
    HcclDataType,
    const void*,
    uint64_t,
    HcclDataType,
    HcclComm,
    aclrtStream);
ORIGIN_METHOD(HcclBarrier, HcclResult, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclBatchSendRecv, HcclResult, HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream);
ORIGIN_METHOD(HcclCommResume, HcclResult, HcclComm)

ORIGIN_METHOD(HcclGetCommAsyncError, HcclResult, HcclComm, HcclResult*);
ORIGIN_METHOD(HcclGetErrorString, const char*, HcclResult);
ORIGIN_METHOD(HcclGetCommConfigCapability, uint32_t);
ORIGIN_METHOD(HcclSetGlobalCommInfo, HcclResult, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
ORIGIN_METHOD(HcclCommInitClusterInfo, HcclResult, const char*, uint32_t, HcclComm*);
ORIGIN_METHOD(HcclCommInitClusterInfoConfig, HcclResult, const char*, uint32_t, HcclCommConfig*, HcclComm*);
ORIGIN_METHOD(
    HcclCommInitRootInfoConfig,
    HcclResult,
    uint32_t,
    const HcclRootInfo*,
    uint32_t,
    const HcclCommConfig*,
    HcclComm*);
ORIGIN_METHOD(
    HcclCreateSubCommConfig,
    HcclResult,
    HcclComm*,
    uint32_t,
    uint32_t*,
    uint64_t,
    uint32_t,
    HcclCommConfig*,
    HcclComm*)
ORIGIN_METHOD(HcclCommDestroy, HcclResult, HcclComm);
ORIGIN_METHOD(HcclGetRankId, HcclResult, void*, uint32_t*);
ORIGIN_METHOD(HcclGetRankSize, HcclResult, void*, uint32_t*);
ORIGIN_METHOD(HcclGetCommName, HcclResult, HcclComm, char*)
ORIGIN_METHOD(HcomGetLocalRankId, HcclResult, const char*, uint32_t*);
ORIGIN_METHOD(HcomGetLocalRankSize, HcclResult, const char*, uint32_t*);
ORIGIN_METHOD(HcomGetWorldRankFromGroupRank, HcclResult, const char*, uint32_t, uint32_t*);
ORIGIN_METHOD(HcomGetGroupRankFromWorldRank, HcclResult, uint32_t, const char*, uint32_t*);
ORIGIN_METHOD(HcclCommWorkingDevNicSet, HcclResult, HcclComm, uint32_t*, bool*, uint32_t);

ORIGIN_METHOD(HcomCreateGroup, HcclResult, const char*, uint32_t, uint32_t*);
ORIGIN_METHOD(HcomDestroyGroup, HcclResult, const char*);
ORIGIN_METHOD(HcomGetRankId, HcclResult, const char*, uint32_t*);
ORIGIN_METHOD(HcomGetRankSize, HcclResult, const char*, uint32_t*);
ORIGIN_METHOD(HcomExecInitialize, HcclResult);
ORIGIN_METHOD(HcomExecFinalize, HcclResult);
ORIGIN_METHOD(HcomDestroy, HcclResult);
#endif // OPS_ASCEND_HCCL_PLUGIN_H
