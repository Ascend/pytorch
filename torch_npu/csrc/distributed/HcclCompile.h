#include <c10/util/CallOnce.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace c10d_npu {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclAlltoAllV)
LOAD_FUNCTION(HcclReduce)
LOAD_FUNCTION(HcclGetCommAsyncError)
LOAD_FUNCTION(HcclScatter)
LOAD_FUNCTION(HcclBatchSendRecv)
LOAD_FUNCTION(HcclAlltoAll)
LOAD_FUNCTION(HcclCommInitRootInfoConfig)
LOAD_FUNCTION(HcclGetCommConfigCapability)
LOAD_FUNCTION(HcclCommInitClusterInfoConfig)
LOAD_FUNCTION(HcclCreateSubCommConfig)

extern HcclResult hcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    HcclDataType recvType, HcclComm comm, aclrtStream stream) {
  typedef HcclResult(*HcclAlltoAllVFunc)(
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

extern HcclResult hcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType sendType,
    HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream) {
  typedef HcclResult(*HcclReduceVFunc)(
      void *, void *, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
  static HcclReduceVFunc func = nullptr;
  if (func == nullptr) {
    func = (HcclReduceVFunc)GET_FUNC(HcclReduce);
  }
  TORCH_CHECK(func, "Failed to find function ", "HcclReduce", DIST_ERROR(ErrCode::NOT_FOUND));
  auto ret = func(sendBuf, recvBuf, count, sendType, op, root, comm, stream);
  return ret;
}

HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError) {
    typedef HcclResult(*HcclGetCommAsyncErrorVFunc)(HcclComm, HcclResult*);
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
    typedef HcclResult(*HcclScatterVFunc)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
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
    typedef HcclResult(*HcclBatchIsendIrecvVFunc)(
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
    typedef HcclResult(*HcclAlltoAllFunc)(
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

HcclResult hcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclCommConfig* config, HcclComm *comm)
{
    typedef HcclResult(*HcclCommInitRootInfoConfigFunc)(
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
    typedef uint32_t(*HcclGetCommConfigCapabilityFunc)();
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
    typedef HcclResult(*HcclCommInitClusterInfoConfigFunc)(const char *, uint32_t, HcclCommConfig *, HcclComm *);
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
    typedef HcclResult(*HcclCreateSubCommConfigFunc)(HcclComm *, uint32_t, uint32_t *, uint64_t, uint32_t, HcclCommConfig *, HcclComm *);
    static HcclCreateSubCommConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclCreateSubCommConfigFunc)GET_FUNC(HcclCreateSubCommConfig)
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclCreateSubCommConfig", DIST_ERROR(ErrCode::NOT_FOUND));
    auto ret = func(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
    return ret;
}
} // namespace c10d_npu
