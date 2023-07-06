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
  TORCH_CHECK(func, "Failed to find function ", "HcclAlltoAllV");
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
  TORCH_CHECK(func, "Failed to find function ", "HcclReduce");
  auto ret = func(sendBuf, recvBuf, count, sendType, op, root, comm, stream);
  return ret;
}
} // namespace c10d_npu
