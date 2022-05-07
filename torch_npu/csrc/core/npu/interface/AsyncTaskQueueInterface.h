#ifndef __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__
#define __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__

#include "c10/core/Storage.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace c10_npu {
namespace queue {
struct CopyParas {
  void *dst = nullptr;
  size_t dstLen = 0;
  void *src = nullptr;
  size_t srcLen = 0;
  aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_HOST;
  c10::SmallVector<c10::Storage, 1> pinMem;
  void Copy(CopyParas& other);
};

enum EventAllocatorType {
  HOST_ALLOCATOR_EVENT = 1,
  NPU_ALLOCATOR_EVENT = 2,
  RESERVED = -1,
};

struct EventParas {
  explicit EventParas(aclrtEvent aclEvent, EventAllocatorType allocatorType) :
      event(aclEvent), eventAllocatorType(allocatorType) {}
  aclrtEvent event = nullptr;
  void Copy(EventParas& other);
  EventAllocatorType eventAllocatorType = RESERVED;
};

enum QueueParamType {
  COMPILE_AND_EXECUTE = 1,
  ASYNC_MEMCPY = 2,
  ASYNC_MEMCPY_EX = 3,
  RECORD_EVENT = 4,
  WAIT_EVENT = 5,
  LAZY_DESTROY_EVENT = 6,
};

struct QueueParas {
  QueueParas(QueueParamType type, size_t len, void *val) : paramType(type), paramLen(len), paramVal(val) {}
  aclrtStream paramStream = nullptr;
  QueueParamType paramType = COMPILE_AND_EXECUTE;
  size_t paramLen = 0;
  void* paramVal = nullptr;
};

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
    c10::Storage& st, bool isPinMem);

aclError HostAllocatorLaunchRecordEventTask(aclrtEvent event,
                                            c10_npu::NPUStream npuStream,
                                            c10::SmallVector<c10::Storage, c10_npu::N>& needClearVec);

aclError NpuAllocatorLaunchRecordEventTask(aclrtEvent event,
                                           c10_npu::NPUStream npuStream);

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream);

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream);

aclError LaunchLazyDestroyEventTask(aclrtEvent event);
} // namespace queue
} // namespace c1c10_npu0

#endif // __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__