#pragma once

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
    void Copy(CopyParas& other);
    static std::map<int64_t, std::string> COPY_PARAS_MAP;
};

enum EventAllocatorType {
    HOST_ALLOCATOR_EVENT = 1,
    NPU_ALLOCATOR_EVENT = 2,
    RESERVED = -1,
};

struct EventParas {
    explicit EventParas(aclrtEvent aclEvent, EventAllocatorType allocatorType)
        : event(aclEvent), eventAllocatorType(allocatorType) {}
    EventParas() = default;
    aclrtEvent event = nullptr;
    void Copy(EventParas& other);
    EventAllocatorType eventAllocatorType = RESERVED;
    static std::map<int64_t, std::string> EVENT_PARAS_MAP;
};

enum QueueParamType {
    COMPILE_AND_EXECUTE = 1,
    ASYNC_MEMCPY = 2,
    RECORD_EVENT = 3,
    WAIT_EVENT = 4,
    LAZY_DESTROY_EVENT = 5,
    RESET_EVENT = 6,
    EXECUTE_OPAPI = 7,
    EXECUTE_OPAPI_V2 = 8,
};

struct QueueParas {
    QueueParas(QueueParamType type, size_t len, void *val) : paramType(type), paramLen(len), paramVal(val) {}
    aclrtStream paramStream = nullptr;
    QueueParamType paramType = COMPILE_AND_EXECUTE;
    size_t paramLen = 0;
    void* paramVal = nullptr;
    static std::atomic<uint64_t> g_correlation_id;
    uint64_t correlation_id = 0;
};

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream);

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream);

aclError LaunchLazyDestroyEventTask(aclrtEvent event, c10::DeviceIndex device_index);

} // namespace queue
} // namespace c10_npu
