#include "AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include <ATen/record_function.h>
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif
namespace c10_npu {
namespace queue {
std::atomic<uint64_t> QueueParas::g_correlation_id{ 0 };
static std::string  batch_copy_paras = "acl_memcpy_batch_async";
std::map<int64_t, std::string> CopyParas::COPY_PARAS_MAP{
    { ACL_MEMCPY_HOST_TO_HOST, "acl_memcpy_host_to_host" },
    { ACL_MEMCPY_HOST_TO_DEVICE, "acl_memcpy_host_to_device" },
    { ACL_MEMCPY_DEVICE_TO_HOST, "acl_memcpy_device_to_host" },
    { ACL_MEMCPY_DEVICE_TO_DEVICE, "acl_memcpy_device_to_device" },
};
std::map<int64_t, std::string> EventParas::EVENT_PARAS_MAP{
    { RECORD_EVENT, "record_event" },
    { WAIT_EVENT, "wait_event" },
    { LAZY_DESTROY_EVENT, "destroy_event" },
    { WRITE_VALUE, "record_event" },
    { WAIT_VALUE, "wait_event" },
};
void CopyParas::Copy(CopyParas &other)
{
    this->dst = other.dst;
    this->dstLen = other.dstLen;
    this->src = other.src;
    this->srcLen = other.srcLen;
    this->kind = other.kind;
}

void EventParas::Copy(EventParas &other)
{
    this->event = other.event;
    this->eventAllocatorType = other.eventAllocatorType;
}

class AsyncCopyTask {
public:
    AsyncCopyTask(void *dst, size_t dstLen, void *src, size_t srcLen, aclrtMemcpyKind kind);
    ~AsyncCopyTask() = default;
    void LaunchCopyTask();

private:
    CopyParas copyParam_;
};

class AsyncBatchCopyTask {
public:
    ~AsyncBatchCopyTask() = default;

    void LaunchBatchCopyTask(void **dsts, size_t *dstLens, void **srcs, size_t *srcLens, size_t numBatches,
                             aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, aclrtStream stream);
    QueueParas LaunchBatchCopyTaskEnQueue(const string &op_name, const PROC_FUNC &acl_call);
};

class EventTask {
public:
    explicit EventTask(aclrtEvent event, EventAllocatorType allocatorType = RESERVED)
        : eventParam_(event, allocatorType){};
    ~EventTask() = default;
    void LaunchRecordTask(c10_npu::NPUStream npuStream, unsigned int flags);
    void LaunchWaitTask(c10_npu::NPUStream npuStream, unsigned int flags);
    void LaunchLazyDestroyTask(c10::DeviceIndex device_index);

private:
    EventParas eventParam_;
};

AsyncCopyTask::AsyncCopyTask(void *dst, size_t dstLen, void *src, size_t srcLen, aclrtMemcpyKind kind)
{
    copyParam_.dst = dst;
    copyParam_.dstLen = dstLen;
    copyParam_.src = src;
    copyParam_.srcLen = srcLen;
    copyParam_.kind = kind;
}

void AsyncCopyTask::LaunchCopyTask()
{
    RECORD_FUNCTION(CopyParas::COPY_PARAS_MAP[copyParam_.kind], std::vector<c10::IValue>({}));
    auto cur_stream = c10_npu::getCurrentNPUStream();
    if (!cur_stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, CopyParas::COPY_PARAS_MAP[copyParam_.kind]);
#endif
        QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
        c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, CopyParas::COPY_PARAS_MAP[copyParam_.kind],
            params.correlation_id);
#endif
    } else {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        if (c10_npu::acl::AclrtMemcpyAsyncWithConditionExist() && copyParam_.kind == aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtMemcpyAsyncWithCondition(copyParam_.dst, copyParam_.dstLen, copyParam_.src, copyParam_.srcLen,
                copyParam_.kind, stream));
        } else {
            NPU_CHECK_ERROR(aclrtMemcpyAsync(copyParam_.dst, copyParam_.dstLen, copyParam_.src, copyParam_.srcLen,
                copyParam_.kind, stream));
        }
    }
}

QueueParas AsyncBatchCopyTask::LaunchBatchCopyTaskEnQueue(const string &op_name, const PROC_FUNC &acl_call)
{
    at_npu::native::ExecuteParasOpApiV2 execParams;
    execParams.opName = const_cast<std::string *>(&op_name);
    execParams.customHandler = const_cast<PROC_FUNC *>(&acl_call);
    QueueParas params(EXECUTE_OPAPI_V2, sizeof(at_npu::native::ExecuteParasOpApiV2), &execParams);
    c10_npu::enCurrentNPUStream(&params);
    return params;
}

void AsyncBatchCopyTask::LaunchBatchCopyTask(void **dsts, size_t *dstLens, void **srcs, size_t *srcLens,
                                             size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes,
                                             size_t numAttrs, aclrtStream stream)
{
    RECORD_FUNCTION(batch_copy_paras, std::vector<c10::IValue>({}));
    auto cur_stream = c10_npu::getCurrentNPUStream();
    if (!cur_stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, batch_copy_paras);
#endif
        auto dsts_vec = std::vector<void *>(dsts, dsts + numBatches);
        auto dstLens_vec = std::vector<size_t>(dstLens, dstLens + numBatches);
        auto srcs_vec = std::vector<void *>(srcs, srcs + numBatches);
        auto srcLens_vec = std::vector<size_t>(srcLens, srcLens + numBatches);
        auto attrs_vec = std::vector<aclrtMemcpyBatchAttr>(attrs, attrs + numBatches);
        auto attrsIndexes_vec = std::vector<size_t>(attrsIndexes, attrsIndexes + numBatches);
        auto acl_call = [dsts_vec, dstLens_vec, srcs_vec, srcLens_vec, numBatches, attrs_vec,
                              attrsIndexes_vec, numAttrs, stream]() -> int {
            void *dsts[numBatches];
            void *srcs[numBatches];
            size_t dstLens[numBatches];
            size_t srcLens[numBatches];
            size_t attrsIndexes[numBatches];
            aclrtMemcpyBatchAttr attrs[numBatches];
            for (size_t i = 0; i < numBatches; ++i) {
                dsts[i] = dsts_vec.at(i);
                srcs[i] = srcs_vec.at(i);
                dstLens[i] = dstLens_vec.at(i);
                srcLens[i] = srcLens_vec.at(i);
                attrsIndexes[i] = attrsIndexes_vec.at(i);
                attrs[i] = attrs_vec.at(i);
            }
            size_t failIdx = SIZE_MAX;
            return c10_npu::acl::AclrtMemcpyBatchAsync(dsts, dstLens, srcs, srcLens, numBatches, attrs, attrsIndexes,
                                                       numAttrs, &failIdx, stream);
        };
        QueueParas params = LaunchBatchCopyTaskEnQueue("aclrtMemcpyBatchAsync", acl_call);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, batch_copy_paras, params.correlation_id);
#endif
    } else {
        size_t failIdx = SIZE_MAX;
        NPU_CHECK_ERROR(
            c10_npu::acl::AclrtMemcpyBatchAsync(dsts, dstLens, srcs, srcLens, numBatches, attrs, attrsIndexes,
                                                numAttrs, &failIdx, stream));
    }
}

aclError LaunchAsyncCopyTask(void *dst, size_t dstLen, void *src, size_t srcLen, aclrtMemcpyKind kind)
{
    AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
    copyTask.LaunchCopyTask();
    return ACL_ERROR_NONE;
}

aclError LaunchBatchAsyncCopyTask(void **dsts, size_t *dstLens, void **srcs, size_t *srcLens,
                                  size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs,
                                  aclrtStream stream)
{
    AsyncBatchCopyTask copyTask;
    copyTask.LaunchBatchCopyTask(dsts, dstLens, srcs, srcLens, numBatches, attrs, attrsIndexes, numAttrs, stream);
    return ACL_ERROR_NONE;
}

void EventTask::LaunchRecordTask(c10_npu::NPUStream npuStream, unsigned int flags)
{
    RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[RECORD_EVENT], std::vector<c10::IValue>({}));
    if (!npuStream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[RECORD_EVENT]);
#endif
        uint64_t prof_correlation_id = 0;
        {
            c10_npu::NPUStreamGuard guard(npuStream);
            QueueParamType eventType = RECORD_EVENT;
            if (flags == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
                eventType = WRITE_VALUE;
            }
            QueueParas params(eventType, sizeof(EventParas), &eventParam_);
            c10_npu::NPUEventManager::GetInstance().IncreaseUnrecordedCount(eventParam_.event);
            c10_npu::enCurrentNPUStream(&params);
            prof_correlation_id = params.correlation_id;
        }
        ASCEND_LOGD("Event: LaunchRecordTask is successfully executed, event=%p", eventParam_.event);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[RECORD_EVENT],
            prof_correlation_id);
#endif
    } else {
        if (flags == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtValueWrite(eventParam_.event, 1, npuStream));
            ASCEND_LOGI("External Event: aclrtValueWrite is successfully executed, stream=%p, event=%p", npuStream.stream(false),
                eventParam_.event);
        } else {
            NPU_CHECK_ERROR(aclrtRecordEvent(eventParam_.event, npuStream));
            ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed, stream=%p, event=%p", npuStream.stream(false),
                eventParam_.event);
        }
    }
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream, unsigned int flags)
{
    EventTask recordTask(event);
    recordTask.LaunchRecordTask(npuStream, flags);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventRecord(reinterpret_cast<uintptr_t>(event),
            reinterpret_cast<uintptr_t>(npuStream.stream(false)));
    }
#endif
    return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream, unsigned int flags)
{
    RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[WAIT_EVENT], std::vector<c10::IValue>({}));
    if (!npuStream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[WAIT_EVENT]);
#endif
        uint64_t prof_correlation_id = 0;
        {
            c10_npu::NPUStreamGuard guard(npuStream);
            QueueParamType eventType = WAIT_EVENT;
            if (flags == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
                eventType = WAIT_VALUE;
            }
            QueueParas params(eventType, sizeof(EventParas), &eventParam_);
            c10_npu::NPUEventManager::GetInstance().IncreaseUnwaitedCount(eventParam_.event);
            c10_npu::enCurrentNPUStream(&params);
            prof_correlation_id = params.correlation_id;
        }
        ASCEND_LOGI("Event: LaunchWaitTask is successfully executed, event=%p", eventParam_.event);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[WAIT_EVENT],
            prof_correlation_id);
#endif
    } else {
        if (flags == ACL_EVENT_EXTERNAL && c10_npu::acl::IsExistValueWaitAndWrite()) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtValueWait(eventParam_.event, npuStream));
            NPU_CHECK_ERROR(c10_npu::acl::AclrtValueWrite(eventParam_.event, 0, npuStream));
            ASCEND_LOGI("External Event: aclrtValueWait and aclrtValueWrite is successfully executed, stream=%p, event=%p",
                npuStream.stream(false), eventParam_.event);
        } else {
            NPU_CHECK_ERROR(aclrtStreamWaitEvent(npuStream, eventParam_.event));
            ASCEND_LOGI("Event: aclrtStreamWaitEvent is successfully executed, stream=%p, event=%p",
                npuStream.stream(false), eventParam_.event);
        }
    }
}

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream, unsigned int flags)
{
    EventTask waitTask(event);
    waitTask.LaunchWaitTask(npuStream, flags);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventWait(reinterpret_cast<uintptr_t>(event),
            reinterpret_cast<uintptr_t>(npuStream.stream(false)));
    }
#endif
    return ACL_ERROR_NONE;
}

void EventTask::LaunchLazyDestroyTask(c10::DeviceIndex device_index)
{
    RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[LAZY_DESTROY_EVENT], std::vector<c10::IValue>({}));
    auto cur_stream = c10_npu::getCurrentNPUStream();
    if (!cur_stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[LAZY_DESTROY_EVENT]);
#endif
        QueueParas params(LAZY_DESTROY_EVENT, sizeof(EventParas), &eventParam_);
        c10_npu::enCurrentNPUStream(&params, device_index);
        ASCEND_LOGD("Event: LaunchLazyDestroyTask is successfully executed, event=%p", eventParam_.event);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[LAZY_DESTROY_EVENT],
            params.correlation_id);
#endif
    } else {
        NPU_CHECK_ERROR(c10_npu::NPUEventManager::GetInstance().LazyDestroy(eventParam_.event), "aclrtDestroyEvent");
    }
}

aclError LaunchLazyDestroyEventTask(aclrtEvent event, c10::DeviceIndex device_index)
{
    EventTask lazyDestroyTask(event);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventDeletion(reinterpret_cast<uintptr_t>(event));
    }
#endif
    lazyDestroyTask.LaunchLazyDestroyTask(device_index);
    return ACL_ERROR_NONE;
}

} // namespace queue
} // namespace c10
