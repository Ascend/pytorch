#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

NPUEventManager::NPUEventManager() : thread_pool_(std::make_shared<c10::TaskThreadPool>(5)){};

NPUEventManager &NPUEventManager::GetInstance()
{
    static NPUEventManager instance;
    return instance;
}

void NPUEventManager::run(aclrtEvent event)
{
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventDeletion(reinterpret_cast<uintptr_t>(event));
    }
#endif
    int err = aclrtDestroyEvent(event);
    if (err != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        C10_NPU_SHOW_ERR_MSG();
        return;
    }
    ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", event);
}

aclError NPUEventManager::QueryAndDestroyEvent()
{
    std::lock_guard<std::mutex> guard(event_queue_mutex_);
    while (!npu_events_.empty()) {
        aclrtEvent event = npu_events_.front();
        acl::aclrtEventRecordedStatus recordStatus = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
        NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event, &recordStatus));
        if (recordStatus != acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            break;
        } else {
            acl::aclrtEventWaitStatus waitStatus = acl::ACL_EVENT_WAIT_STATUS_RESERVED;
            // if the event usage is unknown, ensure the event id not destroyed in advance.
            NPU_CHECK_ERROR(acl::AclQueryEventWaitStatus(event, &waitStatus));
            if (waitStatus != acl::ACL_EVENT_WAIT_STATUS_COMPLETE) {
                break;
            }
        }
        {
            thread_pool_->run(std::bind(&NPUEventManager::run, this, event));
        }

        npu_events_.pop_front();
    }
    return ACL_ERROR_NONE;
}

aclError NPUEventManager::LazyDestroy(aclrtEvent npu_event)
{
    if (c10_npu::acl::IsExistCreateEventExWithFlag()) {
        int err = aclrtDestroyEvent(npu_event);
        if (err == ACL_ERROR_NONE) {
            ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", npu_event);
        } else {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        }
        return err;
    }
    std::lock_guard<std::mutex> guard(event_queue_mutex_);
    npu_events_.push_back(npu_event);
    return ACL_ERROR_NONE;
}

void NPUEventManager::ClearEvent()
{
    if (thread_pool_ != nullptr) {
        thread_pool_->waitWorkComplete();
    }

    while (!npu_events_.empty()) {
        aclrtEvent event = npu_events_.front();
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuEventDeletion(reinterpret_cast<uintptr_t>(event));
        }
#endif
        auto err = aclrtDestroyEvent(event);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
            NPU_CHECK_WARN(err);
        } else {
            ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", event);
        }
        npu_events_.pop_front();
    }
}
void NPUEventManager::IncreaseUnrecordedCount(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    if (it != event_unrecorded_count_.end()) {
        it->second++;
        ASCEND_LOGD("Event: unrecorded count increase, now=%d.", it->second);
    } else {
        event_unrecorded_count_.insert(std::pair<aclrtEvent, int>(event, 1));
        ASCEND_LOGD("Event: unrecorded count increase, now=%d.", 1);
    }
}

void NPUEventManager::DecreaseUnrecordedCount(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    TORCH_CHECK(
        it != event_unrecorded_count_.end(),
        "Event: event must enqueue before dequeue, event=",
        (void *) event, PTA_ERROR(ErrCode::INTERNAL));
    if (it->second == 1) {
        event_unrecorded_count_.erase(event);
        ASCEND_LOGD("Event: unrecorded count decrease, now=%d.", 0);
    } else {
        it->second--;
        ASCEND_LOGD("Event: unrecorded count decrease, now=%d.", it->second);
    }
}

bool NPUEventManager::IsEventRecorded(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    return it == event_unrecorded_count_.end();
}

void NPUEventManager::ClearUnrecordedCount()
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);
    event_unrecorded_count_.clear();
}

}  // namespace c10_npu
