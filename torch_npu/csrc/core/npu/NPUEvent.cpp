#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif


namespace c10_npu {

NPUEvent::NPUEvent()
{
    flags_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
}

NPUEvent::~NPUEvent()
{
    try {
        if (is_created_ && (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag())) {
            NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchLazyDestroyEventTask(event_, device_index_));
            if (!c10_npu::acl::IsExistCreateEventExWithFlag()) {
                c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent();
            }
        }
    }
    catch (...) {
        // stay consistent with pytorch, no throw
    }
}

NPUEvent::NPUEvent(NPUEvent&& other)
{
    moveHelper(std::move(other));
}

NPUEvent& NPUEvent::operator=(NPUEvent&& other)
{
    moveHelper(std::move(other));
    return *this;
}

bool NPUEvent::query() const
{
    if (!is_created_) {
        return true;
    }
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable() &&
        !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
        return false;
    }
    acl::aclrtEventRecordedStatus currStatus =
        acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR_WITHOUT_UCE(acl::AclQueryEventRecordedStatus(event_, &currStatus));

    if (currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        return true;
    }
    return false;
}

void NPUEvent::record()
{
    record(getCurrentNPUStream());
}

void NPUEvent::recordOnce(const NPUStream& stream)
{
    if (!was_recorded_) {
        record(stream);
    }
}

void NPUEvent::record(const NPUStream& stream)
{
    if (!is_created_) {
        createEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
        " does not match recording stream's device ", stream.device_index(), ".",
        PTA_ERROR(ErrCode::PARAM));
    NPUGuard guard(device_index_);
    c10_npu::queue::LaunchRecordEventTask(event_, stream);
    was_recorded_ = true;
}

void NPUEvent::block(const NPUStream& stream)
{
    if (!is_created_ && (flags_ == ACL_EVENT_EXTERNAL)) {
        createEvent(stream.device_index());
    }
    if (is_created_) {
        NPUGuard guard(stream.device_index());
        c10_npu::queue::LaunchWaitEventTask(event_, stream);
    }
}

float NPUEvent::elapsed_time(const NPUEvent& other) const
{
    TORCH_CHECK(is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.",
        PTA_ERROR(ErrCode::INTERNAL));
    float time_ms = 0;
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
        ASCEND_LOGE("Failed to empty NPU task queue, ret: %s", ret.c_str());
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event_);
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtSynchronizeEvent(other.event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, other.event=%p", other.event_);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(other.event_));
    }
#endif
    // raise error if either event is recorded but not yet completed
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
}

uint64_t NPUEvent::recorded_time() const
{
    TORCH_CHECK(is_created_, "Event must be recorded before getting recorded timestamp.", PTA_ERROR(ErrCode::INTERNAL));
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
        ASCEND_LOGE("Failed to empty NPU task queue, ret: %s", ret.c_str());
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent executed successfully, event=%p", event_);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
    }
#endif
    // raise error if either event is recorded but not yet completed
    uint64_t time_stamp = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtEventGetTimestamp(event_, &time_stamp));
    return time_stamp;
}

void NPUEvent::synchronize() const
{
    if (is_created_) {
        NPUStatus ret = c10_npu::emptyAllNPUStream();
        if (ret != SUCCESS) {
            ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
        }
        NPU_CHECK_ERROR_WITHOUT_UCE(aclrtSynchronizeEvent(event_));
        ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event_);
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event_));
        }
#endif
    }
}

void NPUEvent::reset(const NPUStream& stream) const
{
    if (is_created_) {
        TORCH_CHECK(flags_ == ACL_EVENT_EXTERNAL,
                    "API reset() only support ACL_EVENT_EXTERNAL flag event.", PTA_ERROR(ErrCode::INTERNAL));
        NPUGuard guard(stream.device_index());
        NPU_CHECK_ERROR_WITHOUT_UCE(aclrtResetEvent(event_, stream.stream()));
    }
}

void NPUEvent::createEvent(c10::DeviceIndex device_index)
{
    device_index_ = device_index;
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtCreateEventWithFlag(&event_, flags_));
    ASCEND_LOGI("Event: aclrtCreateEventWithFlag is successfully executed, event=%p", event_);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventCreation(reinterpret_cast<uintptr_t>(event_));
    }
#endif
    is_created_ = true;
}

void NPUEvent::moveHelper(NPUEvent&& other)
{
    flags_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
}

} // namespace c10_npu
