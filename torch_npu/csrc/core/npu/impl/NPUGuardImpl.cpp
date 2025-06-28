#pragma GCC visibility push(default)
#include <torch/csrc/jit/serialization/pickler.h>
#include "torch_npu/csrc/core/npu/impl/NPUGuardImpl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUSerialization.h"
#include "torch_npu/csrc/core/npu/NPUHooksInterface.h"

#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

namespace impl {

constexpr c10::DeviceType NPUGuardImpl::static_type;

NPUGuardImpl::NPUGuardImpl(c10::DeviceType t)
{
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1, "DeviceType must be NPU. Actual DeviceType is: ", t,
                          PTA_ERROR(ErrCode::PARAM));
}

c10::Device NPUGuardImpl::exchangeDevice(c10::Device d) const
{
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
        NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(d.index()));
    }
    return old_device;
}

c10::Device NPUGuardImpl::getDevice() const
{
    int device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&device));
    return c10::Device(c10::DeviceType::PrivateUse1, device);
}

void NPUGuardImpl::setDevice(c10::Device d) const
{
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    NPU_CHECK_ERROR(c10_npu::SetDevice(d.index()));
}

void NPUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept
{
    c10_npu::StartMainThreadBind(d.index());
    NPU_CHECK_WARN(c10_npu::SetDevice(d.index()));
}

c10::Stream NPUGuardImpl::getStream(c10::Device d) const noexcept
{
    return c10_npu::getCurrentNPUStream(d.index()).unwrap();
}

c10::Stream NPUGuardImpl::getDefaultStream(c10::Device d) const
{
    return c10_npu::getDefaultNPUStream(d.index());
}

c10::Stream NPUGuardImpl::getStreamFromGlobalPool(c10::Device d, bool isHighPriority) const
{
    return c10_npu::getStreamFromPool(isHighPriority, d.index());
}

c10::Stream NPUGuardImpl::exchangeStream(c10::Stream s) const noexcept
{
    NPUStream cs(s);
    auto old_stream = c10_npu::getCurrentNPUStream(s.device().index());
    c10_npu::setCurrentNPUStream(cs);
    return old_stream.unwrap();
}

c10::DeviceIndex NPUGuardImpl::deviceCount() const noexcept
{
    static c10::DeviceIndex count = c10_npu::device_count();
    return count;
}

// Event-related functions
void NPUGuardImpl::createEvent(aclrtEvent *acl_event, const c10::EventFlag flag) const
{
    auto flag_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtCreateEventWithFlag(acl_event, flag_));
    ASCEND_LOGI("Event: aclrtCreateEventWithFlag is successfully executed, event=%p", *acl_event);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventCreation(reinterpret_cast<uintptr_t>(*acl_event));
    }
#endif
}

void NPUGuardImpl::destroyEvent(void *event, const c10::DeviceIndex device_index) const noexcept
{
    if (!event) {
        return;
    }
    auto acl_event = static_cast<aclrtEvent>(event);
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchLazyDestroyEventTask(acl_event, device_index));
    ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", acl_event);
}

void NPUGuardImpl::record(void **event, const c10::Stream &stream, const c10::DeviceIndex device_index,
                          const c10::EventFlag flag) const
{
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(), "Event device index ", device_index,
                " does not match recording stream's device index ", stream.device_index(), ".",
                PTA_ERROR(ErrCode::PARAM));

    aclrtEvent npu_event = static_cast<aclrtEvent>(*event);
    NPUStream npu_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!npu_event) {
        auto flag_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
        NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtCreateEventWithFlag(&npu_event, flag_));
        ASCEND_LOGI("Event: aclrtCreateEventWithFlag is successfully executed, event=%p", npu_event);
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuEventCreation(reinterpret_cast<uintptr_t>(npu_event));
        }
#endif
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchRecordEventTask(npu_event, npu_stream));
    ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed, stream=%p, event=%p", npu_stream.stream(false),
                npu_event);
    // Makes the void* point to the (possibly just allocated) NPU event
    *event = npu_event;

    // Resets device
    setDevice(orig_device);
}

void NPUGuardImpl::block(void *event, const c10::Stream &stream) const
{
    if (!event) {
        return;
    }
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    NPUStream npu_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchWaitEventTask(npu_event, npu_stream));
    ASCEND_LOGI("Event: aclrtStreamWaitEvent is successfully executed, stream=%p, event=%p",
                npu_stream.stream(false), npu_event);
    setDevice(orig_device);
}

// May be called from any device
bool NPUGuardImpl::queryEvent(void *event) const
{
    if (!event) {
        return true;
    }
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable() != 0 &&
        !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(npu_event)) {
        return false;
    }
    acl::aclrtEventRecordedStatus status = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR_WITHOUT_UCE(acl::AclQueryEventRecordedStatus(npu_event, &status));
    return (status == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE);
}

void NPUGuardImpl::recordDataPtrOnStream(const c10::DataPtr &data_ptr, const c10::Stream &stream) const
{
    NPUStream npu_stream{stream};
    c10_npu::NPUCachingAllocator::recordStream(data_ptr, npu_stream);
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, NPUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                                             \
    int rename_privateuse1_backend()                                                                                   \
    {                                                                                                                  \
        c10::register_privateuse1_backend(#name);                                                                      \
        c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &torch_npu::make_npu_storage_impl);                    \
        at::RegisterPrivateUse1HooksInterface(c10_npu::get_npu_hooks());                                               \
        torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &torch_npu::npu_info_serialization,        \
                                              &torch_npu::npu_info_deserialization);                                   \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(npu)

} // namespace impl

} // namespace c10_npu
#pragma GCC visibility pop
