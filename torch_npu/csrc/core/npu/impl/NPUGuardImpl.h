#pragma once

#include <cassert>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace c10_npu {
namespace impl {

struct C10_NPU_API NPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    NPUGuardImpl() {}
    explicit NPUGuardImpl(c10::DeviceType t);
    c10::DeviceType type() const override
    {
        return c10::DeviceType::PrivateUse1;
    }
    c10::Device exchangeDevice(c10::Device d) const override;
    c10::Device getDevice() const override;
    std::optional<c10::Device> uncheckedGetDevice() const noexcept {
        int32_t device = -1;
        const auto err = c10_npu::GetDevice(&device);
        NPU_CHECK_WARN(err);
        if (err != ACL_ERROR_NONE) {
            return std::nullopt;
        }
        return c10::Device(c10::DeviceType::PrivateUse1, device);
    }
    void setDevice(c10::Device d) const override;
    void uncheckedSetDevice(c10::Device d) const noexcept override;

    c10::Stream getStream(c10::Device d) const noexcept override;
    c10::Stream getDefaultStream(c10::Device d) const override;
    c10::Stream getNewStream(c10::Device d, int priority = 0) const override;
    c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override;
    // NB: These do NOT set the current device
    c10::Stream exchangeStream(c10::Stream s) const noexcept override;
    c10::DeviceIndex deviceCount() const noexcept override;

    // Event-related functions
    void createEvent(aclrtEvent *acl_event, [[maybe_unused]] const c10::EventFlag flag) const;
    void destroyEvent(void *event, const c10::DeviceIndex device_index) const noexcept override;
    void record(void **event, const c10::Stream &stream, const c10::DeviceIndex device_index,
                const c10::EventFlag flag) const;
    void block(void *event, const c10::Stream &stream) const override;
    // May be called from any device
    bool queryEvent(void *event) const override;
    // Stream-related functions
    bool queryStream(const c10::Stream& stream) const override;
    void synchronizeStream(const c10::Stream& stream) const override;
    void synchronizeEvent(void* event) const override;
    void synchronizeDevice(const c10::DeviceIndex device_index) const override;
    void recordDataPtrOnStream(const c10::DataPtr &data_ptr, const c10::Stream &stream) const override;
    double elapsedTime(void *event1, void *event2, c10::DeviceIndex device_index) const override;
};

} // namespace impl
} // namespace c10_npu
