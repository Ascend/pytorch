#pragma once

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "third_party/acl/inc/acl/acl.h"
#include <cstdint>
#include <utility>

namespace c10_npu {
/*
* NPUEvents are movable not copyable wrappers around NPU's events.
*
* NPUEvents are constructed lazily when first recorded unless it is
* reconstructed from a aclrtIpcEventHandle. The event has a device, and this
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct C10_NPU_API NPUEvent {
    // Constructors
    // Default value for `flags` is specified below
    NPUEvent();
    NPUEvent(unsigned int flags) : flags_(flags) {}
    NPUEvent(c10::DeviceIndex device_index, const aclrtIpcEventHandle* handle);

    ~NPUEvent();

    NPUEvent(const NPUEvent&) = delete;
    NPUEvent& operator=(const NPUEvent&) = delete;

    NPUEvent(NPUEvent&& other);
    NPUEvent& operator=(NPUEvent&& other);

    operator aclrtEvent() const { return event(); }

    // aclrtEvent do not support Less than operator until now

    c10::optional<at::Device> device() const
    {
        if (is_created_) {
            return at::Device(c10::DeviceType::PrivateUse1, device_index_);
        } else {
            return {};
        }
    }

    bool isCreated() const { return is_created_; }
    c10::DeviceIndex device_index() const { return device_index_; }
    aclrtEvent event() const { return event_; }

    bool query() const;
    void record();
    void recordOnce(const NPUStream& stream);
    void record(const NPUStream& stream);
    void block(const NPUStream& stream);
    float elapsed_time(const NPUEvent& other) const;
    uint64_t recorded_time() const;
    void synchronize() const;
    void reset(const NPUStream& stream) const;
    void ipc_handle(aclrtIpcEventHandle* handle);

private:
    unsigned int flags_;
    bool is_created_ = false;
    bool was_recorded_ = false;
    c10::DeviceIndex device_index_ = -1;
    aclrtEvent event_ = nullptr;
    bool is_waited_ = false;

    void createEvent(c10::DeviceIndex device_index);
    void moveHelper(NPUEvent&& other);
};

// EventPool - Thread-safe pool of NPU events to avoid expensive AclrtCreateEventWithFlag
// calls. AclrtCreateEventWithFlag when concurrently invoked from multiple threads can be
// very expensive (especially on certain device/driver combinations).
using NPUEventPtr =
    std::unique_ptr<NPUEvent, std::function<void(NPUEvent*)>>;

class EventPool {
public:
    EventPool(unsigned int flags) : flags_(flags), pools_(c10_npu::device_count()) {}
    NPUEventPtr get(const c10::DeviceIndex device);
    void empty_cache();

private:
    // Cache line size for alignment to avoid false sharing
    static constexpr size_t kCacheLineSize = 64;

    struct PerDevicePool {
        alignas(kCacheLineSize) std::mutex mutex_;
        std::vector<std::unique_ptr<NPUEvent>> event_pool_;
    };

    unsigned int flags_;
    std::vector<PerDevicePool> pools_;
};
} // namespace c10_npu
