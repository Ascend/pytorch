#pragma once

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "third_party/acl/inc/acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include <cstdint>
#include <utility>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace c10_npu {
/*
* NPUEvents are movable not copyable wrappers around NPU's events.
* NPUEvents are constructed lazily when first recorded.
*/
struct NPUEvent {
  // Constructors
  // Default value for `flags` is specified below
  NPUEvent() {}
  NPUEvent(unsigned int flags) : flags_(flags) {}

  // npu do not support IpcEventHandle until now

    ~NPUEvent() {
        try {
            if (is_created_ && (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag())) {
                NPU_CHECK_ERROR(c10_npu::queue::LaunchLazyDestroyEventTask(event_, device_index_));
                if (!c10_npu::acl::IsExistCreateEventExWithFlag()) {
                    NPU_CHECK_ERROR(c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent());
                }
            }
        }
        catch (...) {
            // stay consistent with pytorch, no throw
        }
    }

  NPUEvent(const NPUEvent&) = delete;
  NPUEvent& operator=(const NPUEvent&) = delete;

  NPUEvent(NPUEvent&& other) { moveHelper(std::move(other)); }
  NPUEvent& operator=(NPUEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  operator aclrtEvent() const { return event(); }

  // aclrtEvent do not support Less than operator until now

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(c10::DeviceType::PrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  c10::DeviceIndex device_index() const {return device_index_;}
  aclrtEvent event() const { return event_; }

  bool query() const {
    if (!is_created_) {
      return true;
    }
    if (c10_npu::option::OptionsManager::CheckQueueEnable() &&
        !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
      return false;
    }
    acl::aclrtEventRecordedStatus currStatus =
        acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event_, &currStatus));

    if (currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      return true;
    }
    return false;
  }

  void record() { record(getCurrentNPUStream()); }

  void recordOnce(const NPUStream& stream) {
    if (!was_recorded_) record(stream);
  }

  void record(const NPUStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
        " does not match recording stream's device ", stream.device_index(), ".");
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(c10_npu::queue::LaunchRecordEventTask(event_, stream));
    was_recorded_ = true;
  }

  void block(const NPUStream& stream) {
    if (is_created_) {
      NPUGuard guard(stream.device_index());
      NPU_CHECK_ERROR(c10_npu::queue::LaunchWaitEventTask(event_, stream));
    }
  }

  float elapsed_time(const NPUEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
      ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
    }

    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed.");
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(other.event_));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed.");
    // raise error if either event is recorded but not yet completed
    NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      NPUStatus ret = c10_npu::emptyAllNPUStream();
      if (ret != SUCCESS) {
        ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
      }
      NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
      ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed.");
    }
  }

  // npu do not support IpcEventHandle until now

private:
  unsigned int flags_ = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  aclrtEvent event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event_, flags_));
    ASCEND_LOGI("Event: aclrtCreateEvent is successfully executed.");
    is_created_ = true;
  }

  void moveHelper(NPUEvent&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10_npu
