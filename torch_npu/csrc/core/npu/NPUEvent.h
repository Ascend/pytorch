// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "third_party/acl/inc/acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
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

  // flags is an useless parameter for npu
  // NPUEvent(unsigned int flags) : flags_{flags} {}

  // npu do not support IpcEventHandle until now

  ~NPUEvent() {
    try {
      if (is_created_ && (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag())) {
        NPU_CHECK_ERROR(c10_npu::queue::LaunchLazyDestroyEventTask(event_, device_index_));
        NPU_CHECK_ERROR(c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent());
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
      return at::Device(at_npu::key::NativeDeviceType, device_index_);
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
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
      NPU_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
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

  void recordTimeEvent(const NPUStream& stream) {
    if (!is_created_) {
      createInfiniteEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
        " does not match recording stream's device ", stream.device_index(), ".");
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(c10_npu::queue::LaunchRecordEventTask(event_, stream));
    was_recorded_ = true;
  }

  void reset(const NPUStream& stream) {
    if (is_created_) {
      NPUGuard guard(stream.device_index());
      NPU_CHECK_ERROR(c10_npu::queue::LaunchResetEventTask(event_, stream));
    }
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
      NPU_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
    }

    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI("aclrtSynchronizeEvent is successfully executed.");
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(other.event_));
    ASCEND_LOGI("aclrtSynchronizeEvent is successfully executed.");
    // raise error if either event is recorded but not yet completed
    NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      NPUStatus ret = c10_npu::emptyAllNPUStream();
      if (ret != SUCCESS) {
        NPU_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
      }
      NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
      ASCEND_LOGI("aclrtSynchronizeEvent is successfully executed.");
    }
  }

  // npu do not support IpcEventHandle until now

private:
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  aclrtEvent event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(aclrtCreateEvent(&event_));
    ASCEND_LOGI("aclrtCreateEvent is successfully executed.");
    is_created_ = true;
  }

  void createInfiniteEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(aclrtCreateEventWithFlag(&event_, 0x02U));
    ASCEND_LOGI("aclrtCreateEvent is successfully executed.");
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

