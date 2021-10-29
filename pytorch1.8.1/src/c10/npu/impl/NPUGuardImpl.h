// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
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

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <c10/npu/NPUException.h>
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/sys_ctrl/npu_sys_ctrl.h>
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include <cassert>


namespace c10 {
namespace npu {
namespace impl {

struct NPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::NPU;

  NPUGuardImpl() {}
  explicit NPUGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::NPU);
  }
  DeviceType type() const override {
    return DeviceType::NPU;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::NPU);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_NPU_CHECK(aclrtSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device = 0;
    C10_NPU_CHECK(aclrtGetDevice(&device));
    return Device(DeviceType::NPU, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::NPU);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_NPU_CHECK(aclrtSetDevice(d.index()));
    }
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    int old_device = 0;
    aclError ret = aclrtGetDevice(&old_device);
    if (ret != ACL_ERROR_NONE){
      C10_NPU_CHECK_WARN(aclrtSetDevice(d.index()));
    }else if(old_device != d.index()){
      C10_NPU_CHECK_WARN(aclrtSetDevice(d.index()));
    }
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentNPUStream(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultNPUStream(d.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    NPUStream cs(s);
    auto old_stream = getCurrentNPUStream(s.device().index());
    setCurrentNPUStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return c10::npu::device_count();
  }

  // Event-related functions
  void createEvent(aclrtEvent* acl_event, const EventFlag flag) const {
    /*
    // Maps PyTorch's Event::Flag to NPU flag
    auto cuda_flag = cudaEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
      case EventFlag::NPU_EVENT_DISABLE_TIMING:
        cuda_flag = cudaEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
      case EventFlag::NPU_EVENT_DEFAULT:
        cuda_flag = cudaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "NPU event received unknown flag");
    }*/

    // C10_NPU_CHECK(cudaEventCreateWithFlags(cuda_event, cuda_flag));
    C10_NPU_CHECK(aclrtCreateEvent(acl_event));
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto acl_event = static_cast<aclrtEvent>(event);
    int orig_device;
    C10_NPU_CHECK_WARN(aclrtDestroyEvent(acl_event));
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    aclrtEvent npu_event = static_cast<aclrtEvent>(*event);
    NPUStream npu_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!npu_event)
      aclrtCreateEvent(&npu_event);
    C10_NPU_CHECK(aclrtRecordEvent(npu_event, npu_stream));
    // Makes the void* point to the (possibly just allocated) NPU event
    *event = npu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    NPUStream npu_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_NPU_CHECK(aclrtStreamWaitEvent(npu_stream, npu_event));
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    aclrtEventStatus status;
    const aclError err = aclrtQueryEvent(npu_event, &status);
    if (err != ACL_ERROR_NONE) {
      C10_NPU_CHECK(err);
    }
    return (status == ACL_EVENT_STATUS_COMPLETE);
  }
};

} // namespace impl
} // namespace npu
} // namespace c10
