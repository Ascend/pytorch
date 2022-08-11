#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include <cassert>


namespace c10_npu {
namespace impl {

struct NPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = at_npu::key::NativeDeviceType;

  NPUGuardImpl() {}
  explicit NPUGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == at_npu::key::NativeDeviceType);
  }
  c10::DeviceType type() const override {
    return at_npu::key::NativeDeviceType;
  }
  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == at_npu::key::NativeDeviceType);
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_NPU_CHECK(aclrtSetDevice(d.index()));
    }
    return old_device;
  }
  c10::Device getDevice() const override {
    int device = 0;
    C10_NPU_CHECK(aclrtGetDevice(&device));
    return c10::Device(at_npu::key::NativeDeviceType, device);
  }
  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == at_npu::key::NativeDeviceType);
    c10_npu::NpuSysCtrl::GetInstance().BackwardsInit();
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    int old_device = 0;
    aclError ret = aclrtGetDevice(&old_device);
    if (ret != ACL_ERROR_NONE){
      C10_NPU_CHECK_WARN(aclrtSetDevice(d.index()));
    }else if(old_device != d.index()){
      C10_NPU_CHECK_WARN(aclrtSetDevice(d.index()));
    }
  }
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10_npu::getCurrentNPUStream(d.index()).unwrap();
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10_npu::getDefaultNPUStream(d.index());
  }
  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    NPUStream cs(s);
    auto old_stream = c10_npu::getCurrentNPUStream(s.device().index());
    c10_npu::setCurrentNPUStream(cs);
    return old_stream.unwrap();
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    return c10_npu::device_count();
  }

  // Event-related functions
  void createEvent(aclrtEvent* acl_event, const c10::EventFlag flag) const {
    C10_NPU_CHECK(aclrtCreateEvent(acl_event));
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto acl_event = static_cast<aclrtEvent>(event);
    int orig_device;
    C10_NPU_CHECK_WARN(aclrtDestroyEvent(acl_event));
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
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

  void block(void* event, const c10::Stream& stream) const override {
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
} // namespace c10_npu
