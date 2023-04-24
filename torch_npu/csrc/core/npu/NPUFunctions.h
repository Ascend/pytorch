#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <third_party/acl/inc/acl/acl.h>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace c10_npu {

inline c10::DeviceIndex device_count() noexcept {
  unsigned int count = 1;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  aclError error = aclrtGetDeviceCount(&count);
  if (error != ACL_ERROR_NONE) {
    // Clear out the error state, so we don't spuriously trigger someone else.
    // (This shouldn't really matter, since we won't be running very much CUDA
    // code in this regime.)
    ASCEND_LOGE("get device count of NPU failed");
    return 0;
  }
  return static_cast<c10::DeviceIndex>(count);
}

inline c10::DeviceIndex current_device() {
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    return c10_npu::NpuSysCtrl::GetInstance().GetCurrentDeviceIndex();
  }
  int cur_device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&cur_device));
  return static_cast<c10::DeviceIndex>(cur_device);
}

inline void set_device(c10::DeviceIndex device) {
}

} // namespace c10_npu
