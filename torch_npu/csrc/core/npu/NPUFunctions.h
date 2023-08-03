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
  int cur_device = 0;
  NPU_CHECK_ERROR(aclrtGetDevice(&cur_device));
  return static_cast<c10::DeviceIndex>(cur_device);
}

inline void set_device(c10::DeviceIndex device) {
}

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// it's used to store npu synchronization state
// through this global state to determine the synchronization debug mode
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode level) {
    sync_debug_mode = level;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

inline WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

// this function has to be called from callers performing npu synchronizing
// operations, to raise proper error or warning
inline void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing NPU operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_NPU_WARN("called a synchronizing NPU operation");
  }
}

} // namespace c10_npu
