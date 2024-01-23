#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include <third_party/acl/inc/acl/acl.h>

namespace c10_npu {

C10_NPU_API c10::DeviceIndex device_count() noexcept;

C10_NPU_API c10::DeviceIndex device_count_ensure_non_zero();

/**
 * @ingroup torch_npu
 * @brief get device id from local thread cache preferentially for performance.
 * If the thread cache has not been initialized, it will get from ACL interface:
 * aclrtGetDevice, and initialize the local thread cache.
 * If the context is empty, it will set device 0.
 *
 * @param device [IN]           device id
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
C10_NPU_API aclError GetDevice(int32_t *device);

/**
 * @ingroup torch_npu
 * @brief set device id by ACL interface: aclrtSetDevice,
 * and update the local thread cache
 *
 * @param device [IN]           device id
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
C10_NPU_API aclError SetDevice(c10::DeviceIndex device);

/**
 * @ingroup torch_npu
 * @brief reset all device id by ACL interface: aclrtResetDevice.
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError ResetUsedDevices();

aclrtContext GetDeviceContext(int32_t device);

C10_NPU_API inline c10::DeviceIndex current_device() {
  int cur_device = 0;
  NPU_CHECK_ERROR(c10_npu::GetDevice(&cur_device));
  return static_cast<c10::DeviceIndex>(cur_device);
}

C10_NPU_API void set_device(c10::DeviceIndex device);

C10_NPU_API void device_synchronize();

C10_NPU_API int ExchangeDevice(int device);

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

C10_NPU_API inline WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

// this function has to be called from callers performing npu synchronizing
// operations, to raise proper error or warning
C10_NPU_API inline void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing NPU operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_NPU_WARN("called a synchronizing NPU operation");
  }
}

} // namespace c10_npu
