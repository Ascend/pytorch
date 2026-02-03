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
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <third_party/acl/inc/acl/acl.h>

namespace c10_npu {

bool is_lazy_set_device();

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

aclError GetDeviceWithoutSet(int32_t *device);

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

C10_NPU_API aclError MaybeSetDevice(c10::DeviceIndex device);

/**
 * @ingroup torch_npu
 * @brief reset all device id by ACL interface: aclrtResetDevice.
 *
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
std::vector<int8_t> GetUsedDevices();

aclError ResetUsedDevices();

aclError DestroyUsedStreams();

aclError SynchronizeUsedDevices();

aclrtContext GetDeviceContext(int32_t device);

bool isDeviceCtxActive(int32_t device);

C10_NPU_API c10::DeviceIndex current_device();

C10_NPU_API void set_device(c10::DeviceIndex device);

C10_NPU_API void device_synchronize();

C10_NPU_API int ExchangeDevice(int device);

int MaybeExchangeDevice(int to_device);

void SetTargetDevice();

void LazySetDevice(c10::DeviceIndex device);

int GetLocalDevice();

aclError SetDeviceResLimit(int32_t device, int32_t type, uint32_t value);

C10_NPU_API uint32_t GetDeviceResLimit(int32_t deviceId, int32_t type);

aclError ResetDeviceResLimit(int32_t deviceId);

aclError SetStreamResLimit(NPUStream stream, int32_t type, uint32_t value);

aclError ResetStreamResLimit(NPUStream stream);

C10_NPU_API uint32_t GetStreamResLimit(NPUStream stream, int32_t type);

aclError UseStreamResInCurrentThread(aclrtStream stream);

aclError UnuseStreamResInCurrentThread();

C10_NPU_API uint32_t GetResInCurrentThread(int32_t type);

void SetDeterministicLevel(uint32_t level);

uint32_t GetDeterministicLevel();

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// it's used to store npu synchronization state
// through this global state to determine the synchronization debug mode
class WarningState {
public:
    void set_sync_debug_mode(SyncDebugMode level)
    {
        sync_debug_mode = level;
    }

    SyncDebugMode get_sync_debug_mode()
    {
        return sync_debug_mode;
    }

private:
    SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

C10_NPU_API inline WarningState& warning_state()
{
    static WarningState warning_state_;
    return warning_state_;
}

// this function has to be called from callers performing npu synchronizing
// operations, to raise proper error or warning
C10_NPU_API void warn_or_error_on_sync();

enum class CallStateMode { L_UNKNOW = -1, L_FORWARD = 0, L_BACKWARD };

enum class ModelMode { L_UNKNOW = -1, L_TRAIN = 0, L_INFER };

// it's used to store npu call state. eg: forward, backward.
class ModelState {
public:
    void set_call_state(CallStateMode mode)
    {
        call_state_mode = mode;
    }

    CallStateMode get_call_state()
    {
        return call_state_mode;
    }

    void set_model_mode(ModelMode mode)
    {
        model_mode = mode;
    }

    ModelMode get_model_mode()
    {
        return model_mode;
    }

private:
    CallStateMode call_state_mode = CallStateMode::L_UNKNOW;
    ModelMode model_mode = ModelMode::L_UNKNOW;
};

C10_NPU_API inline ModelState& model_state()
{
    static ModelState model_state_;
    return model_state_;
}

bool IsContextInitialized();

C10_NPU_API void stream_synchronize(aclrtStream stream);

} // namespace c10_npu
