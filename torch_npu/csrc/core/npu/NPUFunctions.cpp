#include <mutex>
#include <unistd.h>
#include <unordered_map>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

static uint32_t dev_count = 0;
static thread_local int local_device = -1;
static std::unordered_map<int8_t, aclrtContext> used_devices;
std::recursive_mutex mtx;
thread_local int targetDeviceIndex = -1;

c10::DeviceIndex device_count() noexcept
{
    // initialize number of devices only once
    if (dev_count == 0) {
        aclError error = aclrtGetDeviceCount(&dev_count);
        if (error != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
            ASCEND_LOGE("get device count of NPU failed");
            return 0;
        }
        return static_cast<c10::DeviceIndex>(dev_count);
    }
    return static_cast<c10::DeviceIndex>(dev_count);
}

c10::DeviceIndex device_count_ensure_non_zero()
{
    unsigned int count = 0;

    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetDeviceCount(&count));
    TORCH_CHECK(count, "No NPUs are available", PTA_ERROR(ErrCode::UNAVAIL));

    return static_cast<c10::DeviceIndex>(count);
}

aclError GetDevice(int32_t *device)
{
    if (targetDeviceIndex >= 0) {
        *device = targetDeviceIndex;
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(targetDeviceIndex));
        return ACL_ERROR_NONE;
    }

    if (local_device >= 0) {
        *device = local_device;
        return ACL_ERROR_NONE;
    }
    aclError err =  aclrtGetDevice(device);
    if (err != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
    }
    if (err == ACL_ERROR_NONE) {
        local_device = *device;
    } else if (err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_ERROR_NONE) {
        *device = 0;
        local_device = 0;
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (used_devices.find(local_device) == used_devices.end()) {
            NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetCurrentContext(&used_devices[local_device]));
        }
        return ACL_ERROR_NONE;
    }
    return err;
}

aclError GetDeviceWithoutSet(int32_t *device)
{
    if (targetDeviceIndex >= 0) {
        *device = targetDeviceIndex;
        return ACL_ERROR_NONE;
    }

    if (local_device >= 0) {
        *device = local_device;
        return ACL_ERROR_NONE;
    }
    aclError err =  aclrtGetDevice(device);
    if (err != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
    }
    if (err == ACL_ERROR_NONE) {
        local_device = *device;
    } else if (err == ACL_ERROR_RT_CONTEXT_NULL) {
        *device = -1;
        return ACL_ERROR_NONE;
    }
    return err;
}

aclError SetDevice(c10::DeviceIndex device)
{
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));
    targetDeviceIndex = -1;
    if (local_device == device) {
        return ACL_ERROR_NONE;
    }

    if (c10_npu::NeedMainThreadBind()) {
        c10_npu::SetThreadAffinity(device);
    }

    aclError err = aclrtSetDevice(device);
    if (err == ACL_ERROR_NONE) {
        local_device = device;
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (used_devices.find(local_device) == used_devices.end()) {
            NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetCurrentContext(&used_devices[local_device]));
        }
    }
    return err;
}

aclError ResetUsedDevices()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        aclError err = aclrtResetDevice(it.first);
        if (err != ACL_ERROR_NONE) {
            return err;
        }
    }
    used_devices.clear();
    return ACL_ERROR_NONE;
}

aclError DestroyUsedStreams()
{
    int32_t cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDevice(&cur_device));
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(it.first));
        NPUStream stream = getCurrentNPUStream(it.first);
        aclError acl_ret = acl::AclrtDestroyStreamForce(stream.stream(false));
        if (acl_ret != ACL_ERROR_NONE) {
            return acl_ret;
        }
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclError SynchronizeUsedDevices()
{
    int32_t cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDevice(&cur_device));
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for (const auto it : used_devices) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(it.first));
        aclError acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
        if (acl_ret != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(acl_ret);
            return acl_ret;
        }
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuDeviceSynchronization();
        }
#endif
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclrtContext GetDeviceContext(int32_t device)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        ASCEND_LOGE("NPU device %d has not been initialized! Can not get context", device);
        return nullptr;
    }
    return used_devices[device];
}

bool isDeviceCtxActive(int32_t device)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        return false;
    }
    return used_devices[device] != nullptr;
}

c10::DeviceIndex current_device()
{
    int cur_device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&cur_device));
    return static_cast<c10::DeviceIndex>(cur_device);
}

void set_device(c10::DeviceIndex device)
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device));
}

void device_synchronize()
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtSynchronizeDeviceWithTimeout());
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuDeviceSynchronization();
    }
#endif
}

int ExchangeDevice(int device)
{
    targetDeviceIndex = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(device));

    return device;
}

int MaybeExchangeDevice(int to_device)
{
    int cur_device = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(GetDeviceWithoutSet(&cur_device));
    if (to_device == cur_device) {
        return cur_device;
    }
    if (isDeviceCtxActive(to_device)) {
        ASCEND_LOGI("NPU device %d has not been initialized! We will set targetDeviceIndex.", to_device);
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(to_device));
    } else {
        targetDeviceIndex = to_device;
    }
    return cur_device;
}

void SetTargetDevice()
{
    if (targetDeviceIndex >= 0) {
        NPU_CHECK_ERROR_WITHOUT_UCE(SetDevice(targetDeviceIndex));
    }
}

bool IsContextInitialized()
{
    if (local_device >= 0) {
        return true;
    }

    int32_t device = -1;
    aclError err =  aclrtGetDevice(&device);
    if (err == ACL_ERROR_NONE) {
        return true;
    } else {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        if (err == ACL_ERROR_RT_CONTEXT_NULL) {
            return false;
        }
        NPU_CHECK_ERROR_WITHOUT_UCE(err);
        return false;
    }
}

int GetLocalDevice()
{
    return local_device;
}

void warn_or_error_on_sync()
{
    if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
        TORCH_CHECK(false, "called a synchronizing NPU operation", PTA_ERROR(ErrCode::ACL));
    } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
        TORCH_NPU_WARN("called a synchronizing NPU operation");
    }
}

void stream_synchronize(aclrtStream stream)
{
    if (C10_UNLIKELY(warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
        warn_or_error_on_sync();
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuStreamSynchronization(reinterpret_cast<uintptr_t>(stream));
    }
#endif
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream));
}

} // namespace c10_npu
