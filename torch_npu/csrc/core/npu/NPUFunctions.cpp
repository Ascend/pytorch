#include <mutex>
#include <unistd.h>
#include <unordered_map>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

static uint32_t dev_count = 0;
static thread_local int local_device = -1;
static std::unordered_map<int8_t, aclrtContext> used_devices;
std::recursive_mutex mtx;

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

aclError SetDevice(c10::DeviceIndex device)
{
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));

    if (local_device == device) {
        return ACL_ERROR_NONE;
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
        aclError acl_ret = acl::AclrtDestroyStreamForce(stream);
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
    NPU_CHECK_ERROR(SetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclrtContext GetDeviceContext(int32_t device)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        ASCEND_LOGE("NPU device %d has been initialized! Can not get context", device);
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
    NPU_CHECK_ERROR(c10_npu::GetDevice(&cur_device));
    return static_cast<c10::DeviceIndex>(cur_device);
}

    void set_device(c10::DeviceIndex device)
    {
        NPU_CHECK_ERROR(c10_npu::SetDevice(device));
    }

    void device_synchronize()
    {
        NPU_CHECK_ERROR(aclrtSynchronizeDevice());
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuDeviceSynchronization();
        }
#endif
    }

    int ExchangeDevice(int device)
    {
        NPU_CHECK_ERROR(SetDevice(device));

    return device;
}

int GetLocalDevice()
{
    return local_device;
}

}
