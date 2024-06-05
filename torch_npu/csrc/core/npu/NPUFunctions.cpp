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
std::mutex mtx;

c10::DeviceIndex device_count() noexcept
{
    // initialize number of devices only once
    if (dev_count == 0) {
        aclError error = aclrtGetDeviceCount(&dev_count);
        if (error != ACL_ERROR_NONE) {
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

    NPU_CHECK_ERROR(aclrtGetDeviceCount(&count));
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
    if (err == ACL_ERROR_NONE) {
        local_device = *device;
    } else if (err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_ERROR_NONE) {
        *device = 0;
        local_device = 0;
        if (used_devices.find(local_device) == used_devices.end()) {
            std::lock_guard<std::mutex> lock(mtx);
            if (used_devices.find(local_device) == used_devices.end()) {
                NPU_CHECK_ERROR(aclrtGetCurrentContext(&used_devices[local_device]));
            }
        }
        return ACL_ERROR_NONE;
    }
    return err;
}

inline bool has_set_pthread_affinity()
{
    int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
    int count = 0;

    cpu_set_t mask;
    pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
    for (int i = 0; i < core_nums; i++) {
        count += CPU_ISSET(i, &mask);
    }

    return count != core_nums;
}

aclError SetDevice(c10::DeviceIndex device)
{
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));

    if (local_device == device) {
        return ACL_ERROR_NONE;
    }

    static const bool set_pthread_affinity = has_set_pthread_affinity();
    if (!set_pthread_affinity) {
        uint32_t bind_conf = c10_npu::option::OptionsManager::GetBindCpuConf();
        // bind_conf=1, bind cores averagely based on device_id
        if (bind_conf == 1) {
            int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
            int device_nums = device_count_ensure_non_zero();
            int block_size = (core_nums + device_nums - 1) / device_nums;
            int start_core = device * block_size;
            int end_core = std::min((device + 1) * block_size, core_nums);

            cpu_set_t mask;
            CPU_ZERO(&mask);
            for (int i = start_core; i < end_core; i++) {
                CPU_SET(i, &mask);
            }
            pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        }
    }

    aclError err = aclrtSetDevice(device);
    if (err == ACL_ERROR_NONE) {
        local_device = device;
        if (used_devices.find(local_device) == used_devices.end()) {
            std::lock_guard<std::mutex> lock(mtx);
            if (used_devices.find(local_device) == used_devices.end()) {
                NPU_CHECK_ERROR(aclrtGetCurrentContext(&used_devices[local_device]));
            }
        }
    }
    return err;
}

aclError ResetUsedDevices()
{
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
    NPU_CHECK_ERROR(GetDevice(&cur_device));
    for (const auto it : used_devices) {
        NPU_CHECK_ERROR(SetDevice(it.first));
        NPUStream stream = getCurrentNPUStream(it.first);
        aclError acl_ret = acl::AclrtDestroyStreamForce(stream);
        if (acl_ret != ACL_ERROR_NONE) {
            return acl_ret;
        }
    }
    NPU_CHECK_ERROR(SetDevice(cur_device));
    return ACL_ERROR_NONE;
}

aclError SynchronizeUsedDevices()
{
    int32_t cur_device = 0;
    NPU_CHECK_ERROR(GetDevice(&cur_device));
    for (const auto it : used_devices) {
        NPU_CHECK_ERROR(SetDevice(it.first));
        aclError acl_ret = aclrtSynchronizeDevice();
        if (acl_ret != ACL_ERROR_NONE) {
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
    if (used_devices.find(device) == used_devices.end()) {
        ASCEND_LOGE("NPU device %d has been initialized! Can not get context", device);
        return nullptr;
    }
    return used_devices[device];
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
