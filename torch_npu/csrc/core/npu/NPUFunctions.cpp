#include <unordered_set>
#include <unistd.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {

static uint32_t dev_count = 0;
static thread_local int local_device = -1;
static std::unordered_set<int8_t> used_devices;

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
        used_devices.insert(local_device);
    }
    return err;
}

inline bool has_set_pthread_affinity()
{
    unsigned int core_nums = static_cast<unsigned int>(sysconf(_SC_NPROCESSORS_ONLN));

    cpu_set_t mask;
    pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
    for (unsigned int i = 0; i < core_nums; i++) {
        if (!CPU_ISSET(i, &mask)) {
            return true;
        }
    }
    return false;
}

aclError SetDevice(c10::DeviceIndex device)
{
    TORCH_CHECK(device >= 0, "device id must be positive!", PTA_ERROR(ErrCode::VALUE));

    if (local_device == device) {
        return ACL_ERROR_NONE;
    }

    static uint32_t bind_conf = c10_npu::option::OptionsManager::GetCpuAffinityConf();
    // bind_conf=1, bind cores averagely based on device_id
    if (bind_conf == 1) {
        static const bool set_pthread_affinity = has_set_pthread_affinity();
        if (!set_pthread_affinity) {
            int core_nums = sysconf(_SC_NPROCESSORS_ONLN);
            int device_nums = device_count_ensure_non_zero();
            int block_size = (core_nums + device_nums - 1) / device_nums;
            unsigned int start_core = static_cast<unsigned int>(device * block_size);
            unsigned int end_core = static_cast<unsigned int>(std::min((device + 1) * block_size, core_nums));

            cpu_set_t mask;
            CPU_ZERO(&mask);
            for (unsigned int i = start_core; i < end_core; i++) {
                CPU_SET(i, &mask);
            }
            pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        }
    }

    aclError err = aclrtSetDevice(device);
    if (err == ACL_ERROR_NONE) {
        local_device = device;
        used_devices.insert(device);
    }
    return err;
}

aclError ResetUsedDevices()
{
    for (const auto i : used_devices) {
        aclError err = aclrtResetDevice(i);
        if (err != ACL_ERROR_NONE) {
            return err;
        }
    }
    used_devices.clear();
    return ACL_ERROR_NONE;
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
