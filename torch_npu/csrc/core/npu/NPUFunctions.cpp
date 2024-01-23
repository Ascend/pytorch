#include <unordered_set>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

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
        TORCH_CHECK(count, "No NPUs are available");

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

    aclError SetDevice(c10::DeviceIndex device)
    {
        TORCH_CHECK(device >= 0, "device id must be positive!");

        if (local_device == device) {
            return ACL_ERROR_NONE;
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
}
