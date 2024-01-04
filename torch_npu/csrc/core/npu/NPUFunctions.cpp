#include <unordered_set>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace c10_npu {
    static thread_local int local_device = -1;
    static std::unordered_set<int8_t> used_devices;

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
}
