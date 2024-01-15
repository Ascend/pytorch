#include <mutex>
#include <unordered_map>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace c10_npu {
    static thread_local int local_device = -1;
    static std::unordered_map<int8_t, aclrtContext> used_devices;
    std::mutex mtx;

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

    aclError SetDevice(c10::DeviceIndex device)
    {
        TORCH_CHECK(device >= 0, "device id must be positive!");

        if (local_device == device) {
            return ACL_ERROR_NONE;
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

    aclrtContext GetDeviceContext(int32_t device)
    {
        if (used_devices.find(device) == used_devices.end()) {
            ASCEND_LOGE("NPU device %d has been initialized! Can not get context", device);
            return nullptr;
        }
        return used_devices[device];
    }
}
