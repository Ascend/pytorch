#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace c10_npu {
    static thread_local int local_device = -1;

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
        }
        return err;
    }
}