#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace c10_npu {
    static thread_local int local_device = -1;

    aclError GetDevice(int32_t *device)
    {
        if (local_device >= 0) {
            *device = local_device;
            return ACL_ERROR_NONE;
        }
        return aclrtGetDevice(&local_device);
    }

    aclError SetDevice(c10::DeviceIndex device)
    {
        TORCH_CHECK(device >= 0, "device id must be positive!");
        aclError err = aclrtSetDevice(device);
        if (err == ACL_ERROR_NONE) {
            local_device = device;
        }
        return err;
    }
}