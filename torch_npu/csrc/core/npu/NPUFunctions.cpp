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
        }
        return err;
    }

    aclError SetDevice(c10::DeviceIndex device)
    {
        TORCH_CHECK(device >= 0, "device id must be positive!");
        int cur_device = -1;
        aclError ret = c10_npu::GetDevice(&cur_device);
        if (ret == ACL_ERROR_NONE && device == cur_device) {
            return ACL_ERROR_NONE;
        }
        aclError err = aclrtSetDevice(device);
        if (err == ACL_ERROR_NONE) {
            local_device = device;
        }
        return err;
    }
}