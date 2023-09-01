#pragma once

#include <c10/core/Device.h>

namespace torch_npu {

// device init related funcs
void init_npu(const c10::DeviceIndex device_index=0);
void init_npu(const std::string& device_str);
void init_npu(const at::Device& device);

// device finalize related funcs
void finalize_npu();

bool is_npu_device(const at::Device& device);
c10::DeviceIndex current_device();

// device Synchronize
bool npuSynchronizeDevice(bool check_error = true);

} // namespace torch_npu
