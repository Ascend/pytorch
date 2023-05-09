#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"


namespace torch_npu {

// device init related funcs
void init_npu(const c10::DeviceIndex device_index=0);
void init_npu(const std::string& device_str);
void init_npu(const at::Device& device);

// device finalize related funcs
void finalize_npu();

bool is_npu_device(const at::Device& device);
c10::DeviceIndex current_device();

} // namespace torch_npu
