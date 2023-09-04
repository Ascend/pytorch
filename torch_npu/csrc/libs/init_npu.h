#pragma once

#include <c10/core/Device.h>

namespace torch_npu {

// device init related funcs
void init_npu(const c10::DeviceIndex device_index=0);
void init_npu(const std::string& device_str);
void init_npu(const at::Device& device);

} // namespace torch_npu


namespace torch {
namespace npu {

// device synchronize
void synchronize(int64_t device_index=-1);

} // namespace npu
} // namespace torch


namespace c10 {
namespace npu {

DeviceIndex current_device();

} // namespace npu
} // namespace c10
