#pragma once

#include <c10/core/Device.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {

// device init related funcs
TORCH_NPU_API void init_npu(const c10::DeviceIndex device_index = 0);
TORCH_NPU_API void init_npu(const std::string& device_str);
TORCH_NPU_API void init_npu(const at::Device& device);

// device finalize related funcs
TORCH_NPU_API void finalize_npu();

} // namespace torch_npu


namespace torch {
namespace npu {

// device synchronize
TORCH_NPU_API void synchronize(int64_t device_index = -1);

} // namespace npu
} // namespace torch


namespace c10 {
namespace npu {

C10_NPU_API DeviceIndex current_device();

} // namespace npu
} // namespace c10
