#pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>


namespace torch_npu {
namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
  return tensor.device().is_privateuseone();
}

inline bool is_npu(const at::TensorOptions& options) {
    return options.device().is_privateuseone();
}

inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::PrivateUse1;
}
}
}
