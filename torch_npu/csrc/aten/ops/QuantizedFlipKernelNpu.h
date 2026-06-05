#pragma once

#include <ATen/ATen.h>

namespace at_npu {
namespace native {

at::Tensor quantized_flip(const at::Tensor& self, at::IntArrayRef dims);

} // namespace native
} // namespace at_npu
