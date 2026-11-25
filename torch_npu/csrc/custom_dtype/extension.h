#pragma once

#include <ATen/ATen.h>
#include "torch_npu/csrc/custom_dtype/CustomDType.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
C10_NPU_API at::Tensor cast_to_fp8(const at::Tensor& input, int otype);

C10_NPU_API void cast_to_fp8_noalloc(const at::Tensor& input, at::Tensor output, int otype);

C10_NPU_API at::Tensor cast_from_fp8(const at::Tensor& input, int itype, int otype);
} // namespace c10_npu
