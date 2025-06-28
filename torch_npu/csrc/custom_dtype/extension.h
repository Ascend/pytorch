#pragma once

#include <ATen/ATen.h>
#include "torch_npu/csrc/custom_dtype/Init.h"

namespace c10_npu {
at::Tensor cast_to_fp8(const at::Tensor &input, int otype);

void cast_to_fp8_noalloc(const at::Tensor &input, at::Tensor output, int otype);

at::Tensor cast_from_fp8(const at::Tensor &input, int itype, int otype);
}
