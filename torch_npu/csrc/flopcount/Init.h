#ifndef BUILD_LIBTORCH
#pragma once

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace flopcount {
TORCH_NPU_API PyMethodDef* flops_count_functions();

} // namespace flopcount
} // namespace torch_npu
#endif // BUILD_LIBTORCH
