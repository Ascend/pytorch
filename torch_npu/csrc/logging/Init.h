#ifndef BUILD_LIBTORCH
#pragma once

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace logging {
TORCH_NPU_API PyMethodDef* logging_functions();

} // namespace logging
} // namespace torch_npu
#endif