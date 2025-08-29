#pragma once

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace afd {

TORCH_NPU_API PyMethodDef *python_functions();

} // namespace afd
} // namespace torch_npu