#ifndef BUILD_LIBTORCH
#pragma once

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace reductions {

TORCH_NPU_API PyMethodDef* reductions_functions();

} // namespace reductions
} // namespace torch_npu

#endif