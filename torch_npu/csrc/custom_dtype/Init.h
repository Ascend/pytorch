#pragma once

#include "torch_npu/csrc/custom_dtype/CustomDType.h"
#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>

namespace c10_npu {

TORCH_NPU_API PyMethodDef* custom_dtype_functions();

} // namespace c10_npu
#endif
