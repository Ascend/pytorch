#pragma once

#include <torch/csrc/python_headers.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {

TORCH_NPU_API void bind_npu_recovery_functions(PyObject* module);

} // namespace c10_npu
