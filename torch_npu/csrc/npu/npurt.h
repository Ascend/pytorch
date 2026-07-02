#pragma once

#include <Python.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace npurt {

TORCH_NPU_API PyMethodDef* npurt_functions();

} // namespace npurt
} // namespace torch_npu
