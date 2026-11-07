#pragma once

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu::utils {
TORCH_NPU_API void registerPythonGilAdapter();
} // namespace torch_npu::utils
