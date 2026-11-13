#pragma once

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu::impl {
using PythonLazyInitFn = void (*)();
C10_NPU_API void registerPythonLazyInit(PythonLazyInitFn fn);
C10_NPU_API void runPythonLazyInit();
} // namespace c10_npu::impl
