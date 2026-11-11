#pragma once

#include <c10/util/FunctionRef.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu::impl {

using PythonGilCallback = void (*)(c10::function_ref<void()>);

// Registered callbacks must remain valid for the process lifetime.
struct PythonGilHooks {
  void* (*release_if_held)();
  void (*restore)(void*);
  // Execute fn synchronously on the calling thread; do not retain it.
  PythonGilCallback run_with_gil_released;
};

C10_NPU_API void registerPythonGilHooks(const PythonGilHooks* hooks);
C10_NPU_API const PythonGilHooks* getPythonGilHooks();
C10_NPU_API PythonGilCallback getPythonGilCallback();

} // namespace c10_npu::impl
