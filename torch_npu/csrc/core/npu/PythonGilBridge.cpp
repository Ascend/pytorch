#include "torch_npu/csrc/core/npu/PythonGilBridge.h"

#include <atomic>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

namespace c10_npu::impl {
namespace {
std::atomic<const PythonGilHooks*> python_gil_state_hooks{nullptr};
std::atomic<PythonGilCallback> python_gil_executor{nullptr};
} // namespace

void registerPythonGilHooks(const PythonGilHooks* hooks) {
  TORCH_CHECK(
      hooks != nullptr && hooks->release_if_held != nullptr && hooks->restore != nullptr &&
      hooks->run_with_gil_released != nullptr);
  static c10::once_flag register_once;
  c10::call_once(register_once, [hooks] {
    python_gil_state_hooks.store(hooks, std::memory_order_release);
    python_gil_executor.store(hooks->run_with_gil_released, std::memory_order_release);
  });
}

const PythonGilHooks* getPythonGilHooks() {
  return python_gil_state_hooks.load(std::memory_order_acquire);
}

PythonGilCallback getPythonGilCallback() {
  return python_gil_executor.load(std::memory_order_acquire);
}

} // namespace c10_npu::impl
