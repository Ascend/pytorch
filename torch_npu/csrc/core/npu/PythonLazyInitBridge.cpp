#include "torch_npu/csrc/core/npu/PythonLazyInitBridge.h"

#include <atomic>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

namespace c10_npu::impl {
namespace {
std::atomic<PythonLazyInitFn> python_lazy_init{nullptr};
} // namespace

void registerPythonLazyInit(PythonLazyInitFn fn) {
  TORCH_CHECK(fn != nullptr);
  static c10::once_flag register_once;
  c10::call_once(register_once, [fn] { python_lazy_init.store(fn, std::memory_order_release); });
}

void runPythonLazyInit() {
  auto fn = python_lazy_init.load(std::memory_order_acquire);
  if (fn != nullptr) {
    fn();
  }
}
} // namespace c10_npu::impl
