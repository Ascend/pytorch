#include <torch/csrc/python_headers.h>

#include "torch_npu/csrc/core/npu/PythonGilBridge.h"
#include "torch_npu/csrc/utils/PythonGilAdapter.h"

namespace torch_npu::utils {
namespace {
void* releaseIfHeld() {
  return PyGILState_Check() ? PyEval_SaveThread() : nullptr;
}

void restore(void* state) {
  PyEval_RestoreThread(static_cast<PyThreadState*>(state));
}

void runWithGilReleased(c10::function_ref<void()> fn) {
  PyThreadState* state = nullptr;
  if (PyGILState_Check()) {
    state = PyEval_SaveThread();
  }
  fn();
  if (state != nullptr) {
    PyEval_RestoreThread(state);
  }
}

const c10_npu::impl::PythonGilHooks hooks{releaseIfHeld, restore, runWithGilReleased};
} // namespace

void registerPythonGilAdapter() {
  c10_npu::impl::registerPythonGilHooks(&hooks);
}
} // namespace torch_npu::utils
