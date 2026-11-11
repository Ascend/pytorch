#include "torch_npu/csrc/distributed/PythonDebugBridge.h"

#include <atomic>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

namespace c10d_npu {
namespace {
std::atomic<PythonTracebackDumpFn> python_traceback_dump{nullptr};
} // namespace

void registerPythonTracebackDump(PythonTracebackDumpFn fn) {
  TORCH_CHECK(fn != nullptr);
  static c10::once_flag register_once;
  c10::call_once(register_once, [fn] { python_traceback_dump.store(fn, std::memory_order_release); });
}

bool runPythonTracebackDump(int rank, const std::string& prefix) {
  auto fn = python_traceback_dump.load(std::memory_order_acquire);
  return fn != nullptr ? fn(rank, prefix) : false;
}
} // namespace c10d_npu
