#pragma once

#include <string>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10d_npu {
using PythonTracebackDumpFn = bool (*)(int rank, const std::string& prefix);
C10_NPU_API void registerPythonTracebackDump(PythonTracebackDumpFn fn);
C10_NPU_API bool runPythonTracebackDump(int rank, const std::string& prefix);
} // namespace c10d_npu
