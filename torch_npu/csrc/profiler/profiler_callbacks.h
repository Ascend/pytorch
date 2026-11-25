#pragma once

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {

namespace python_tracer {

enum class Command { kStartOne = 0, kStartAll, kStop, kStopOne, kClear };
using CallFn = void (*)(Command);
C10_NPU_API void registerFunctions(CallFn call);

void call(Command c);

} // namespace python_tracer

} // namespace profiler
} // namespace torch_npu
