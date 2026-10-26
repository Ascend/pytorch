#pragma once

namespace torch_npu {
namespace profiler {

namespace python_tracer {

enum class Command { kStartOne = 0, kStartAll, kStop, kStopOne, kClear };
using CallFn = void (*)(Command);
void registerFunctions(CallFn call);

void call(Command c);

} // namespace python_tracer

} // namespace profiler
} // namespace torch_npu
