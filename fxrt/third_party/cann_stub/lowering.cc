// Link-only stubs. Never use this library at runtime.
#include "../cann/include/exe_graph/runtime/eager_op_execution_context.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace gert {
auto EagerOpExecutionContext::GetStream() const -> rtStream {
  __builtin_trap();
}
auto EagerOpExecutionContext::MallocOutputTensor(size_t, const StorageShape &, const StorageFormat &, ge::DataType) -> Tensor * {
  __builtin_trap();
}
auto EagerOpExecutionContext::MakeOutputRefInput(size_t, size_t) const -> Tensor * {
  __builtin_trap();
}
}  // namespace gert
