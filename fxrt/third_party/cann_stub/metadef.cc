// Link-only stubs. Never use this library at runtime.
#include "../cann/include/exe_graph/runtime/compute_node_info.h"
#include "../cann/include/exe_graph/runtime/runtime_attrs.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace gert {
auto RuntimeAttrs::GetPointerByIndex(size_t) const -> const void * {
  __builtin_trap();
}
auto ComputeNodeInfo::MutableAttrs() const -> RuntimeAttrs * {
  __builtin_trap();
}
auto ComputeNodeInfo::CalcSize(const size_t, const size_t, const size_t, const size_t, size_t &) -> ge::graphStatus {
  __builtin_trap();
}
auto ComputeNodeInfo::Init(const size_t, const size_t, const size_t, const size_t, const size_t, const ge::char_t *, const ge::char_t *) -> void {
  __builtin_trap();
}
}  // namespace gert
