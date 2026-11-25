// Link-only stubs. Never use this library at runtime.
#include "../cann/include/ge/ge_api_v2.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace ge {
auto GEInitializeV2(const std::map<AscendString, AscendString> &) -> Status {
  __builtin_trap();
}
GeSession::GeSession(const std::map<AscendString, AscendString> &) {
  __builtin_trap();
}
GeSession::~GeSession() {
  __builtin_trap();
}
auto GeSession::AddGraph(uint32_t, const Graph &) -> Status {
  __builtin_trap();
}
auto GeSession::AddGraph(uint32_t, const Graph &, const std::map<AscendString, AscendString> &) -> Status {
  __builtin_trap();
}
auto GeSession::RemoveGraph(uint32_t) -> Status {
  __builtin_trap();
}
auto GeSession::CompileGraph(uint32_t) -> Status {
  __builtin_trap();
}
auto GeSession::LoadGraph(const uint32_t, const std::map<AscendString, AscendString> &, void *) const -> Status {
  __builtin_trap();
}
auto GeSession::RunGraphWithStreamAsync(uint32_t, void *, const std::vector<gert::Tensor> &, std::vector<gert::Tensor> &) -> Status {
  __builtin_trap();
}
auto GeSession::RegisterExternalAllocator(const void *const, AllocatorPtr) const -> Status {
  __builtin_trap();
}
auto GeSession::UnregisterExternalAllocator(const void *const) const -> Status {
  __builtin_trap();
}
}  // namespace ge
