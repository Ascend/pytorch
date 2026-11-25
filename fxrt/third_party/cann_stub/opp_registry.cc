// Link-only stubs. Never use this library at runtime.
#include "../cann/include/base/registry/op_impl_space_registry_v2.h"
#include "../cann/include/base/registry/opp_package_utils.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace gert { class OppSoDescImpl {}; }
namespace gert {
OppSoDesc::OppSoDesc(const std::vector<ge::AscendString> &, const ge::AscendString &) {
  __builtin_trap();
}
OppSoDesc::~OppSoDesc() {
  __builtin_trap();
}
OpImplSpaceRegistryV2::OpImplSpaceRegistryV2() {
  __builtin_trap();
}
auto OpImplSpaceRegistryV2::AddSoToRegistry(const OppSoDesc &) -> ge::graphStatus {
  __builtin_trap();
}
auto OpImplSpaceRegistryV2::GetOpImpl(const char_t *) const -> const OpImplKernelRegistry::OpImplFunctionsV2 * {
  __builtin_trap();
}
auto OpImplSpaceRegistryV2::CreateOrGetOpImpl(const char_t *) -> OpImplKernelRegistry::OpImplFunctionsV2 * {
  __builtin_trap();
}
}  // namespace gert
