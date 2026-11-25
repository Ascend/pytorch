// Link-only stubs. Never use this library at runtime.
#include "../cann/include/platform/platform_info.h"
#include "../cann/include/platform/platform_infos_def.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace fe {
auto PlatFormInfos::GetPlatformRes(const std::string &, const std::string &, std::string &) -> bool {
  __builtin_trap();
}
auto PlatFormInfos::GetPlatformResWithLock(const std::string &, const std::string &, std::string &) -> bool {
  __builtin_trap();
}
auto PlatFormInfos::GetCoreNum() const -> uint32_t {
  __builtin_trap();
}
auto PlatFormInfos::SetCoreNumByCoreType(const std::string &) -> void {
  __builtin_trap();
}
auto PlatformInfoManager::Instance() -> PlatformInfoManager & {
  __builtin_trap();
}
auto PlatformInfoManager::GeInstance() -> PlatformInfoManager & {
  __builtin_trap();
}
auto PlatformInfoManager::InitializePlatformInfo() -> uint32_t {
  __builtin_trap();
}
auto PlatformInfoManager::GetPlatformInfos(const std::string, PlatFormInfos &, OptionalInfos &) -> uint32_t {
  __builtin_trap();
}
auto PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatFormInfos &, OptionalInfos &) -> uint32_t {
  __builtin_trap();
}
auto PlatformInfoManager::GetRuntimePlatformInfosByDevice(const uint32_t &, PlatFormInfos &, bool) -> uint32_t {
  __builtin_trap();
}
}  // namespace fe
