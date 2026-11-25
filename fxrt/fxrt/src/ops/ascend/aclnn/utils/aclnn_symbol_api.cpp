#include "aclnn_symbol_api.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"

namespace {
void SetError(char* error, uint64_t capacity, const char* message) noexcept {
  if (error != nullptr && capacity != 0U) {
    (void)std::snprintf(error, capacity, "%s", message);
  }
}
} // namespace

extern "C" int32_t FxrtAclnnEnsureInitializedV1(char* error, uint64_t error_capacity) {
  SetError(error, error_capacity, "");
  try {
    fxrt::ops::LoadOpApiLib();
    fxrt::ops::AclnnInit();
    return 0;
  } catch (const std::exception& exception) {
    SetError(error, error_capacity, exception.what());
  } catch (...) {
    SetError(error, error_capacity, "Unknown exception while initializing FXRT ACLNN");
  }
  return 1;
}

extern "C" int32_t FxrtAclnnResolveV1(
    const char* const* names,
    uint64_t count,
    void** functions,
    char* error,
    uint64_t error_capacity) {
  SetError(error, error_capacity, "");
  if (functions != nullptr) {
    std::fill_n(functions, count, nullptr);
  }
  if (count == 0U) {
    return 0;
  }
  if (names == nullptr || functions == nullptr) {
    SetError(error, error_capacity, "ACLNN names and output arrays must not be null");
    return 1;
  }
  try {
    std::vector<void*> resolved(count, nullptr);
    for (uint64_t i = 0U; i < count; ++i) {
      if (names[i] == nullptr || names[i][0] == '\0') {
        SetError(error, error_capacity, "ACLNN symbol name must not be empty");
        return 1;
      }
      resolved[i] = fxrt::ops::GetAclnnOpApiFunc(names[i]);
      if (resolved[i] == nullptr) {
        if (error != nullptr && error_capacity != 0U) {
          (void)std::snprintf(error, error_capacity, "FXRT ACLNN symbol not found: %s", names[i]);
        }
        return 1;
      }
    }
    std::copy(resolved.begin(), resolved.end(), functions);
    return 0;
  } catch (const std::exception& exception) {
    SetError(error, error_capacity, exception.what());
  } catch (...) {
    SetError(error, error_capacity, "Unknown exception while resolving FXRT ACLNN symbols");
  }
  return 1;
}
