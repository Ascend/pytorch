#ifndef FXRT_ACLNN_SYMBOL_API_H
#define FXRT_ACLNN_SYMBOL_API_H

#include <stdint.h>

#if defined(__GNUC__)
#define FXRT_ACLNN_API __attribute__((visibility("default")))
#else
#define FXRT_ACLNN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Same-wheel ABI between the ABI=1 common SO and ABI=0 converter. No ownership
// transfers: addresses remain valid while the existing FXRT loader is alive.
// Return 0 on success, nonzero on failure; error is optional and NUL terminated
// when error_capacity > 0. C++ exceptions never cross this boundary.
FXRT_ACLNN_API int32_t FxrtAclnnEnsureInitializedV1(char* error, uint64_t error_capacity);

// Resolve a complete group through the existing cache/lookup policy. On failure
// all count output slots are cleared. count==0 accepts null input/output arrays.
// Resolving addresses alone does not initialize ACLNN or the NPU device.
FXRT_ACLNN_API int32_t
FxrtAclnnResolveV1(const char* const* names, uint64_t count, void** functions, char* error, uint64_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif // FXRT_ACLNN_SYMBOL_API_H
