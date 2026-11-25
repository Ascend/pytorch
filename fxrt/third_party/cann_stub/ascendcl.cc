// Link-only stubs. Never use this library at runtime.
#include "../cann/include/acl/acl_base_rt.h"
#include "../cann/include/acl/acl_rt.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

extern "C" auto aclrtGetSocName() -> const char * {
  __builtin_trap();
}
extern "C" auto aclInit(const char *) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtSetDevice(int32_t) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtGetDevice(int32_t *) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtMalloc(void **, size_t, aclrtMemMallocPolicy) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtFree(void *) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtMemcpy(void *, size_t, const void *, size_t, aclrtMemcpyKind) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtMemcpyAsync(void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtCreateStream(aclrtStream *) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtDestroyStream(aclrtStream) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtSynchronizeStream(aclrtStream) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtSetStreamResLimit(aclrtStream, aclrtDevResLimitType, uint32_t) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtUseStreamResInCurrentThread(aclrtStream) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtCreateNotify(aclrtNotify *, uint64_t) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtRecordNotify(aclrtNotify, aclrtStream) -> aclError {
  __builtin_trap();
}
extern "C" auto aclrtWaitAndResetNotify(aclrtNotify, aclrtStream, uint32_t) -> aclError {
  __builtin_trap();
}

extern "C" auto aclrtGetHardwareSyncAddr(void **) -> aclError {
  __builtin_trap();
}

extern "C" auto aclrtGetStreamResLimit(aclrtStream, aclrtDevResLimitType, uint32_t *) -> aclError {
  __builtin_trap();
}
