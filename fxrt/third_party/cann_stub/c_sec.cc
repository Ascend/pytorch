// Link-only stubs. Never use this library at runtime.
#include "securec.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

extern "C" auto memset_s(void *, size_t, int, size_t) -> errno_t {
  __builtin_trap();
}
extern "C" auto vsnprintf_s(char *, size_t, size_t, const char *, va_list) -> int {
  __builtin_trap();
}

extern "C" auto memcpy_s(void *, size_t, const void *, size_t) -> errno_t {
  __builtin_trap();
}
