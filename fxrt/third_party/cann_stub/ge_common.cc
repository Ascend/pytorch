// Link-only stubs. Never use this library at runtime.
// The GE-tree ge_api_error_codes.h declares Instance() out-of-line; any TU
// instantiating the header's ErrorNoRegisterar globals imports it. The
// runtime tree's copy defines it inline, so include the GE one here on purpose.
#include "../cann/include/ge/ge_api_error_codes.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace ge {
auto StatusFactory::Instance() -> StatusFactory * {
  __builtin_trap();
}
}  // namespace ge
