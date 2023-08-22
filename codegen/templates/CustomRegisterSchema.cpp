#include <torch/library.h>

namespace at_npu {

namespace native {


namespace {

TORCH_LIBRARY_FRAGMENT(aten, m) {

  ${custom_function_registrations}
}

} // anonymous namespace

} // namespace native

} // namespace at_npu
