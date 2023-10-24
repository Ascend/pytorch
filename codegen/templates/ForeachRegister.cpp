#include <torch/library.h>

${include_headers}

namespace {

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

// (Ascend) TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  ${foreach_kernel}
}

}
