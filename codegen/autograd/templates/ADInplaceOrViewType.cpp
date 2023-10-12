#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include <torch/library.h>

#include "torch_npu/csrc/aten/CustomRedispatch.h"

// ${generated_comment}
$ops_headers

using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace at_npu {

namespace ADInplaceOrView {

namespace {
${inplace_or_view_method_definitions}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(npu, ADInplaceOrView, m) {
  ${inplace_or_view_wrapper_registrations};
}

}  // namespace
} // namespace at_npu
