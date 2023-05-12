#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

static void check1d(
    const char* function_name,
    const char* argument_name,
    at::IntArrayRef x) {
  TORCH_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

at::Tensor NPUNativeFunctions::adaptive_avg_pool1d(const at::Tensor& self, at::IntArrayRef output_size) {
  at::checkDimRange("adaptive_avg_pool1d", at::TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  auto output = NPUNativeFunctions::adaptive_avg_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  return output.squeeze(-2);
}

} // namespace native
} // namespace at_npu