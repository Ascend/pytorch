#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& NPUNativeFunctions::adaptive_avg_pool3d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& result) {
  // reuse the mean out when d,h,w=1
  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
    at::mean_out(result, self, {self.dim() - 3, self.dim() - 2, self.dim() - 1}, true);
  } else {
   TORCH_CHECK(false,
               "adaptive_avg_pool3d only support D=1 && H=1 && W=1 current!");
  }
  return result;
}

at::Tensor NPUNativeFunctions::adaptive_avg_pool3d(const at::Tensor& self, at::IntArrayRef output_size) {
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pooling3d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (self.dim() == 4 || self.dim() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
  auto outputSize = adaptive_avg_pool3d_npu_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);

  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
    return at::mean(self, {self.dim() - 3, self.dim() - 2, self.dim() - 1}, true);
  } else {
   TORCH_CHECK(false,
               "adaptive_avg_pool3d only support D=1 && H=1 && W=1 current!");
  }
  return result;
}

at::Tensor NPUNativeFunctions::_adaptive_avg_pool3d(const at::Tensor& self, at::IntArrayRef output_size) {
  return NPUNativeFunctions::adaptive_avg_pool3d(self, output_size);
}

} // namespace native
} // namespace at_npu
