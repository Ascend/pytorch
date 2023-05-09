#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::npu_sub_sample(
    const at::Tensor &self, 
    int64_t per_images,
    double positive_fraction) {

  at::Tensor result = OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("SubSample")
      .Input(self)
      .Output(result)
      .Attr("batch_size_per_images", per_images)
      .Attr("positive_fraction", (float)positive_fraction)
      .Run();
  return result;
}
} // namespace native
} // namespace at_npu