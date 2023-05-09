#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_prelu_kernel(const at::Tensor& self, const at::Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  // calculate the output size
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  
  OpCommand cmd;
  cmd.Name("PRelu")
     .Input(self)
     .Input(weight)
     .Output(result)
     .Run();
  return result;
}
} // namespace native
} // namespace at_npu