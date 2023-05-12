#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::glu_backward_out(const at::Tensor &grad_output, const at::Tensor &self, int64_t dim, at::Tensor &result) {
  auto outputSize = input_same_output_size(self);
  OpPreparation::CheckOut(
      {grad_output, self},
      result,
      grad_output,
      outputSize);

  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn); 
      
  auto chunkedInput = self.chunk(2, dim);
  at::Tensor firstHalf = chunkedInput[0];
  at::Tensor secondHalf = chunkedInput[1];

  secondHalf = secondHalf.sigmoid();

  at::Tensor gradFirst = secondHalf.mul(grad_output);

  at::Tensor gradSecond = firstHalf.mul(secondHalf).mul_(1-secondHalf).mul_(grad_output);

  result = at::cat({gradFirst, gradSecond}, dim);
  return result;
}

at::Tensor NPUNativeFunctions::glu_backward(const at::Tensor &grad_output, const at::Tensor &self, int64_t dim) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeFunctions::glu_backward_out(grad_output, self, dim, result);
  return result;
  
}
}  // namespace native
}  // namespace at_npu