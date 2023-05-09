#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::glu_out(const at::Tensor& self, int64_t dim, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional at::Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);
  
  auto chunkedInput = self.chunk(2, dim);
  at::Tensor firstHalf = chunkedInput[0];
  at::Tensor secondHalf = chunkedInput[1];
  result = firstHalf.mul(secondHalf.sigmoid());
  return result;
}

at::Tensor NPUNativeFunctions::glu(const at::Tensor& self, int64_t dim) {
  auto outputSize = glu_npu_output_size(self, dim);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeFunctions::glu_out(self, dim, result);
  return result;
  
}
}  // namespace native
}  // namespace at_npu