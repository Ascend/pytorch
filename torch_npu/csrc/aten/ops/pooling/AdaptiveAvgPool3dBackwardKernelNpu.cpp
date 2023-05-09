#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


int64_t adaptive_avg_pool3d_backward_safe_size(const at::Tensor& self){
  c10::SmallVector<int64_t, N> dims = {-3, -2, -1};
  int64_t size = 1;
  if (self.sizes().empty()) {
     return size;
  }
  for (int64_t ndim : dims) {
    ndim = CalcuOpUtil::MakeWrapDim(ndim, self.sizes().size());
    size *= self.sizes()[ndim];
  }
  return size;
}

at::Tensor& NPUNativeFunctions::adaptive_avg_pool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& result){
  if (grad_output.size(grad_output.dim() - 3) == 1 && grad_output.size(grad_output.dim() - 2) == 1 &&
        grad_output.size(grad_output.dim() - 1) == 1){
    result.fill_(1.0 / adaptive_avg_pool3d_backward_safe_size(self));
    result.mul_(grad_output);
  } else {
    TORCH_CHECK(false,
        "adaptive_avg_pool3d_backward only support D=1 && H=1 && W=1 current!");
  }
  return result;
}

at::Tensor NPUNativeFunctions::_adaptive_avg_pool3d_backward(const at::Tensor& grad_output, const at::Tensor& self){
  at::Tensor result = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::adaptive_avg_pool3d_backward_out(grad_output, self, result);
  return result;
}

} // native
} // at_npu
