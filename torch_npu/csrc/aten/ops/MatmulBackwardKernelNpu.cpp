#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& other,
    std::array<bool,2> mask) {
  if (!grad.defined()) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }
  at::Tensor grad_self, grad_other;
  //check grad input mask
  if (mask[0]) {
    grad_self = NPUNativeFunctions::matmul(grad, other.transpose(-1, -2));
  }
  if (mask[1]) {
    grad_other = NPUNativeFunctions::matmul(self.transpose(-1, -2), grad);
  }
  return std::make_tuple(grad_self, grad_other);
}

} // namespace native
} // namespace at_npu
