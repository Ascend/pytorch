#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor NPUNativeFunctions::npu_fast_gelu(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("FastGelu")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

namespace {
at::Tensor& fast_gelu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self) {
  // constructs the input and output NPUTensorDesc
  OpCommand cmd;
  cmd.Name("FastGeluGrad")
    .Input(grad)
    .Input(self)
    .Output(grad_input)
    .Run();

  return grad_input;
}
}

at::Tensor NPUNativeFunctions::npu_fast_gelu_backward(
    const at::Tensor& grad, 
    const at::Tensor& self) {
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  fast_gelu_backward_npu_nocheck(grad_input, grad, self);
  
  return grad_input;
}

at::Tensor NPUNativeFunctions::fast_gelu(const at::Tensor& self) {
    return NPUNativeFunctions::npu_fast_gelu(self);
}

} // namespace native
} // namespace at_npu