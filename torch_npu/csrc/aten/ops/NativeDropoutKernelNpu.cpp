#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::native_dropout(
    const at::Tensor& input,
    double p,
    c10::optional<bool> train) {
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }

  bool dropout_train = !train.has_value() ? true : train.value();

  at::TensorOptions options = input.options();
  if (p == 0 || !dropout_train) {
    at::Tensor mask = NPUNativeFunctions::ones(
        input.sizes(),
        at::kBool,
        options.layout(),
        options.device(),
        options.pinned_memory());
    return std::make_tuple(input.clone(), mask);
  }
  if (p == 1) {
    at::Tensor output = at::zeros(input.sizes(), options);
    at::Tensor mask = at::zeros(input.sizes(), options.dtype(at::kBool));
    return std::make_tuple(output, mask);
  }

  return NPUNativeFunctions::_npu_dropout(input, p);
}

at::Tensor NPUNativeFunctions::native_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  double p = (scale == 0.0) ? 1 : (1 - 1 / scale);
  at::TensorOptions options = grad_output.options();
  if (p == 0) {
    return grad_output;
  }
  if (p == 1) {
    return at::zeros(grad_output.sizes(), options);
  }
  return NPUNativeFunctions::npu_dropout_backward(grad_output, mask, p);
}

} // namespace native
} // namespace at_npu
