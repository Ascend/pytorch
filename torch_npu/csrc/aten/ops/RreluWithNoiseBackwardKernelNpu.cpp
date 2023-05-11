#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::rrelu_with_noise_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self_or_result,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    bool is_result) {
  auto minimum = 1E-6;
  auto folat_lower = lower.toFloat();
  auto float_upper = upper.toFloat();
  if (training && (float_upper - folat_lower > minimum)) {
    return grad_output.mul(noise);
  } else {
    at::Scalar negative_slope = (folat_lower + float_upper) / 2;
    return NPUNativeFunctions::leaky_relu_backward(grad_output, self_or_result, negative_slope, is_result);
  }
}

} // namespace native
} // namespace at_npu