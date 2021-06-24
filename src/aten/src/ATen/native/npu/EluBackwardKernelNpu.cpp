#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;
SmallVector<NPUTensorDesc, N> elu_backward_npu_input(const Tensor& grads,const Tensor& activations) {
  auto inputs = CalcuOpUtil::create_npu_input_tensor_desc({grads, activations});
  return inputs;
}

SmallVector<NPUTensorDesc, N> elu_backward_npu_output(const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> elu_backward_npu_attr(Scalar alpha) {
  float value = CalcuOpUtil::get_scalar_float_value(alpha);
  NPUAttrDesc npuAttrScalar = NPUAttrDesc("alpha", value);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrScalar};
  return attrs;
}

Tensor& elu_backward_out_npu(Tensor& grad_input, const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor& output) {
    auto inputs = elu_backward_npu_input(grad_output, output);
    auto outputs = elu_backward_npu_output({grad_input});
    auto attrs = elu_backward_npu_attr(alpha);
    CalcuOpUtil::execute_npu_operate("EluGradV2", inputs, outputs, attrs);
    return grad_input;
}
Tensor elu_backward_npu(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor& output) {
    // calculate the output size
    auto outputSize = input_same_output_size(grad_output);
    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(outputSize, grad_output.options(), CalcuOpUtil::get_tensor_npu_format(grad_output));
    // calculate the output result of the NPU
    elu_backward_out_npu(result, grad_output, alpha, scale, input_scale, output);
    return result;
}
} // namespace native
} // namespace at
