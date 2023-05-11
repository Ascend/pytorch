#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{

at::Tensor softmax_backward_out_npu(
    at::Tensor &grad_input,
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype) {
  c10::SmallVector<int64_t, N> dimList = {dim};
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("SoftmaxGrad")
      .Input(output)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("axes", dimList)
      .Run();

  return grad_input;
}

at::Tensor& NPUNativeFunctions::_softmax_backward_data_out(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype,
    at::Tensor &result) {
  OpPreparation::CheckOut(
      {grad_output, output},
      result,
      grad_output);

  // calculate the output result of the NPU
  softmax_backward_out_npu(result, grad_output, output, dim, input_dtype);

  return result;
}

at::Tensor NPUNativeFunctions::_softmax_backward_data(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    int64_t dim,
    at::ScalarType input_dtype) {
  // calculate the output size
  auto outputSize = input_same_output_size(grad_output);

  // output'format must be same with grad_output
  at::Tensor temp_output = output;
  if (CalcuOpUtil::GetTensorNpuFormat(temp_output) == ACL_FORMAT_NC1HWC0) {
    NPUNativeFunctions::npu_format_cast_(temp_output, CalcuOpUtil::GetTensorNpuFormat(grad_output));
  }

  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(temp_output, outputSize);

  // calculate the output result of the NPU
  softmax_backward_out_npu(grad_input, grad_output, temp_output, dim, input_dtype);

  return grad_input;
}

} // namespace native
} // namespace at_npu