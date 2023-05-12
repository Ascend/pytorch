#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &reverse_out(
    const at::Tensor &self,
    at::IntArrayRef axis,
    at::Tensor &result)
{
  OpCommand cmd;
  cmd.Name("ReverseV2")
      .Input(self)
      .Input(axis, at::kInt)
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::reverse(
    const at::Tensor &self,
    at::IntArrayRef axis)
{
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  reverse_out(self, axis, result);

  return result;
}

} // namespace native
} // namespace at_npu