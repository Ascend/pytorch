#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &img_to_tensor_out(const at::Tensor &self, at::Tensor &result)
{
  OpCommand cmd;
  cmd.Name("ImgToTensor")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::img_to_tensor(const at::Tensor &self)
{
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kFloat),
      CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  img_to_tensor_out(self, result);

  return result;
}

} // namespace native
} // namespace at_npu