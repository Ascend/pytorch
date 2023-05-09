#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& pad_npu_nocheck(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef paddings) {
  c10::SmallVector<int64_t, N> paddingsVector = array_to_small_vector(paddings);
  paddingsVector.resize(2 * input.dim(), 0);

  OpCommand cmd;
  cmd.Name("Pad")
      .Input(input)
      .Input(paddingsVector)
      .Output(output)
      .Run();
  return output;
}

at::Tensor NPUNativeFunctions::npu_pad(const at::Tensor& input, at::IntArrayRef paddings) {
  auto outputSize = pad_npu_output_size(input, paddings);
  at::Tensor output = OpPreparation::ApplyTensor(input, outputSize);
  pad_npu_nocheck(output, input, paddings);
  return output;
}

} // namespace native
} // namespace at_npu