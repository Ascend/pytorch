#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& take_out_nocheck(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  at::Tensor input_tensor = self.reshape(-1);
  at::Tensor contiguousSelf = NpuUtils::format_contiguous(input_tensor);
  at::Tensor contiguousIndex = NpuUtils::format_contiguous(index);

  OpCommand cmd;
  cmd.Name("Gather")
      .Input(contiguousSelf)
      .Input(contiguousIndex)
      .Output(result)
      .Attr("validate_indices", false)
      .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);

  OpPreparation::CheckOut(
      {self, index},
      result,
      self,
      outputSize);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    take_out_nocheck(self, index, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    take_out_nocheck(self, index, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::take(const at::Tensor& self, const at::Tensor& index) {
  at::Tensor result = OpPreparation::ApplyTensor(self, index.sizes());
  take_out_nocheck(self, index, result);
  return result;
}
} // namespace native
} // namespace at_npu
