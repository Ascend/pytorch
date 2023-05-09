#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::conv_tbc_backward(
    const at::Tensor& self,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int64_t pad) {
  // construct the output tensor of the NPU
  auto output = NPUNativeFunctions::npu_conv2d_backward(
      input.permute({1, 2, 0}).unsqueeze(2),
      self.permute({1, 2, 0}).unsqueeze(2),
      weight.permute({2, 1, 0}).unsqueeze(2),
      {1, 1},
      {0, pad},
      {1, 1},
      1,
      {1, 1, 1});

  return std::make_tuple(
      std::move((std::get<0>(output)).squeeze(2).permute({2, 0, 1})), 
      std::move((std::get<1>(output)).squeeze(2).permute({2, 1, 0})), 
      std::move(std::get<2>(output)));
}
} // namespace native
} // namespace at_npu