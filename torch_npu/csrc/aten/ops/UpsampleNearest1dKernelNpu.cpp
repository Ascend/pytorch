#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_npu_output_size(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t W = output_size[0];
  outputSize = {N, C, 1, W};
  return outputSize;
}

at::Tensor& upsample_nearest1d_out_nocheck(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& result) {
  TORCH_CHECK(
      (self.size(1) != 0 && self.size(2) != 0) && self.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      self.sizes());

  at::Tensor self_cp = self.unsqueeze(2);
  OpCommand cmd;
  if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf) {
    c10::SmallVector<int64_t, SIZE> result_size = {1, output_size[0]};
    cmd.Name("ResizeNearestNeighborV2")
        .Input(self_cp)
        .Input(result_size, at::kInt)
        .Output(result)
        .Attr("align_corners", false)
        .Attr("half_pixel_centers", false)
        .Run();
  } else {
    cmd.Name("Resize")
        .Input(self_cp)
        .Input(output_size, at::kFloat)
        .Input(output_size, at::kFloat)
        .Input(result.sizes(), at::kLong)
        .Output(result)
        .Attr("mode", (string)"nearest")
        .Attr("nearest_mode", (string)"floor")
        .Attr("coordinate_transformation_mode", (string)"pytorch_half_pixel")
        .Run();
  }
  result = result.squeeze(2);
  return result;
}

at::Tensor& NPUNativeFunctions::upsample_nearest1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(self, output_size);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    upsample_nearest1d_out_nocheck(self, output_size, scales, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    upsample_nearest1d_out_nocheck(self, output_size, scales, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  upsample_nearest1d_out_nocheck(self, output_size, scales, result);
  return result;
}
} // namespace native
} // namespace at_npu
