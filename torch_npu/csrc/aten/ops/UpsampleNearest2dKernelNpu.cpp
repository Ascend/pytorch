#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> upsample_nearest2d_npu_output_size(
    const at::Tensor& input,
    at::IntArrayRef output_size){
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  at::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  return outputSize;
}

at::Tensor& NPUNativeFunctions::upsample_nearest2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  if (!result.sizes().equals(outputSize)){
    result.resize_(outputSize);
  }
  at::SmallVector<int64_t,N> outputSizeVec = array_to_small_vector(output_size);
  OpCommand cmd;
  cmd.Name("ResizeNearestNeighborV2")
    .Input(self, "x", ACL_FORMAT_NCHW)
    .Input(outputSizeVec, at::kInt)
    .Output(result, "y", ACL_FORMAT_NCHW)
    .Attr("align_corners", false)
    .Attr("half_pixel_centers", false)
    .Run();
  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest2d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeFunctions::upsample_nearest2d_out(self, output_size, scales_h, scales_w, result);

  return result;
}

} // namespace native
} // namespace at_npu