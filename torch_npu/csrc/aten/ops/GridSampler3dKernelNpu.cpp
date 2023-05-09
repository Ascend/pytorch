#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& grid_sampler_3d_npu_nocheck(
    const at::Tensor& self, 
    const at::Tensor& grid,
    std::string interMode,
    std::string paddingMode,
    bool align_corners,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("GridSampler3D")
      .Input(self)
      .Input(grid)
      .Output(result)
      .Attr("interpolation_mode", interMode)
      .Attr("padding_mode", paddingMode)
      .Attr("align_corners", align_corners)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::grid_sampler_3d(
    const at::Tensor& self, 
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  TORCH_CHECK(
      (0 <= interpolation_mode && interpolation_mode <= 2),
      "interpolation_mode must be in range [0~2].")
  TORCH_CHECK(
      (0 <= padding_mode && padding_mode <= 2),
      "padding_mode must be in range [0~2].")
  at::Tensor formatCastOfSelf = self;
  at::Tensor formatCastOfGrid = grid;
  if (formatCastOfSelf.scalar_type() == at::ScalarType::Half) {
    formatCastOfSelf = NPUNativeFunctions::npu_dtype_cast(formatCastOfSelf, at::ScalarType::Float);
  }
  if (formatCastOfGrid.scalar_type() == at::ScalarType::Half) {
    formatCastOfGrid = NPUNativeFunctions::npu_dtype_cast(formatCastOfGrid, at::ScalarType::Float);
  }

  // calculate the output size
  c10::SmallVector<int64_t, SIZE> outputSize = {formatCastOfSelf.size(0),
                                           formatCastOfSelf.size(1),
                                           formatCastOfGrid.size(1),
                                           formatCastOfGrid.size(2),
                                           formatCastOfGrid.size(3)};

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, formatCastOfSelf.options(), ACL_FORMAT_ND);
  std::string interMode[] = {"bilinear", "nearest", "bicubic"};
  std::string paddingMode[] = {"zeros", "border", "reflection"};

  // calculate the output result of the NPU
  grid_sampler_3d_npu_nocheck(
      formatCastOfSelf,
      formatCastOfGrid,
      interMode[interpolation_mode],
      paddingMode[padding_mode],
      align_corners,
      result);

  at::ScalarType selfScalarType(self.scalar_type());
  if (result.scalar_type() != selfScalarType) {
    result = NPUNativeFunctions::npu_dtype_cast(result, selfScalarType);
  }
  return result;
}
} // namespace native
} // namespace at_npu