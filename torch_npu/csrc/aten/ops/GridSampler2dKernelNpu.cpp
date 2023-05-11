#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::grid_sampler_2d(
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

  at::Tensor dtypeCastOfSelf = self;
  at::Tensor dtypeCastOfGrid = grid;
  if (dtypeCastOfSelf.scalar_type() == c10::ScalarType::Half) {
    dtypeCastOfSelf = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfSelf, c10::ScalarType::Float);
  }
  if (dtypeCastOfGrid.scalar_type() == c10::ScalarType::Half) {
    dtypeCastOfGrid = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfGrid, c10::ScalarType::Float);
  }

  c10::SmallVector<int64_t, SIZE> outputSize = {dtypeCastOfSelf.size(0),
                                                dtypeCastOfSelf.size(1),
                                                dtypeCastOfGrid.size(1),
                                                dtypeCastOfGrid.size(2)};

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(dtypeCastOfSelf, outputSize, ACL_FORMAT_ND);
  std::string interMode[] = {"bilinear", "nearest", "bicubic"};
  std::string paddingMode[] = {"zeros", "border", "reflection"};
  OpCommand cmd;
  cmd.Name("GridSampler2D")
      .Input(dtypeCastOfSelf)
      .Input(dtypeCastOfGrid)
      .Output(result)
      .Attr("interpolation_mode", interMode[interpolation_mode])
      .Attr("padding_mode", paddingMode[padding_mode])
      .Attr("align_corners", align_corners)
      .Run();

  c10::ScalarType selfScalarType(self.scalar_type());
  if (result.scalar_type() != selfScalarType) {
    result = NPUNativeFunctions::npu_dtype_cast(result,selfScalarType);
  }
  return result;
}
} // namespace native
} // namespace at_npu
