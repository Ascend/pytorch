#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

void mask_fill_range_nocheck(
    const at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& value,
    int64_t axis){
  int64_t x_dim = self.dim();
  int64_t min = -x_dim;
  int64_t max = x_dim - 1;
  TORCH_CHECK(
      !(axis < min || axis > max),
      "axis overfloaw the range, expected in range [",
      -x_dim,
      " ",
      x_dim - 1,
      "] ");
  TORCH_CHECK(
      start.ndimension() == 2 && start.sizes() == end.sizes(),
      "Expected noempty 2D start tensor and start' sizes() should be equal end's sizes() ");
  TORCH_CHECK(
      start.size(0) == value.size(0),
      "Expected value.length equal start loop num ");
  TORCH_CHECK(
      self.scalar_type() == value.scalar_type(),
      "value dtype should be equal self dtype !, but value dtype is ",
      value.scalar_type(),
      " and self dtype is ",
      self.scalar_type());
}

at::Tensor NPUNativeFunctions::npu_masked_fill_range(
    const at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& value,
    int64_t axis){
  mask_fill_range_nocheck(self, start, end, value, axis);
  at::Tensor result = OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("MaskedFillRange")
      .Input(self)
      .Input(start)
      .Input(end)
      .Input(value)
      .Output(result)
      .Attr("axis", axis)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu
