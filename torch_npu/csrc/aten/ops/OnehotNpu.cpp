#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& one_hot_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t axis,
    int64_t depth,
    at::Scalar on_value,
    at::Scalar off_value) {
  at::Tensor selfCp = NPUNativeFunctions::npu_dtype_cast(self, at::kInt);
  at::Tensor on_tmp = OpPreparation::ApplyTensor(
      {1},
      selfCp.options().dtype(at::ScalarType::Float),
      selfCp)
      .fill_(on_value);
  at::Tensor off_tmp = OpPreparation::ApplyTensor(
      {1},
      selfCp.options().dtype(at::ScalarType::Float),
      selfCp)
      .fill_(off_value);
  OpCommand cmd;
  cmd.Name("OneHotD")
      .Input(selfCp)
      .Input(on_tmp)
      .Input(off_tmp)
      .Output(result)
      .Attr("axis", axis)
      .Attr("depth", depth)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::npu_one_hot(
    const at::Tensor& self,
    int64_t axis,
    int64_t depth,
    const at::Scalar& on_value,
    const at::Scalar& off_value) {
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);

  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      self.options().dtype(at::ScalarType::Float),
      self);
  one_hot_out_npu(result, self, axis, depth, on_value, off_value);

  return result;
}
} // namespace native
} // namespace at_npu