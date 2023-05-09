#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_random_choice_with_mask(
    const at::Tensor& self,
    int64_t count,
    int64_t seed,
    int64_t seed2) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Bool,
      "The input.dtype should be bool, but get",
      self.scalar_type());
  TORCH_CHECK(
      self.dim() <= 5 && self.dim() >= 1,
      "The input.dim should be in [1, 5], but get",
      self.dim());
  TORCH_CHECK(count > 0, "The count must greater than 0, but get", count);

  at::Tensor result = OpPreparation::ApplyTensor({count, self.dim()}, self.options().dtype(at::kInt), self);
  at::Tensor mask = OpPreparation::ApplyTensor(self, {count});
  OpCommand cmd;
  cmd.Name("RandomChoiceWithMask")
      .Input(self)
      .Output(result)
      .Output(mask)
      .Attr("count", count)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();

  return std::tie(result, mask);
}

} // namespace native
} // namespace at