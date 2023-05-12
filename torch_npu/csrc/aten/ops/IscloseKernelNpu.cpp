#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu { 
namespace native {
namespace {

at::Tensor& isclose_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    double rtol,
    double atol,
    bool equal_nan,
    at::Tensor& result) {
  auto rtol1 = static_cast<float>(rtol);
  auto atol1 = static_cast<float>(atol);
  OpCommand cmd;
  cmd.Name("IsClose")
      .Input(self)
      .Input(other)
      .Attr("rtol", rtol1)
      .Attr("atol", atol1)
      .Attr("equal_nan", equal_nan)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor NPUNativeFunctions::isclose(
    const at::Tensor& self,
    const at::Tensor& other,
    double rtol, 
    double atol, 
    bool equal_nan) {
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kBool), self);
  result = isclose_nocheck(self, other, rtol, atol, equal_nan, result);
  return result;  
}
} // namespace native
} // namespace at_npu
