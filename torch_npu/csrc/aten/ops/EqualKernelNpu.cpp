#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

bool NPUNativeFunctions::equal(const at::Tensor& self, const at::Tensor& other) {
  // check the shape of self and other
  if(self.sizes() != other.sizes()) {
    return false;
  }

  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "Expected object of scalar type ",
      self.scalar_type(),
      ", but got ",
      other.scalar_type(),
      " for argument #2 'other' in call to equal_npu");

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("TensorEqual")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();

  return result.item().to<bool>();
}
} // namespace native
} // namespace at_npu