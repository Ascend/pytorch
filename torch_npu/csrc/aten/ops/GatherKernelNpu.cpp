#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& gather_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  if (self.scalar_type() == at::kLong) {
    TORCH_NPU_WARN_ONCE("The oprator of gather is executed, Currently High Accuracy but Low Performance OP"
      "with 64-bit has been used,Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }

  OpCommand cmd;
  cmd.Name("GatherElements")
      .Input(self)
      .Input(index)
      .Attr("dim", dim)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  auto outputSize = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  return gather_out_npu_nocheck(self, dim, index, sparse_grad, result);
}

at::Tensor& NPUNativeFunctions::gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  auto outputSize = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  return gather_out_npu_nocheck(self, dimname_to_position(self, dim), index, sparse_grad, result);
}

at::Tensor NPUNativeFunctions::gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  auto outputSize = input_same_output_size(index);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  gather_out_npu_nocheck(self, dim, index, sparse_grad, result);
  return result;
}

at::Tensor NPUNativeFunctions::gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad) {
  return gather(self, dimname_to_position(self, dim), index, sparse_grad);
}
} // namespace native
} // namespace at_npu