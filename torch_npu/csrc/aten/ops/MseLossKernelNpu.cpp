#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& mse_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  if (self.numel() == 0 || target.numel() == 0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }
  auto unified_result = OpPreparation::binary_op_check(result, self, target, true);
  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));
  OpCommand cmd;
  cmd.Name("MseLoss")
      .Expect(unified_result)
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::mse_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }

  OpPreparation::CheckOut(
      {self, target},
      result,
      self,
      outputSize);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

at::Tensor NPUNativeFunctions::mse_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  at::Tensor result =
      reduction == at::Reduction::None ?
      OpPreparation::ApplyTensor(self, outputSize) :
      OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_ND);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

} // namespace native
} // namespace at_npu