#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& smooth_l1_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  if (self.numel()==0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));
  OpCommand cmd;
  cmd.Name("SmoothL1LossV2")
    .Input(self)
    .Input(target)
    .Output(result)
    .Attr("reduction", reductionStr)
    .Attr("sigma", static_cast<float>(beta))
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::smooth_l1_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& result) {
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);

  OpPreparation::CheckOut(
      {self, target},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      outputSize);

  OpPreparation::CheckMemory({self, target}, {result});
  smooth_l1_loss_out_npu_nocheck(result, self, target, reduction, beta);
  return result;
}

at::Tensor NPUNativeFunctions::smooth_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  smooth_l1_loss_out_npu_nocheck(result, self, target, reduction, beta);
  return result;
}

} // namespace native
} // namespace at_npu