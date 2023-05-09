#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::soft_margin_loss_out(
    const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& result) {
  at::Tensor target_broadcast = target;
  if(target.sizes() != self.sizes()) {
    target_broadcast = NPUNativeFunctions::npu_broadcast(target, self.sizes());
  }

  OpPreparation::CheckMemory({self, target}, {result});
  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));
  OpCommand cmd;
  cmd.Name("SoftMarginLoss")
      .Input(self)
      .Input(target_broadcast)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::soft_margin_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
  auto outputSize = soft_margin_loss_npu_output_size(
      self,
      target,
      reduction);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  NPUNativeFunctions::soft_margin_loss_out(self, target, reduction, result);
  if (reduction == at::Reduction::None) {
    return result;
  } else {
    return result.reshape({});
  }
}

} // namespace native
} // namespace at_npu
