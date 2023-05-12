#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::multilabel_margin_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& output,
    at::Tensor& is_target) {

  OpPreparation::CheckMemory({self, target}, {output, is_target});
  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("MultilabelMarginLoss")
    .Input(self)
    .Input(target)
    .Output(output)
    .Output(is_target)
    .Attr("reduction", reductionStr)
    .Run();
  return std::tuple<at::Tensor&, at::Tensor&>(output, is_target);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::multilabel_margin_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  int64_t nframe;
  if (self.dim() <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }
  if (reduction == at::Reduction::None) {
    outputSize = {nframe};
  }
  auto output = OpPreparation::ApplyTensor(self, outputSize);
  auto is_target = OpPreparation::ApplyTensor(target);

  NPUNativeFunctions::multilabel_margin_loss_forward_out(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

} // namespace native
} // namespace at_npu
