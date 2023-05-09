#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& embedding_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& weight,
    const at::Tensor& indices) {
  c10::SmallVector<int64_t, N> dimVec = {0};
  int64_t batch_dims = 0;

  OpCommand cmd;
  cmd.Name("GatherV2")
     .Input(weight)
     .Input(indices)
     .Input(dimVec)
     .Output(result)
     .Attr("batch_dims", batch_dims)
     .Run();

return result;

}

at::Tensor NPUNativeFunctions::embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  // calculate the output size
  auto outputSize = array_to_small_vector(indices.sizes());
  outputSize.emplace_back(weight.size(weight.dim() - 1));
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      weight.options(),
      CalcuOpUtil::GetTensorNpuFormat(weight));

  // calculate the output resugt of the NPU
  embedding_out_npu_nocheck(result, weight, indices);
  return result;
}
} // namespace native
} // namespace at_npu
