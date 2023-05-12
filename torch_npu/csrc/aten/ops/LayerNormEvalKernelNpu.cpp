#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::npu_layer_norm_eval(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt,
    double eps) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const int normalized_ndim = normalized_shape.size();
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int axis = input_ndim - normalized_ndim;
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());
  at::Tensor result = OpPreparation::ApplyTensor(input);
  int64_t numels = 1;
  int64_t begin_dim = 0;
  c10::SmallVector<int64_t, SIZE> tmpSize;
  for (int64_t i = input.dim() - 1; i >= 0; i--) {
    numels *= input.size(i);
    tmpSize.emplace_back(input.size(i));
    if(numels == N) {
      begin_dim = i;
      break;
    }
  }
  std::reverse(tmpSize.begin(), tmpSize.end());
  at::Tensor resizeWeight = weight.detach().clone();
  at::Tensor resizeBias = bias.detach().clone();
  if (!resizeWeight.defined()) {
    resizeWeight = at::ones(tmpSize, input.options());
  } else if (!resizeWeight.sizes().equals(tmpSize)) {
    resizeWeight.resize_(tmpSize);
  }
  if (!resizeBias.defined()) {
    resizeBias = at::zeros(tmpSize, input.options());
  } else if (!resizeBias.sizes().equals(tmpSize)) {
    resizeBias.resize_(tmpSize);
  }

  c10::SmallVector<int64_t, SIZE> output_size;
  for (int64_t i = 0; i < input_ndim; i++) {
    if (i < begin_dim) {
      output_size.emplace_back(input.size(i));
    } else {
      output_size.emplace_back(1);
    }
  }

  at::Tensor mean = OpPreparation::ApplyTensor(resizeWeight, output_size);
  at::Tensor rstd = OpPreparation::ApplyTensor(resizeWeight, output_size);
  OpCommand cmd;
  cmd.Name("LayerNorm")
    .Input(input)
    .Input(resizeWeight)
    .Input(resizeBias)
    .Output(result)
    .Output(mean)
    .Output(rstd)
    .Attr("begin_norm_axis", begin_dim)
    .Attr("begin_params_axis", begin_dim)
    .Attr("epsilon", static_cast<float>(eps))
    .Run();
  return result;
}
}}
