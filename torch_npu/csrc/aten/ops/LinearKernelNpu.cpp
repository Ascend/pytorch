// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor linear_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor> & bias_opt) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  c10::SmallVector<int64_t, SIZE> outputSize = {input.size(0), weight.size(0)};
  at::Tensor output = OpPreparation::ApplyTensor(input, outputSize);

  int64_t offset_x = 0;
  OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(output)
      .Attr("transpose_x1", false)
      .Attr("transpose_x2", true)
      .Attr("offset_x", offset_x)
      .Run();

  return output;
}

at::Tensor linear_backward_out_npu(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transpose_x1,
    bool transpose_x2) {
  int64_t offset_x = 0;
  OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight)
      .Output(result)
      .Attr("transpose_x1", transpose_x1)
      .Attr("transpose_x2", transpose_x2)
      .Attr("offset_x", offset_x)
      .Run();
  return result;
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_linear_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight) {
  c10::SmallVector<int64_t, SIZE> inputGradOutputSize = {
      grad.size(0),
      weight.size(1)};
  c10::SmallVector<int64_t, SIZE> weightGradOutputSize = {
      grad.size(1),
      input.size(1)};
  at::Tensor inputGrad = OpPreparation::ApplyTensor(input, inputGradOutputSize);
  at::Tensor weightGrad = OpPreparation::ApplyTensor(weight, weightGradOutputSize);

  if (CalcuOpUtil::GetTensorNpuFormat(grad) == CalcuOpUtil::GetTensorNpuFormat(weight)) {
    linear_backward_out_npu(inputGrad, grad, weight, false, false);
    linear_backward_out_npu(weightGrad, grad, input, true, false);
  } else {
    at::Tensor gradFormatcast = OpPreparation::ApplyTensor(grad, grad.sizes());
    gradFormatcast = NPUNativeFunctions::npu_format_cast(grad, CalcuOpUtil::GetTensorNpuFormat(weight));
    linear_backward_out_npu(inputGrad, gradFormatcast, weight, false, false);
    linear_backward_out_npu(weightGrad, gradFormatcast, input, true, false);
  }

  return std::tie(inputGrad, weightGrad);
}

class NPULinearFunction : public torch::autograd::Function<NPULinearFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
    ctx->saved_data["bias_has_value"] = (bias_opt.has_value() == true) ? bias_opt.value().requires_grad() : false;

    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({input, weight});
    return linear_npu(input, weight, bias_opt);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];

    tuple<at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_linear_backward(grad_outputs[0], input, weight);

    tensor_list output;
    if (bias_has_value) {
      output = {std::get<0>(result),
        std::get<1>(result),
        grad_outputs[0]};
    } else {
      output = {std::get<0>(result),
        std::get<1>(result),
        at::Tensor()};
    }
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_linear(const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  auto isAligin = [&]() {
  return (!(static_cast<uint64_t>(input.size(0)) & 0x0000000F)) &&
          (!(static_cast<uint64_t>(input.size(1)) & 0x0000000F)) &&
          (!(static_cast<uint64_t>(weight.size(0)) & 0x0000000F)) &&
          (!(static_cast<uint64_t>(weight.size(1)) & 0x0000000F));
  };
  static auto mm_bmm_nd = !env::CheckMmBmmNDDisable();
  static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
  at::Tensor input_cast = (FormatHelper::IsBaseFormatType(input) && mm_bmm_nd &&
  ((is_support_nd_out && CalcuOpUtil::IsNdToNzOnTheFly(input, weight)) ||
  (!is_support_nd_out && isAligin()))) ?
      input : NPUNativeFunctions::npu_format_cast(input, ACL_FORMAT_FRACTAL_NZ);
  return NPULinearFunction::apply(input_cast, weight, bias_opt);
}

} // namespace native
} // namespace at_npu