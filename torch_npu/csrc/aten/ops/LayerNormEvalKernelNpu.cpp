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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
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
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());
  at::Tensor result = OpPreparation::ApplyTensor(input);
  int64_t numels = 1;
  int64_t begin_dim = 0;
  c10::SmallVector<int64_t, 8> tmpSize;
  for (int64_t i = input.dim() - 1; i >= 0; i--) {
    numels *= input.size(i);
    tmpSize.emplace_back(input.size(i));
    if(numels == N) {
      begin_dim = i;
      break;
    }
  }
  std::reverse(tmpSize.begin(), tmpSize.end());
  at::Tensor resizeWeight = weight;
  resizeWeight.requires_grad_(false);
  at::Tensor resizeBias = bias;
  resizeBias.requires_grad_(false);
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
  at::Tensor mean = OpPreparation::ApplyTensor(resizeWeight, {M});
  at::Tensor rstd = OpPreparation::ApplyTensor(resizeWeight, {M});
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
