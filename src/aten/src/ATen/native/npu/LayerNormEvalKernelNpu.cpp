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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor layer_norm_eval_npu(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
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

  Tensor Y = at::empty_with_format(input.sizes(), input.options(), CalcuOpUtil::get_tensor_npu_format(input));

  int64_t numels = 1;
  int64_t begin_dim = 0;
  SmallVector<int64_t, 8> tmpSize;
  for (int64_t i = input.dim() - 1; i >= 0; i--) {
    numels *= input.size(i);
    tmpSize.emplace_back(input.size(i));
    if(numels == N) {
      begin_dim = i;
      break;
    }
  }
  std::reverse(tmpSize.begin(), tmpSize.end());

  Tensor resizeWeight = weight;
  Tensor resizeBias = bias;
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

  Tensor mean = at::empty_with_format({M}, resizeWeight.options(), CalcuOpUtil::get_tensor_npu_format(resizeWeight));
  Tensor rstd = at::empty_with_format({M}, resizeWeight.options(), CalcuOpUtil::get_tensor_npu_format(resizeWeight));

  OpCommand cmd;
  cmd.Name("LayerNorm")
    .Input(input)
    .Input(resizeWeight)
    .Input(resizeBias)
    .Output(Y)
    .Output(mean)
    .Output(rstd)
    .Attr("begin_norm_axis", begin_dim)
    .Attr("begin_params_axis", begin_dim)
    .Attr("epsilon", static_cast<float>(eps))
    .Run();

  return Y;
}

}}