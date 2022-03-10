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

std::tuple<Tensor, Tensor, Tensor> layer_norm_npu(
    const Tensor& input,
    const Tensor& weight_ex,
    const Tensor& bias_ex,
    int64_t M,
    int64_t N,
    double eps) {
  Tensor weight = weight_ex;
  Tensor bias = bias_ex;
  DCHECK_EQ(input.numel(), M * N);
  DCHECK(!weight.defined() || weight.numel() == N);
  DCHECK(!bias.defined() || bias.numel() == N);

  Tensor Y = at::empty_with_format(input.sizes(), input.options(), CalcuOpUtil::get_tensor_npu_format(input));
  Tensor mean;
  Tensor variance;
  if (M < 0) {
    mean = at::empty_with_format({M}, input.options());
    variance = at::empty_with_format({M}, input.options());
  } else {
    int64_t numels = 1;
    int64_t begin_dim = 0;
    SmallVector<int64_t, 8> reduceDims;
    SmallVector<int64_t, 8> weightDims;
    for (int64_t i = 0; i < input.dim(); i++) {
      numels *= input.size(i);
      reduceDims.emplace_back(input.size(i));
      if(numels == M){
        begin_dim = i + 1;
        while (++i < input.dim()) {
           reduceDims.emplace_back(1);
           weightDims.emplace_back(input.size(i));
        }
        break;
      }
    }

    if (!weight.defined()) {
      weight = at::ones(weightDims, input.options());
    } else if (!weight.sizes().equals(weightDims)) {
      weight.resize_(weightDims);
    }

    if (!bias.defined()) {
      bias = at::zeros(weightDims, input.options());
    } else if (!bias.sizes().equals(weightDims)) {
      bias.resize_(weightDims);
    }
    
    mean = at::empty_with_format(reduceDims, weight.options());
    variance = at::empty_with_format(reduceDims, weight.options());

    OpCommand cmd;
    cmd.Name("LayerNorm")
      .Input(input)
      .Input(weight)
      .Input(bias)
      .Output(Y)
      .Output(mean)
      .Output(variance)
      .Attr("begin_norm_axis", begin_dim)
      .Attr("begin_params_axis", begin_dim)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();
  }
  Tensor meanResult = mean.reshape({M});
  Tensor varianceResult = variance.reshape({M});
  return std::tie(Y, meanResult, varianceResult);
}

}}