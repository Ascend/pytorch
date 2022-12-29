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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_npu_support(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_ex,
    const c10::optional<at::Tensor>& bias_ex,
    int64_t M,
    int64_t N,
    double eps) {
  const at::Tensor& weight_ = c10::value_or_else(weight_ex, [] {return at::Tensor();});
  at::Tensor weight = weight_;
  const at::Tensor& bias_ = c10::value_or_else(bias_ex, [] {return at::Tensor();});
  at::Tensor bias = bias_;

  DCHECK_EQ(input.numel(), M * N);
  DCHECK(!weight.defined() || weight.numel() == N);
  DCHECK(!bias.defined() || bias.numel() == N);

  at::Tensor Y = OpPreparation::ApplyTensor(input);
  at::Tensor mean;
  at::Tensor variance;
  if (M < 0) {
    mean = OpPreparation::ApplyTensorWithFormat({M}, input.options(), ACL_FORMAT_ND);
    variance = OpPreparation::ApplyTensorWithFormat({M}, input.options(), ACL_FORMAT_ND);
  } else {
    int64_t numels = 1;
    int64_t begin_dim = 0;
    
    // the output of mean and rstd is Multidimension
    at::SmallVector<int64_t, 8> reduceDims;
    
    // the input of weight is Multidimension
    at::SmallVector<int64_t, 8> weightDims;
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

    at::SmallVector<int64_t, SIZE> ori_weight_shape = array_to_small_vector(weight_.sizes());
    at::SmallVector<int64_t, SIZE> ori_bias_shape = array_to_small_vector(bias_.sizes());

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
    
    mean = OpPreparation::ApplyTensorWithFormat(reduceDims, weight.options(), ACL_FORMAT_ND);
    variance = OpPreparation::ApplyTensorWithFormat(reduceDims, weight.options(), ACL_FORMAT_ND);

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

    weight.resize_(ori_weight_shape);
    bias.resize_(ori_bias_shape);
  }

  mean = mean.reshape({M});
  variance = variance.reshape({M});
        
  return std::tie(Y, mean, variance);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_ex,
    const c10::optional<at::Tensor>& bias_ex,
    double eps) {
  const at::Tensor& weight = c10::value_or_else(weight_ex, [] {return at::Tensor();});
  const at::Tensor& bias = c10::value_or_else(bias_ex, [] {return at::Tensor();});
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

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

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();
  return layer_norm_npu_support(X, gamma, beta, M, N, eps);
}

}}