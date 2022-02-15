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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu { 
namespace native {

tuple<at::Tensor &, at::Tensor &, at::Tensor &> layer_norm_backward_npu_nocheck(
    at::Tensor& dX, 
    at::Tensor& dgamma, 
    at::Tensor& dbeta, 
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& variance,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N) 
{
  // constructs the input and output NPUTensorDesc
  at::SmallVector<int64_t, SIZE> tmpSize = array_to_small_vector(X.sizes());
  for (int i = X.dim() - gamma.dim(); i < X.dim(); i++) {
    tmpSize[i] = 1;
  }
  at::Tensor mean_ex = mean.reshape(tmpSize);
  at::Tensor variance_ex = variance.reshape(tmpSize);
  double eps = 1e-05;

  OpCommand cmd;
  cmd.Name("LayerNormGrad")
    .Input(dY)
    .Input(X)
    .Input(variance_ex)
    .Input(mean_ex)
    .Input(gamma)
    .Output(dX)
    .Output(dgamma)
    .Output(dbeta)
    .Run();

  return tuple<at::Tensor &, at::Tensor &, at::Tensor &>(dX, dgamma, dbeta);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_npu_support(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& variance,
    const c10::optional<at::Tensor>& gamma_ex,
    int64_t M,
    int64_t N,
    std::array<bool, 3> output_mask) {
  const at::Tensor& gamma = c10::value_or_else(gamma_ex, [] {return at::Tensor();});
  at::Tensor dX;
  at::Tensor dgamma;
  at::Tensor dbeta;
  at::Tensor gammaTemp = gamma;  
  
  at::SmallVector<int64_t, 8> tmpSize;
  int64_t numels = 1;
  for (int64_t i = X.dim() - 1; i >= 0; i--) {
    numels *= X.size(i);
    tmpSize.emplace_back(X.size(i));
    if(numels == N) {
        break;
    }
  }
  std::reverse(tmpSize.begin(), tmpSize.end());
  if (!gamma.defined()) {
    gammaTemp = at::ones(tmpSize, X.options());
  } else if (!gamma.sizes().equals(tmpSize)) {
    gammaTemp.resize_(tmpSize);
  }
  
  // calculate the output size
  auto outputSizes = layer_norm_backward_npu_output_size(dY, X, mean, variance, gammaTemp, M, N);
  
  if (M <= 0) {
    dX = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    dgamma = at::native::zeros_like(gammaTemp, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    dbeta = at::native::zeros_like(gammaTemp, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
  }  

  // construct the output tensor
  dX = OpPreparation::ApplyTensor(X, std::get<0>(outputSizes));
  dgamma = OpPreparation::ApplyTensor(gammaTemp, std::get<1>(outputSizes));
  dbeta = OpPreparation::ApplyTensor(gammaTemp, std::get<2>(outputSizes));
  
  // calculate the output result of the NPU
  return layer_norm_backward_npu_nocheck(dX, dgamma, dbeta, dY, X, mean, variance, gammaTemp, M, N);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::native_layer_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& variance,
    const c10::optional<at::Tensor>& gamma,
    const c10::optional<at::Tensor>& beta,
    std::array<bool, 3> output_mask) {
  const int normalized_ndim = normalized_shape.size();
  const auto input_shape = X.sizes();
  const auto input_ndim = X.dim();

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
  
  return layer_norm_backward_npu_support(dY, X, mean, variance, gamma, M, N, output_mask);
}

}}  // namespace at_npu::native