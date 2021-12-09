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

#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at { 
namespace native {
using namespace at::native::npu;


tuple<Tensor &, Tensor &, Tensor &> layer_norm_backward_npu_nocheck(
    Tensor& dX, 
    Tensor& dgamma, 
    Tensor& dbeta, 
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    int64_t M,
    int64_t N) 
{
  // constructs the input and output NPUTensorDesc
  SmallVector<int64_t, SIZE> tmpSize = array_to_small_vector(X.sizes());
  for (int i = X.dim() - gamma.dim(); i < X.dim(); i++) {
    tmpSize[i] = 1;
  }
  Tensor mean_ex = mean.reshape(tmpSize);
  Tensor variance_ex = variance.reshape(tmpSize);
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

  return tuple<Tensor &, Tensor &, Tensor &>(dX, dgamma, dbeta);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_npu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> output_mask) 
{
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  Tensor gammaTemp = gamma;  
  
  SmallVector<int64_t, 8> tmpSize;
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

}}  // namespace at::native