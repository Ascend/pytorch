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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeOpApiFunctions::native_layer_norm_backward(
    const at::Tensor& dY, const at::Tensor& X, at::IntArrayRef normalized_shape, const at::Tensor& mean,
    const at::Tensor& variance, const c10::optional<at::Tensor>& gamma, const c10::optional<at::Tensor>& beta,
    std::array<bool, 3> output_mask) {
  DO_COMPATIBILITY(aclnnLayerNormBackward, NPUNativeFunctions::native_layer_norm_backward(dY, X, normalized_shape, mean, variance, gamma, beta, output_mask));
  const at::Tensor& weight = c10::value_or_else(gamma, [] { return at::Tensor(); });
  const at::Tensor& bias = c10::value_or_else(beta, [] { return at::Tensor(); });
  at::Tensor weight_refined =
      weight.defined() ? weight.resize_(normalized_shape) : at::ones(normalized_shape, X.options());
  at::Tensor bias_refined = bias.defined() ? bias.resize_(normalized_shape) : at::zeros(normalized_shape, X.options());

  // 根据输入input和normalized_shape计算M
  const size_t norm_dim = normalized_shape.size();
  const auto input_shape = X.sizes();
  const size_t input_dim = X.dim();
  const size_t begin_axis = input_dim - norm_dim;

  const int64_t M =
      std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());

  at::SmallVector<int64_t, SIZE> mean_shape = array_to_small_vector(X.sizes());
  for (size_t index = begin_axis; index < input_dim; index++) {
    mean_shape[index] = 1;
  }
  at::Tensor mean_refined = mean.reshape(mean_shape);
  at::Tensor variance_refined = variance.reshape(mean_shape);

  // 构造输出tensor
  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  // 根据mask初始化输出tensor
  if (output_mask[0]) {
    grad_input =
        at::native::empty_like(X, c10::nullopt /* dtype */, c10::nullopt /* layout */, c10::nullopt /* device */,
                               c10::nullopt /* pin_memory */, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (output_mask[1]) {
    grad_weight = at::native::zeros_like(weight_refined, at::kFloat /* dtype */, c10::nullopt /* layout */,
                                         c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (output_mask[2]) {
    grad_bias = at::native::zeros_like(bias_refined, at::kFloat /* dtype */, c10::nullopt /* layout */,
                                       c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                       LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // 调用HostAPI接口
  EXEC_NPU_CMD(aclnnLayerNormBackward, dY, X, normalized_shape, mean_refined, variance_refined, weight_refined,
               bias_refined, output_mask, grad_input, grad_weight, grad_bias);
  return std::tie(grad_input, grad_weight, grad_bias);
}

}  // namespace native
}  // namespace at_npu
