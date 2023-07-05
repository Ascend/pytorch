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

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeOpApiFunctions::native_layer_norm(
    const at::Tensor& input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight_ex,
    const c10::optional<at::Tensor>& bias_ex, double eps) {
  DO_COMPATIBILITY(aclnnLayerNorm,
                   NPUNativeFunctions::native_layer_norm(input, normalized_shape, weight_ex, bias_ex, eps));
  const at::Tensor& weight_op = c10::value_or_else(weight_ex, [] { return at::Tensor(); });
  const at::Tensor& bias_op = c10::value_or_else(bias_ex, [] { return at::Tensor(); });
  at::Tensor weight =
      weight_op.defined() ? weight_op.resize_(normalized_shape) : at::ones(normalized_shape, input.options());
  at::Tensor bias =
      bias_op.defined() ? bias_op.resize_(normalized_shape) : at::zeros(normalized_shape, input.options());

  // 构造HostApi接口所需的输出
  auto output = OpPreparation::ApplyTensorWithoutFormat(input);
  at::Tensor mean_out;
  at::Tensor rstd_out;

  const size_t norm_ndim = normalized_shape.size();
  const size_t input_ndim = input.dim();
  const size_t begin_axis = input_ndim - norm_ndim;

  const auto input_shape = input.sizes();

  const int64_t M =
      std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());
  // 根据M是否大于0，决定输出shape的大小
  if (M <= 0) {
    mean_out = OpPreparation::ApplyTensorWithoutFormat({M}, input.options());
    rstd_out = OpPreparation::ApplyTensorWithoutFormat({M}, input.options());
  } else {
    at::SmallVector<int64_t, 8> mean_shape;
    for (size_t index = 0; index < begin_axis; index++) {
      mean_shape.emplace_back(input.size(index));
    }
    for (size_t index = begin_axis; index < input_ndim; index++) {
      mean_shape.emplace_back(1);
    }
    mean_out = OpPreparation::ApplyTensorWithoutFormat(mean_shape, input.options());
    rstd_out = OpPreparation::ApplyTensorWithoutFormat(mean_shape, input.options());
  }
  // 调用HostAPI接口
  EXEC_NPU_CMD(aclnnLayerNorm, input, normalized_shape, weight, bias, eps, output, mean_out, rstd_out);
  return std::tie(output, mean_out, rstd_out);
}

}  // namespace native
}  // namespace at_npu
