// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using torch::autograd::Function;

std::vector<at::Tensor> NPUNativeFunctions::npu_flash_attention(
    const at::Tensor &query_layer, const at::Tensor &key_layer,
    const at::Tensor &value_layer, const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask, bool query_transpose, bool key_transpose, bool value_transpose,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, bool is_transpose_out)
{
  return NPUNativeOpApiFunctions::npu_flash_attention(
      query_layer, key_layer, value_layer, pse, drop_mask, padding_mask,
      atten_mask, query_transpose, key_transpose, value_transpose, scale, keep_prob,
      pre_tockens, next_tockens, is_transpose_out);
}
} // namespace native
} // namespace at_npu
