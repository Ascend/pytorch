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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::npu_prompt_flash_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    int64_t num_heads,
    double scale_value,
    int64_t pre_tokens)
{

  at::Tensor attention_out = OpPreparation::ApplyTensor(query);

  EXEC_NPU_CMD(
      aclnnPromptFlashAttention, query, key, value, padding_mask, atten_mask, actual_seq_lengths,
      num_heads, scale_value, pre_tokens, attention_out);

  return attention_out;
}

} // namespace native
} // namespace at_npu
