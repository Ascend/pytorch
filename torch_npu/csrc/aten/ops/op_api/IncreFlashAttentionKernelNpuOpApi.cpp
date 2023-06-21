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

at::Tensor NPUNativeOpApiFunctions::npu_incre_flash_attention(
    const at::Tensor &query,
    at::TensorList key,
    at::TensorList value,
    const c10::optional<at::Tensor> &padding_mask_opt,
    const c10::optional<at::Tensor> &atten_mask_opt,
    at::IntArrayRef actual_seq_lengths,
    int64_t num_heads,
    double scale_value)
{
  const at::Tensor& padding_mask = padding_mask_opt.value_or(at::Tensor());
  const at::Tensor& atten_mask = atten_mask_opt.value_or(at::Tensor());
  at::Tensor format_query = NPUNativeFunctions::npu_format_cast(query, ACL_FORMAT_ND);
  at::Tensor format_padding_mask;
  at::Tensor format_atten_mask;
  if (padding_mask.defined()) {
    format_padding_mask = NPUNativeFunctions::npu_format_cast(padding_mask, ACL_FORMAT_ND);
  }
  if (atten_mask.defined()) {
    format_atten_mask = NPUNativeFunctions::npu_format_cast(atten_mask, ACL_FORMAT_ND);
  }

  at::Tensor attention_out = OpPreparation::ApplyTensor(query);
  uint32_t key_batch_num = key.size();
  uint32_t value_batch_num = value.size();
  TORCH_CHECK(key_batch_num > 0, "dynamic input must be non-empty");
  TORCH_CHECK(key_batch_num == value_batch_num, "dynamic key input num must be equal to value input num");

  uint32_t dim = key[0].dim();
  uint32_t count = 1;

  size_t tensor_desc_size = sizeof(uint64_t) + (sizeof(uint64_t) + dim * sizeof(uint64_t)) * key_batch_num + key_batch_num * sizeof(uint64_t);
  uint64_t tensor_desc_host[1 + (1 + dim) * key_batch_num + key_batch_num] = { 0 };
  uint64_t ptr_offset = sizeof(uint64_t) + (sizeof(uint64_t) + dim * sizeof(uint64_t)) * key_batch_num;
  // append ptr_offset
  tensor_desc_host[0] = ptr_offset;
  for (size_t i = 0; i < key.size(); i++) {
    tensor_desc_host[1 + i * dim ] = (dim) + (count << 32);
    for (size_t j = 0; j < dim; j++) {
      tensor_desc_host[2 + j] = key[i].size(j);
    }
    // append data ptr
    tensor_desc_host[ptr_offset + i] = (uint64_t)(void *)&key[i];
  }

  int deviceIdx = 0;
  NPU_CHECK_ERROR(aclrtGetDevice(&deviceIdx));
  auto key_tensor = at::empty({static_cast<int64_t>(tensor_desc_size)},
                    at::TensorOptions().device(at_npu::key::NativeDeviceType, deviceIdx).dtype(at::kByte));
  aclrtMemcpy(key_tensor.storage().data(), tensor_desc_size, (void*)tensor_desc_host, tensor_desc_size, ACL_MEMCPY_HOST_TO_DEVICE);

  tensor_desc_host[0] = ptr_offset;
  for (size_t i = 0; i < value.size(); i++) {
    // append data ptr
    tensor_desc_host[ptr_offset + i] = (uint64_t)(void *)&value[i];
  }
  auto value_tensor = at::empty({static_cast<int64_t>(tensor_desc_size)},
                      at::TensorOptions().device(at_npu::key::NativeDeviceType, deviceIdx).dtype(at::kByte));
  aclrtMemcpy(value_tensor.storage().data(), tensor_desc_size, (void*)tensor_desc_host, tensor_desc_size, ACL_MEMCPY_HOST_TO_DEVICE);

  EXEC_NPU_CMD(
      aclnnIncreFlashAttention, format_query, key_tensor, value_tensor, format_padding_mask, format_atten_mask, actual_seq_lengths,
      num_heads, scale_value, attention_out);

  return attention_out;
}

} // namespace native
} // namespace at_npu
