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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

int64_t NPUNativeFunctions::npu_change_data_ptr(const at::Tensor& dst, const at::Tensor& src, int64_t offset) {

  TORCH_CHECK(
      offset >= 0,
      "Expect offset equal or greater than zero, got: ", offset);

  const auto& src_scalar_type = src.scalar_type();
  const auto& dst_scalar_type = dst.scalar_type();

  TORCH_CHECK(
      src_scalar_type == dst_scalar_type,
      "Expect src and dst tensors having the same dtype, got: ",
      "src with dtype ", src_scalar_type,
      ", dst with dtype ", dst_scalar_type);
  TORCH_CHECK(
      (src_scalar_type == at::ScalarType::Half) || (dst_scalar_type == at::ScalarType::Float),
      "Only supports src and dst tensors with dtype float32 or float16, got: ", src_scalar_type);
    
  auto dst_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_.storage_sizes_;
  auto src_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_.storage_sizes_;
  int64_t dst_storage_size = c10::multiply_integers(dst_sizes);
  int64_t src_storage_size = c10::multiply_integers(src_sizes);

  TORCH_CHECK(
      offset + dst_storage_size * dst.element_size() <
      src_storage_size * src.element_size(),
      "Offsets overflow, got: ",
      "offset ", offset,
      ", dst storage size ", dst_storage_size,
      ", src storage size ", src_storage_size);

  at::DataPtr aim_data_ptr;
  if (src_scalar_type == at::ScalarType::Float) {
    float* data_ptr = static_cast<float*>(src.storage().data_ptr().get()) + offset;
    aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
  } else {
    at::Half* data_ptr = static_cast<at::Half*>(src.storage().data_ptr().get()) + offset;
    aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
  }
  dst.storage().set_data_ptr(std::move(aim_data_ptr));

  return 0;
}

} // namespace native
} // namespace at_npu
