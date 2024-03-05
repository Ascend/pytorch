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
#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::contiguous(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  TORCH_CHECK(
      memory_format == c10::MemoryFormat::Contiguous,
      "NPU contiguous operator only supportted contiguous memory format.", OPS_ERROR(ErrCode::NOT_SUPPORT));
  return self.clone();
}

at::Tensor NPUNativeFunctions::format_contiguous(const at::Tensor &self) {
  return NpuUtils::format_contiguous(self);
}

bool NPUNativeFunctions::is_set_to(const at::Tensor& self, const at::Tensor& src) {
  if (self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() && self.dim() == src.dim() &&
      NPUNativeFunctions::get_storage_size(self) == NPUNativeFunctions::get_storage_size(src) &&
      NPUNativeFunctions::get_npu_format(self) == NPUNativeFunctions::get_npu_format(src)) {
    for (const auto d : c10::irange(self.dim())) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace native
} // namespace at_npu
