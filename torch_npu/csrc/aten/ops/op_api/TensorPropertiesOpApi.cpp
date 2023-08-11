// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

bool NPUNativeOpApiFunctions::is_set_to(const at::Tensor& self, const at::Tensor& src) {
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
