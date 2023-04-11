// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <stdexcept>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CPUFunctions.h>
#include <c10/core/TensorImpl.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"

namespace at_npu {
namespace native {

bool NPUNativeFunctions::is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }

  return false;
}

at::Tensor NPUNativeFunctions::_pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
  // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_npu());
  auto allocator = getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), 
          self.strides(), 
          self.dtype().itemsize()),
      allocator,
      false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace native
} // namespace at_npu
