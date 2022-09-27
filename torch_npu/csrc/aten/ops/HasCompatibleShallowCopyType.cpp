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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include <torch/library.h>

namespace at_npu {
namespace native {

// True if `self` and `from` have compatible tensor type so that `from`'s
// TensorImpl can be copied to `self`.
bool _has_compatible_shallow_copy_type(const at::Tensor &self,
                                       const at::Tensor &from) {
  c10::DispatchKeySet self_key = self.key_set();
  c10::DispatchKeySet from_key = from.key_set();
  auto is_dense = [](c10::DispatchKeySet ts) {
    return ts.has(c10::DispatchKey::CPU) ||
           ts.has(at_npu::key::NativeDispatchKey);
  };
  return (self_key == from_key) || (is_dense(self_key) && is_dense(from_key));
}

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl("_has_compatible_shallow_copy_type",
         TORCH_FN(_has_compatible_shallow_copy_type));
}
} // namespace native
} // namespace at_npu