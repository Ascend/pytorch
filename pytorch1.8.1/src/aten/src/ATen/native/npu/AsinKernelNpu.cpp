// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& asin_out_npu(
    const Tensor& self,
    Tensor& result) {
  OpCommand cmd;
  cmd.Name("Asin")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

Tensor asin_npu(const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  asin_out_npu(self, result);
  return result;
}

Tensor& asin_npu_(Tensor& self) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = asin_out_npu(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    asin_out_npu(self, self);
  }

  return self;
}
TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("asin_", TORCH_FN(asin_npu_));
  m.impl("asin", TORCH_FN(asin_npu));
  m.impl("asin.out", TORCH_FN(asin_out_npu));
} 
} // namespace native
} // namespace at
