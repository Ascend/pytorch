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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> logsumexp_npu_output_size(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim) {
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

Tensor& logsumexp_out_npu(Tensor& result, const Tensor& self, DimnameList dims, bool keepdim) {
  return logsumexp_out_npu(result, self, dimnames_to_positions(self, dims), keepdim);
}

Tensor& logsumexp_out_npu(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  return at::native::logsumexp_out(result, self, dims, keepdim);
}

Tensor logsumexp_npu(const Tensor& self, IntArrayRef dims, bool keepdim) {
  auto outputSize = logsumexp_npu_output_size(self, dims, keepdim);
  Tensor result =  at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  return logsumexp_out_npu(result, self, dims, keepdim);
}

Tensor logsumexp_npu(const Tensor& self, DimnameList dims, bool keepdim) {
  return logsumexp_npu(self, dimnames_to_positions(self, dims), keepdim);
}
} // namespace native
} // namespace at
