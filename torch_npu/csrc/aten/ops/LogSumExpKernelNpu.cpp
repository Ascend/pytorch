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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> logsumexp_npu_output_size(
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim) {
  return reduce_ops_npu_output_size(self, dims, keepdim);
}

at::Tensor& logsumexp_out_nocheck(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result) {
  return at::native::logsumexp_out(result, self, dims, keepdim);
}

at::Tensor& NPUNativeFunctions::logsumexp_out(const at::Tensor& self, at::DimnameList dims, bool keepdim, at::Tensor& result) {
  return logsumexp_out(self, dimnames_to_positions(self, dims), keepdim, result);
}

at::Tensor& NPUNativeFunctions::logsumexp_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result) {
  auto outputSize = logsumexp_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  return logsumexp_out_nocheck(self, dims, keepdim, result);
}

at::Tensor NPUNativeFunctions::logsumexp(const at::Tensor& self, at::IntArrayRef dims, bool keepdim) {
  auto outputSize = logsumexp_npu_output_size(self, dims, keepdim);
  at::Tensor result =  OpPreparation::ApplyTensor(self, outputSize);
  return logsumexp_out_nocheck(self, dims, keepdim, result);
}

at::Tensor NPUNativeFunctions::logsumexp(const at::Tensor& self, at::DimnameList dims, bool keepdim) {
  return logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

} // namespace native
} // namespace at_npu