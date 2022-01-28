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

at::Tensor& NPUNativeFunctions::addr_out(    
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha,
    at::Tensor& result) {
  NpuUtils::check_1d(vec1, "vec1", "addr");
  NpuUtils::check_1d(vec2, "vec2", "addr");

  at::Tensor mat1 = vec1.unsqueeze(1);
  at::Tensor mat2 = vec2.unsqueeze(0);

  // vecmul vec1&vec2
  at::Tensor mmResult = at::mm(mat1, mat2);

  // matmul*alpha
  at::Tensor mmMulResult = at::mul(mmResult, alpha);

  // matmul*alpha+self*beta
  at::add_out(result, mmMulResult, self, beta);

  return result;
}

at::Tensor NPUNativeFunctions::addr(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha) {
  auto outputSize = addr_npu_output_size(self, vec1, vec2, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addr_out(self, vec1, vec2, beta, alpha, result);

  return result;
}

at::Tensor& NPUNativeFunctions::addr_(
    at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha) {
  OpPreparation::CheckMemory({self, vec1, vec2}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result =
        addr_out(contiguousSelf, vec1, vec2, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addr_out(self, vec1, vec2, beta, alpha, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu
