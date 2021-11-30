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

Tensor& addr_out_npu(    
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha,
    Tensor& result) {
  NpuUtils::check_1d(vec1, "vec1", "addr");
  NpuUtils::check_1d(vec2, "vec2", "addr");

  Tensor mat1 = vec1.unsqueeze(1);
  Tensor mat2 = vec2.unsqueeze(0);

  // vecmul vec1&vec2
  Tensor mmResult = at::mm(mat1, mat2);

  // matmul*alpha
  Tensor mmMulResult = at::mul(mmResult, alpha);

  // matmul*alpha+self*beta
  at::add_out(result, mmMulResult, self, beta);

  return result;
}

Tensor addr_npu(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  auto outputSize = addr_npu_output_size(self, vec1, vec2, beta, alpha);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addr_out_npu(self, vec1, vec2, beta, alpha, result);

  return result;
}

Tensor& addr_npu_(
    Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  OpPreparation::CheckMemory({self, vec1, vec2}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        addr_out_npu(contiguousSelf, vec1, vec2, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addr_out_npu(self, vec1, vec2, beta, alpha, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("addr", TORCH_FN(addr_npu));
  m.impl("addr_", TORCH_FN(addr_npu_));
  m.impl("addr.out", TORCH_FN(addr_out_npu));
}
} // namespace native
} // namespace at
