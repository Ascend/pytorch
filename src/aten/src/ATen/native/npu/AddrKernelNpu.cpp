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

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
  TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

Tensor& _addr_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
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

Tensor& addr_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_addr_out(result, self, vec1, vec2, beta, alpha);
}

Tensor _addr_npu(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {

  // calculate the output size
  auto outputSize = addr_npu_output_size(self, vec1, vec2, beta, alpha);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  _addr_out_npu(result, self, vec1, vec2, beta, alpha);

  return result;
}

Tensor addr_npu(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_addr(self, vec1, vec2, beta, alpha);
}

Tensor& _addr_npu_(
    Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  SmallVector<Tensor, N> inputs = {self, vec1, vec2};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        _addr_out_npu(contiguousSelf, contiguousSelf, vec1, vec2, beta, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    _addr_out_npu(self, self, vec1, vec2, beta, alpha);
  }
  return self;
}

Tensor& addr_npu_(
    Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_addr_(self, vec1, vec2, beta, alpha);
}

} // namespace native
} // namespace at
