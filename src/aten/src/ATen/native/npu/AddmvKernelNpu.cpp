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

Tensor& addmv_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha) {
    
  check_1d(vec, "vec", "addmv");
  
  Tensor mat1 = vec.unsqueeze(1);

  // matmul mat*alpha
  Tensor mat_alpha = at::mul(mat, alpha);

  // matmul*alpha
  Tensor mmMulResult = at::mm(mat_alpha, mat1);
  
  Tensor mmMulResult1 = mmMulResult.squeeze();

  // calculate the output size
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);

  if (!result.sizes().equals(outputSize)) {
    result.resize_(outputSize);
  }
  // matmul*alpha+self*beta
  at::add_out(result, mmMulResult1, self, beta);

  return result;
}

Tensor addmv_npu(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha) {
    
  check_1d(vec, "vec", "addmv");
  // calculate the output size
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  addmv_out_npu(result, self, mat, vec, beta, alpha);

  return result;
}

Tensor& addmv_npu_(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha) {
    
  check_1d(vec, "vec", "addmv");
  SmallVector<Tensor, N> inputs = {self, mat, vec};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        addmv_out_npu(contiguousSelf, contiguousSelf, mat, vec, beta, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addmv_out_npu(self, self, mat, vec, beta, alpha);
  }
  return self;
}

} // namespace native
} // namespace at


