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

Tensor& addmv_out_npu(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha,
    Tensor& result) {    
  NpuUtils::check_1d(vec, "vec", "addmv");
  
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
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addmv_out_npu(self, mat, vec, beta, alpha, result);

  return result;
}

Tensor& addmv_npu_(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha) {
  OpPreparation::CheckMemory({self, mat, vec}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        addmv_out_npu(contiguousSelf, mat, vec, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addmv_out_npu(self, mat, vec, beta, alpha, self);
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("addmv", TORCH_FN(addmv_npu));
  m.impl("addmv_", TORCH_FN(addmv_npu_));
  m.impl("addmv.out", TORCH_FN(addmv_out_npu));
}
} // namespace native
} // namespace at


