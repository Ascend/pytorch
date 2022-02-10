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

at::Tensor& NPUNativeFunctions::addmv_out(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha,
    at::Tensor& result) {    
  NpuUtils::check_1d(vec, "vec", "addmv");
  
  at::Tensor mat1 = vec.unsqueeze(1);

  // matmul mat*alpha
  at::Tensor mat_alpha = at::mul(mat, alpha);

  // matmul*alpha
  at::Tensor mmMulResult = at::mm(mat_alpha, mat1);
  
  at::Tensor mmMulResult1 = mmMulResult.squeeze();

  // calculate the output size
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);

  if (!result.sizes().equals(outputSize)) {
    result.resize_(outputSize);
  }
  // matmul*alpha+self*beta
  at::add_out(result, mmMulResult1, self, beta);

  return result;
}

at::Tensor NPUNativeFunctions::addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha) {
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addmv_out(self, mat, vec, beta, alpha, result);

  return result;
}

at::Tensor& NPUNativeFunctions::addmv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    at::Scalar beta,
    at::Scalar alpha) {
  OpPreparation::CheckMemory({self, mat, vec}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result =
        addmv_out(contiguousSelf, mat, vec, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addmv_out(self, mat, vec, beta, alpha, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu


