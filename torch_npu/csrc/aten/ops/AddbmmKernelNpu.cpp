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

at::Tensor& NPUNativeFunctions::addbmm_out( 
    const at::Tensor& self, 
    const at::Tensor& batch1, 
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha,
    at::Tensor& result) {
  at::Tensor MulResult = at::mul(batch1, alpha);
  at::Tensor bmmResult = at::bmm(MulResult,batch2);
  int64_t dim[2] = {batch1.size(1), batch2.size(2)};
  at::Tensor sumResult = at::sum_to(bmmResult, dim);
  // sumResult + self*beta
  at::add_out(result, sumResult, self, beta); 
  return result;
}

at::Tensor NPUNativeFunctions::addbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  // calculate the output size
  auto outputSize = addbmm_npu_output_size(self, batch1, batch2, beta, alpha);
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  // calculate the output result of the NPU
  addbmm_out(self, batch1, batch2, beta, alpha, result);
  return result;
}

at::Tensor& NPUNativeFunctions::addbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  OpPreparation::CheckMemory({self, batch1, batch2}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = addbmm_out(contiguousSelf, batch1, batch2, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addbmm_out(self, batch1, batch2, beta, alpha, self);
  }
  return self;
}

} // namespace native
} // namespace at