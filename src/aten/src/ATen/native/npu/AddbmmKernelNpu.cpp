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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor& addbmm_out_npu(
    Tensor& result, 
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor MulResult = at::mul(batch1, alpha);
  Tensor bmmResult = at::bmm(MulResult,batch2);
  int64_t dim[2] = {batch1.size(1), batch2.size(2)};
  Tensor sumResult = at::sum_to(bmmResult, dim);
  //sumResult + self*beta
  at::add_out(result, sumResult, self, beta); 
  return result;
}

Tensor addbmm_npu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  // calculate the output size
  auto outputSize = addbmm_npu_output_size(self, batch1, batch2, beta, alpha);
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  // calculate the output result of the NPU
  addbmm_out_npu(result, self, batch1, batch2, beta, alpha);
  return result;
}

Tensor& addbmm_npu_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  SmallVector<Tensor, N> inputs = {self, batch1, batch2};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = addbmm_out_npu(contiguousSelf, contiguousSelf, batch1, batch2, beta, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addbmm_out_npu(self, self, batch1, batch2, beta, alpha);
  }
  return self;
}
} // namespace native
} // namespace at