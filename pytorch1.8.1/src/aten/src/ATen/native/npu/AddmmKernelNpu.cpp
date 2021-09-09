// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
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
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& addmm_out_npu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha,
    Tensor& result) {
  // mat1*alpha
  Tensor mulResult = at::mul(mat1, alpha);

  // mulmat1 mm mat2
  Tensor mmResult = at::mm(mulResult, mat2);

  // matmul*alpha+self*beta
  at::add_out(result, mmResult, self, beta);

  return result;
}

Tensor addmm_npu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  // calculate the output size
  auto outputSize = addmm_npu_output_size(self, mat1, mat2, beta, alpha);

  // add算子支持NZ与1维且该轴能被16整除的ND相加，直接得到NZ result
  int64_t resFormat = (self.dim() == 1 && self.size(0) % 16 == 0 && self.scalar_type() == at::kHalf) ? 
    ACL_FORMAT_FRACTAL_NZ : 
    ACL_FORMAT_ND;
  Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), resFormat);

  // calculate the output result of the NPU
  addmm_out_npu(self, mat1, mat2, beta, alpha, result);

  return result;
}

Tensor& addmm_npu_(
    Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  SmallVector<Tensor, N> inputs = {self, mat1, mat2};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        addmm_out_npu(contiguousSelf, mat1, mat2, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addmm_out_npu(self, mat1, mat2, beta, alpha, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("addmm", TORCH_FN(addmm_npu));
  m.impl("addmm.out", TORCH_FN(addmm_out_npu));
  m.impl("addmm_", TORCH_FN(addmm_npu_));
}
 
} // namespace native
} // namespace at
