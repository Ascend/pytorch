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

at::Tensor& cumsum_out_nocheck(at::Tensor& result, const at::Tensor& self, int64_t dim) {
  OpCommand cmd;
  // if dim = 0, performance in Aicpu is better than Aicore
  // if dim > INT32_MAX, we should use long to store dim for ensuring function correctness.
  // use host memory instead of scalar to improve delivery performance
  at::Scalar dim_scalar(dim);
  cmd.Name("Cumsum")
      .Input(self);
  if (dim == 0 || dim > INT32_MAX) {
    cmd.Input(dim_scalar, at::kLong, CompileType::MEMORY_HOST_COMPILE_DEPENDENT);
  } else {
    cmd.Input(dim_scalar, at::kInt, CompileType::MEMORY_HOST_COMPILE_DEPENDENT);
  }
  cmd.Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::_cumsum_out(const at::Tensor& self, int64_t dim, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      self.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    cumsum_out_nocheck(contiguous_result, self, dim);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    cumsum_out_nocheck(result, self, dim);
  }
  return result;
}

at::Tensor NPUNativeFunctions::_cumsum(const at::Tensor& self, int64_t dim) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  cumsum_out_nocheck(result, self, dim);
  return result;
}

} // namespace native
} // namespace at
